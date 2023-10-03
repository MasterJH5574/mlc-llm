from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te, tir
from tvm.relax.op import ccl
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList
from .param_manager import ParamManager


@dataclass
class LlamaConfig:
    def __init__(
        self,
        dtype="float32",
        max_sequence_length=2048,
        vocab_size=32000,  # some models like WizardMath can have 32001
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        position_embedding_base=10000,
        combine_matmul=True,
        build_model_only=False,
        num_shards=1,
        **kwargs,
    ):
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_base = position_embedding_base
        self.combine_matmul = combine_matmul
        if build_model_only and num_shards > 1:
            self.num_shards = num_shards
        else:
            self.num_shards = 1
        self.kwargs = kwargs

    def get_num_key_value_heads(self):
        if self.num_key_value_heads is None:
            return self.num_attention_heads

        return self.num_key_value_heads


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), dtype=dtype, name="linear_weight")
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype: str):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="embedding_weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        from tvm.relax.op import reshape, take

        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = self.weight.struct_info.shape.values[-1]
            x = nn.emit(reshape(x, shape=[-1]))
            embedding = nn.emit(take(self.weight, x, axis=0))
            return nn.emit(reshape(embedding, [*x_shape, emb_size]))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-6):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="rms_norm_weight")
        self.variance_epsilon = tvm.tir.const(eps, dtype)

    def forward(self, hidden_states):
        from tvm import te, tir

        def f_rms_norm(x, weight):
            is_float32 = x.dtype == "float32"

            def f_square(x):
                return tir.Cast("float32", x) * tir.Cast("float32", x) if not is_float32 else x * x

            k = te.reduce_axis((0, x.shape[2]), name="k")
            square_sum = te.compute(
                (x.shape[0], x.shape[1]),
                lambda bsz, i: te.sum(f_square(x[bsz, i, k]), axis=k),
                name=x.op.name + "red_temp",
            )

            def f_div_cast(bsz, i, k):
                x_val = x[bsz, i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(square_sum[bsz, i] / x.shape[2] + self.variance_epsilon)

            def f_mul_cast(x, y):
                value = x * y
                if not is_float32:
                    value = tir.Cast(x.dtype, value)
                return value

            return te.compute(
                x.shape,
                lambda bsz, i, k: f_mul_cast(weight(k), f_div_cast(bsz, i, k)),
                name="rms_norm",
            )

        return nn.emit_te(f_rms_norm, hidden_states, self.weight, primfunc_name_hint="rms_norm")


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.combine_matmul = config.combine_matmul
        self.num_shards = config.num_shards
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // self.num_shards
        dtype = config.dtype
        if self.combine_matmul:
            self.gate_up_proj = Linear(hidden_size, 2 * intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.gate_up_proj.weight.shard_dim = 0
            self.gate_up_proj.weight.shard_strategy = "shard_gate_up"
            self.down_proj.weight.shard_dim = 1
            self.down_proj.weight.shard_strategy = "shard_mlp_k"
        else:
            self.gate_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)

    def forward(self, x):
        if self.combine_matmul:
            gate_up_results = nn.emit(
                relax.op.split(
                    self.gate_up_proj(x),
                    indices_or_sections=2,
                    axis=-1,
                )
            )
            gate_result = relax.TupleGetItem(gate_up_results, 0)
            up_result = relax.TupleGetItem(gate_up_results, 1)
        else:
            gate_result = self.gate_proj(x)
            up_result = self.up_proj(x)

        result = self.down_proj(relax.op.nn.silu(gate_result) * up_result)
        return result


def apply_rotary_pos_emb(q, k, position_embedding_base, offset: int = 0):
    def f_rotary_embedding(tensor, offset):
        dtype = tensor.dtype
        head_dim = tensor.shape[-1]
        n_feat_half = tensor.shape[-1] // 2

        def rotary_compute(*idx):
            i, j = idx[-3], idx[-1]
            pos = (offset + i).astype("float32")
            inv_freq = te.const(1, "float32") / (
                te.power(
                    te.const(position_embedding_base, "float32"),
                    ((2 * j) % head_dim).astype("float32") / head_dim.astype("float32"),
                )
            )
            freq = pos * inv_freq
            return te.cos(freq).astype(dtype) * tensor(*idx) + te.sin(freq).astype(
                dtype
            ) * tvm.tir.Select(
                j >= n_feat_half,
                tensor[idx[0], i, idx[2], j - n_feat_half],
                -tensor[idx[0], i, idx[2], j + n_feat_half],
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, offset, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, offset, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, prefill: bool):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.position_embedding_base = config.position_embedding_base

        self.combine_matmul = config.combine_matmul
        if self.combine_matmul:
            self.query_key_value_proj = Linear(
                self.hidden_size,
                (self.num_query_heads + 2 * self.num_key_value_heads) * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.query_key_value_proj.weight.shard_dim = 0
            self.query_key_value_proj.weight.shard_strategy = "shard_qkv"
        else:
            self.q_proj = Linear(
                self.hidden_size,
                self.num_query_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.k_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.v_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.q_proj.weight.shard_dim = 0
            self.k_proj.weight.shard_dim = 0
            self.v_proj.weight.shard_dim = 0

        self.o_proj = Linear(
            self.head_dim * self.num_query_heads, self.hidden_size, dtype=dtype, bias=False
        )
        self.o_proj.weight.shard_dim = 1
        self.o_proj.weight.shard_strategy = "shard_o_proj_k"

        ctx_mod = relax.BlockBuilder.current().get()
        self.kv_cache_transpose_append = ctx_mod.get_global_var("kv_cache_transpose_append")
        self.attention_compute = ctx_mod.get_global_var(
            "attention_prefill" if prefill else "attention_decode"
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        past_key_values: relax.Expr,
        layer_id: int,
    ) -> Tuple[relax.Expr, relax.Expr]:
        from tvm.relax.op import reshape, split

        bsz, q_len, _ = hidden_states.struct_info.shape
        # assert bsz == 1, "Only support batch size 1 at this moment."

        if self.combine_matmul:
            qkv_states = nn.emit(
                split(
                    self.query_key_value_proj(hidden_states),
                    indices_or_sections=[
                        self.num_query_heads * self.head_dim,
                        (self.num_query_heads + self.num_key_value_heads) * self.head_dim,
                    ],
                    axis=-1,
                )
            )
            query_states = relax.TupleGetItem(qkv_states, 0)
            key_states = relax.TupleGetItem(qkv_states, 1)
            value_states = relax.TupleGetItem(qkv_states, 2)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = nn.emit(
            reshape(
                query_states,
                (bsz, q_len, self.num_query_heads, self.head_dim),
            ),
        )
        key_states = nn.emit(
            reshape(
                key_states,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )
        value_states = nn.emit(
            reshape(
                value_states,
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
            ),
        )

        f_kv_cache_append = relax.extern("vm.builtin.paged_attention_kv_cache_append")
        past_key_values = nn.emit(
            relax.call_pure_packed(
                f_kv_cache_append,
                past_key_values,
                self.kv_cache_transpose_append,
                key_states,
                value_states,
                relax.PrimValue(layer_id),
                sinfo_args=relax.ObjectStructInfo(),
            )
        )

        f_kv_cache_attention = relax.extern("vm.builtin.paged_attention_kv_cache_attention")
        attn_output = nn.emit(
            relax.call_dps_packed(
                f_kv_cache_attention,
                [
                    past_key_values,
                    self.attention_compute,
                    query_states,
                    relax.PrimValue(layer_id),
                    True,
                    1.0,
                    self.position_embedding_base,
                ],
                out_sinfo=relax.TensorStructInfo(
                    ((bsz, q_len, self.num_query_heads, self.head_dim)),
                    hidden_states.struct_info.dtype,
                ),
            )
        )
        attn_output = nn.emit(
            reshape(attn_output, (bsz, q_len, self.head_dim * self.num_query_heads))
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_values


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, prefill: bool):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, prefill)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        past_key_values: relax.Expr,
        layer_id: int,
    ) -> Tuple[relax.Expr, relax.Expr]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            layer_id=layer_id,
        )
        if self.self_attn.num_shards > 1:
            residual = nn.emit(
                residual / R.const(self.self_attn.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if self.self_attn.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp.num_shards > 1:
            residual = nn.emit(
                residual / R.const(self.mlp.num_shards, dtype=residual.struct_info.dtype)
            )
        hidden_states = nn.emit(residual + hidden_states)
        if self.mlp.num_shards > 1:
            hidden_states = nn.emit(ccl.allreduce(hidden_states, "sum"))
        return hidden_states, present_key_value


class LlamaEmbedTokens(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tir.Var):
        self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class LlamaEmbedTokensWrapper(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tir.Var):
        # build a wrapper to ensure that the naming of the embed_tokens parameter is consistent
        self.model = LlamaEmbedTokens(config, vocab_size_var)

    def forward(self, input_ids: relax.Expr):
        inputs_embeds = self.model(input_ids)
        return inputs_embeds


class LlamaModel(nn.Module):
    def __init__(
        self, config: LlamaConfig, vocab_size_var: tir.Var, prefill: bool, sep_embed: bool = False
    ):
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [LlamaDecoderLayer(config, prefill) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: relax.Expr,
        seq_lengths: Optional[relax.Expr],
        past_key_values: relax.Expr,
    ):
        if self.num_shards > 1:
            inputs = nn.emit(ccl.broadcast_from_worker0(inputs))
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        f_kv_cache_prepare = relax.extern("vm.builtin.paged_attention_kv_cache_prepare")
        cache_prepare_args = [past_key_values]
        if seq_lengths is not None:
            cache_prepare_args.append(seq_lengths)
        past_key_values = nn.emit(
            relax.call_pure_packed(
                f_kv_cache_prepare,
                *cache_prepare_args,
                sinfo_args=relax.ObjectStructInfo(),
            )
        )

        for idx, decoder_layer in enumerate(self.layers):
            assert past_key_values is not None
            hidden_states, past_key_values = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                layer_id=idx,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tir.Var,
        prefill: bool,
        sep_embed: bool = False,
    ):
        self.model = LlamaModel(config, vocab_size_var, prefill, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads

        # Set the cached sin/cos to the maximum of 2048 and max seq len.
        # This will be eliminated further with online rotary embedding calculation.
        cache_len = te.var("cache_len", "int64")
        self.cos_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="cos_cached")
        self.sin_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="sin_cached")
        ############ End ############

    def forward(
        self,
        inputs: relax.Expr,
        seq_lengths: Optional[relax.Expr],
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.model(
            inputs=inputs,
            seq_lengths=seq_lengths,
            past_key_values=past_key_values,
        )

        def te_slicing(x: te.Tensor):
            return te.compute(
                shape=(1, 1, x.shape[-1]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )

        logits = self.lm_head(nn.emit_te(te_slicing, hidden_states, primfunc_name_hint="slice"))
        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "embed_tokens" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_embed_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "embed"

    bsz = tir.Var("nseq", "int64")
    seq_len = tir.Var("n", "int64")
    with bb.function(func_name):
        model = LlamaEmbedTokensWrapper(config, tir.Var("vocab_size", "int64"))
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        with bb.dataflow():
            inputs_embeds = model(input_ids)
            params = [input_ids] + model.parameters()
            gv = bb.emit_output(inputs_embeds)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 1))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    # assert sep_embed
    sep_embed = True
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    bsz = 1
    total_seq_len = tir.Var("n", "int64")
    seq_lengths = relax.Var("seq_lengths", relax.ShapeStructInfo())

    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, tir.Var("vocab_size", "int64"), prefill=True, sep_embed=sep_embed
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder(
                (bsz, total_seq_len, config.hidden_size), dtype=config.dtype, name="inputs_embeds"
            )
            if sep_embed
            else nn.Placeholder((bsz, total_seq_len), dtype="int32", name="input_ids")
        )
        past_key_values = relax.Var("kv_cache", relax.ObjectStructInfo())
        with bb.dataflow():
            logits, key_value_cache = model(inputs, seq_lengths, past_key_values=past_key_values)
            params = [inputs, seq_lengths, past_key_values] + model.parameters()
            gv = bb.emit_output((logits, key_value_cache))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"
    sep_embed = False

    bsz = tir.Var("nseq", "int64")
    seq_lengths = None

    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, tir.Var("vocab_size", "int64"), prefill=False, sep_embed=sep_embed
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder((bsz, 1, config.hidden_size), dtype=config.dtype, name="inputs_embeds")
            if sep_embed
            else nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        )
        past_key_values = relax.Var("kv_cache", relax.ObjectStructInfo())
        with bb.dataflow():
            logits, key_value_cache = model(inputs, seq_lengths, past_key_values=past_key_values)
            params = [inputs, past_key_values] + model.parameters()
            gv = bb.emit_output((logits, key_value_cache))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 2))


def create_kv_cache_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_heads = config.get_num_key_value_heads() // config.num_shards

    page_size = tir.Var("page_size", "int64")
    total_seq_len = tir.Var("total_seq_len", "int64")
    reserved_nseq = tir.Var("reserved_nseq", "int64")
    cache_config = relax.Var(
        "cache_config",
        relax.ShapeStructInfo([reserved_nseq, total_seq_len, page_size]),
    )

    with bb.function("create_kv_cache", [cache_config]):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros((), config.dtype))
            f_kv_cache_create = relax.extern("vm.builtin.paged_attention_kv_cache_create")
            cache = bb.emit_output(
                relax.Call(
                    f_kv_cache_create,
                    args=[
                        cache_config,
                        relax.PrimValue(config.num_hidden_layers),
                        relax.PrimValue(num_key_value_heads),
                        relax.PrimValue(head_dim),
                        zeros,
                    ],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
        bb.emit_func_output(cache)


def create_softmax_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, tir.Var("vocab_size", "int64")), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def emit_kv_cache_op(bb: relax.BlockBuilder, dtype: str) -> None:
    from tvm.script import tir as T

    # fmt: off
    @T.prim_func
    def kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_page_table_indptr: T.handle,
        var_page_table_values: T.handle,
        var_last_page_offset: T.handle,
        var_append_length_indptr: T.handle,
        var_pos2seqidx: T.handle,
        layer_id: T.int32,
    ):
        nseq = T.int32()
        ntoken = T.int32()
        nhead = T.int32()
        nfeat = T.int32()
        nlayer = T.int32()
        npage = T.int32()
        page_size = T.int32()
        num_page_chunks = T.int32()
        page_chunk_size = T.int32()

        pages = T.match_buffer(var_pages, (num_page_chunks, nlayer, page_chunk_size, 2, nhead, page_size, nfeat), dtype)
        k_data = T.match_buffer(var_k_data, (ntoken, nhead, nfeat), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, nhead, nfeat), dtype)
        last_page_offset = T.match_buffer(var_last_page_offset, (nseq,), "int32")
        page_table_indptr = T.match_buffer(var_page_table_indptr, (nseq + 1,), "int32")
        page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
        append_length_indptr = T.match_buffer(var_append_length_indptr, (nseq + 1,), "int32")
        pos2seqidx = T.match_buffer(var_pos2seqidx, (ntoken,), "int32")

        for global_pos, h, f in T.grid(ntoken, nhead, nfeat):
            with T.block("k_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx = pos2seqidx[vgpos]
                seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
                pages[
                    T.floordiv(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                    layer_id,
                    T.floormod(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                    0,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = k_data[vgpos, vh, vf]
            with T.block("v_transpose_append"):
                vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                seq_idx = pos2seqidx[vgpos]
                seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
                pages[
                    T.floordiv(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                    layer_id,
                    T.floormod(page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)], page_chunk_size),
                    1,
                    vh,
                    T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                    vf,
                ] = v_data[vgpos, vh, vf]
    # fmt: on

    bb.add_func(kv_cache_transpose_append, "kv_cache_transpose_append")
    bb.add_func(relax.extern("FlashInferBatchPrefillWithPagedKVCache"), "attention_prefill")
    bb.add_func(relax.extern("FlashInferBatchDecodeWithPagedKVCache"), "attention_decode")


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = args.sep_embed

    position_embedding_base = 10000
    max_position_embeddings = 2048
    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]
    if "max_position_embeddings" in hf_config:
        max_position_embeddings = hf_config["max_position_embeddings"]

    config = LlamaConfig(
        **hf_config,
        dtype=dtype,
        position_embedding_base=position_embedding_base,
        combine_matmul=True,
        num_shards=args.num_shards,
        build_model_only=args.build_model_only,
    )
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    emit_kv_cache_op(bb, dtype)

    if sep_embed:
        create_embed_func(bb, param_manager, config, args.quantization)
    create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, args.quantization)
    create_kv_cache_func(bb, config)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model_name,
        max_window_size=config.max_sequence_length,
        stop_tokens=[2],
        add_prefix_space=False,
    )

    # ext_mod = tvm.runtime.load_static_library(
    #     "/home/ruihangl/flashinfer/build/CMakeFiles/tvm_binding.dir/src/tvm_wrapper.cu.o",
    #     ["FlashInferBatchPrefillWithPagedKVCache", "FlashInferBatchDecodeWithPagedKVCache"],
    # )

    mod = bb.get()
    # mod = mod.with_attr("external_mods", [ext_mod])
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr(
                "tir_var_upper_bound",
                {
                    "n": config.max_sequence_length,
                    "m": config.max_sequence_length,
                },
            )

    if args.build_model_only:
        return mod, param_manager, None, config

    def f_convert_pname_fwd(pname: str) -> List[str]:
        if not config.combine_matmul:
            return [pname]

        qkv_str = "query_key_value_proj"
        gate_up_str = "gate_up_proj"
        if qkv_str in pname:
            return [
                pname.replace(qkv_str, "q_proj"),
                pname.replace(qkv_str, "k_proj"),
                pname.replace(qkv_str, "v_proj"),
            ]
        elif gate_up_str in pname:
            return [
                pname.replace(gate_up_str, "gate_proj"),
                pname.replace(gate_up_str, "up_proj"),
            ]
        else:
            return [pname]

    def f_convert_param_bkwd(torch_pname: str, torch_param):
        if not config.combine_matmul:
            return [(torch_pname, torch_param.astype(dtype))]

        combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        if any([name in torch_pname for name in combined_layers]):
            return None
        return [(torch_pname, torch_param.astype(dtype))]

    def f_compute_relax_param(relax_pname: str, torch_params: List[Any]):
        # Expected to enter this function only for the combined linear matmul weights.
        # Other weights are supposed to be loaded in `f_convert_param_bkwd` since
        # each other relax param has a unique corresponding torch param.
        if not config.combine_matmul:
            # When matmul combination is not turned on, each relax param has a unique
            # corresponding torch param, and this function is not expected to be entered.
            raise NotImplementedError(
                "Matmul combination is not turned on, and the function "
                "is not expected to be entered"
            )
        hidden_size = config.hidden_size
        head_dim = config.hidden_size // config.num_attention_heads

        if "query_key_value_proj" in relax_pname:
            q_heads = config.num_attention_heads
            kv_heads = config.get_num_key_value_heads()
            q, k, v = torch_params
            assert q.shape == (q_heads * head_dim, hidden_size)
            assert k.shape == (kv_heads * head_dim, hidden_size)
            assert v.shape == (kv_heads * head_dim, hidden_size)
            qkv = np.concatenate([q, k, v], axis=0).astype(dtype)
            return qkv
        if "gate_up_proj" in relax_pname:
            gate, up = torch_params
            gate_up = np.concatenate([gate, up], axis=0).astype(dtype)
            return gate_up
        raise ValueError("Unexpected param loading")

    param_manager.set_param_loading_func(
        args.model_path,
        args.use_safetensors,
        f_convert_pname_fwd,
        f_convert_param_bkwd,
        f_compute_relax_param,
    )

    device = tvm.cpu()
    param_list = [None] * param_manager.nparam_to_load

    head_dim = config.hidden_size / config.num_attention_heads
    inv_freq = 1.0 / (
        config.position_embedding_base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
    )

    # The following cos/sin values can be removed but **are kept for compatibility issues**.
    t = np.arange(2048, dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    param_list[-2] = tvm.nd.array(np.cos(emb).astype(config.dtype), device)
    param_list[-1] = tvm.nd.array(np.sin(emb).astype(config.dtype), device)

    return mod, param_manager, param_list, config
