"""
Implementation for Whisper architecture.
"""

import dataclasses
from typing import Any, Dict, List, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WhisperConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Whisper model."""

    vocab_size: int
    num_hidden_layers: int
    num_mel_bins: int
    encoder_layers: int
    encoder_attention_heads: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    encoder_ffn_dim: int
    decoder_start_token_id: int
    d_model: int
    max_source_positions: int
    max_target_positions: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    begin_suppress_tokens: List[int]
    head_dim: int = 0
    max_batch_size: int = 1
    suppress_tokens: List[int] = dataclasses.field(default_factory=list)
    forced_decoder_ids: List[int] = dataclasses.field(default_factory=list)
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert self.tensor_parallel_shards == 1, "TP is not supported right now."
        assert self.num_hidden_layers == self.encoder_layers == self.decoder_layers
        assert self.encoder_attention_heads == self.decoder_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.d_model // self.decoder_attention_heads
        if self.context_window_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("context_window_size"),
                self.max_source_positions,
            )
            self.context_window_size = self.max_source_positions
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                self.max_source_positions,
            )
            self.prefill_chunk_size = self.max_source_positions


# pylint: disable=invalid-name,missing-docstring


class WhisperAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def encoder_self_attn(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ) -> Tensor:
        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape

        q = nn.reshape(self.q_proj(hidden_states), (bsz, q_len, h, d))
        k = nn.reshape(self.k_proj(hidden_states), (bsz, q_len, h, d))
        v = nn.reshape(self.v_proj(hidden_states), (bsz, q_len, h, d))

        attn_output = paged_kv_cache.attention_no_append(layer_id, q, k, v)

        attn_output = nn.reshape(attn_output, hidden_states.shape)  # [b, q_len, h * d]
        attn_output = self.out_proj(attn_output)
        return attn_output

    def decoder_self_attn(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ) -> Tensor:
        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape

        q = nn.reshape(self.q_proj(hidden_states), (bsz, q_len, h, d))
        k = nn.reshape(self.k_proj(hidden_states), (bsz, q_len, h, d))
        v = nn.reshape(self.v_proj(hidden_states), (bsz, q_len, h, d))

        # Todo: Add q scaling to split as a next step before merging qkv
        qkv = op.concat([q, k, v], dim=2)
        attn_output = paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_heads)

        attn_output = nn.reshape(attn_output, hidden_states.shape)  # [b, q_len, h * d]
        attn_output = self.out_proj(attn_output)
        return attn_output

    def decoder_cross_attn(
        self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ) -> Tensor:
        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape

        q = nn.reshape(self.q_proj(hidden_states), (bsz, q_len, h, d))
        attn_output = paged_kv_cache.cross_attention(layer_id, q)

        attn_output = nn.reshape(attn_output, hidden_states.shape)  # [b, q_len, h * d]
        attn_output = self.out_proj(attn_output)
        return attn_output

    def compute_cross_attn_kv(
        self, encoder_hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ) -> PagedKVCache:
        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = encoder_hidden_states.shape

        k = nn.reshape(self.k_proj(encoder_hidden_states), (bsz, q_len, h, d))
        v = nn.reshape(self.v_proj(encoder_hidden_states), (bsz, q_len, h, d))
        return paged_kv_cache.push_cross_attention_kv(layer_id, k, v)


class EncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(self.embed_dim, config.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn.encoder_self_attn(hidden_states, paged_kv_cache, layer_id)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.maximum(tir.min_value(hidden_states.dtype))
        hidden_states = hidden_states.minimum(tir.max_value(hidden_states.dtype))

        return hidden_states


class DecoderLayer(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(self.embed_dim, config.decoder_attention_heads)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(self.embed_dim, config.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn.decoder_self_attn(hidden_states, paged_kv_cache, layer_id)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn.decoder_cross_attn(
            hidden_states, paged_kv_cache, layer_id
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def compute_cross_attn_kv(
        self, encoder_hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int
    ) -> PagedKVCache:
        return self.encoder_attn.compute_cross_attn_kv(
            encoder_hidden_states, paged_kv_cache, layer_id
        )


class WhisperEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: WhisperConfig):
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = 1.0

        self.conv1 = nn.Conv1D(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1D(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_features: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        inputs_embeds = nn.gelu(self.conv1(input_features))
        inputs_embeds = nn.gelu(self.conv2(inputs_embeds))

        inputs_embeds = nn.permute_dims(inputs_embeds, [0, 2, 1])
        embed_pos = self.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos

        for layer_id, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperEmbedding(nn.Embedding):
    """The embedding module specialized for whisper so that
    it can be shared with the final proj_out.
    """

    def proj_out_forward(self, x: nn.Tensor):
        """The proj_out forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = 1.0
        self.embed_tokens = WhisperEmbedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        input_embeds = self.embed_tokens(input_ids)
        input_positions = paged_kv_cache.get_query_positions(
            input_ids.shape[0] * input_ids.shape[1]
        )
        position_embeds = self.embed_positions(input_positions).reshape(*input_embeds.shape)
        hidden_states = input_embeds + position_embeds

        for layer_id, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, paged_kv_cache, layer_id)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

    def compute_cross_attn_kv(
        self, encoder_hidden_states: Tensor, paged_kv_cache: Tensor
    ) -> PagedKVCache:
        for layer_id, decoder_layer in enumerate(self.layers):
            paged_kv_cache = decoder_layer.compute_cross_attn_kv(
                encoder_hidden_states, paged_kv_cache, layer_id
            )
        return paged_kv_cache


class WhisperModel(nn.Module):
    def __init__(self, config: WhisperConfig):
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = WhisperModel(config)
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_encode(self, input_features: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        return self.model.encoder(input_features, paged_kv_cache)

    def batch_compute_cross_attn_kv(
        self, encoder_hidden_states: Tensor, paged_kv_cache: PagedKVCache
    ) -> PagedKVCache:
        return self.model.decoder.compute_cross_attn_kv(encoder_hidden_states, paged_kv_cache)

    def get_logits(self, hidden_states: Tensor):
        op_ext.configure()
        logits = self.model.decoder.embed_tokens.proj_out_forward(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_decoder_forward(
        self,
        input_ids: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ) -> Tensor:
        # Todo: embedding in
        op_ext.configure()
        hidden_states = self.model.decoder(input_ids, paged_kv_cache)
        if logit_positions is not None:
            # Todo: how many dimensions does hidden_states have?
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        return self.get_logits(hidden_states)

    def batch_prefill(
        self, input_ids: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ) -> Tensor:
        return self.batch_decoder_forward(input_ids, paged_kv_cache, logit_positions)

    def batch_decode(self, input_ids: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        return self.batch_decoder_forward(input_ids, paged_kv_cache)

    def prefill(self, input_ids: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        op_ext.configure()

        # Todo: how many dimensions does hidden_states have?
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model.decoder(input_ids, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits

    def decode(self, input_ids: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        op_ext.configure()
        hidden_states = self.model.decoder(input_ids, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.decoder_attention_heads
            // self.config.tensor_parallel_shards,
            num_key_value_heads=self.config.decoder_attention_heads
            // self.config.tensor_parallel_shards,
            head_dim=self.config.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=1,
            rope_theta=1,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        encode_input_ndim = 16000 * 30 // 160
        mod_spec = {
            "batch_encode": {
                "input_features": nn.spec.Tensor(
                    ["batch_size", self.config.num_mel_bins, encode_input_ndim], self.dtype
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_compute_cross_attn_kv": {
                "encoder_hidden_states": nn.spec.Tensor(
                    ["batch_size", self.config.max_source_positions, self.config.d_model],
                    self.dtype,
                ),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_ids": nn.spec.Tensor([1, "seq_len"], "int32"),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_ids": nn.spec.Tensor(["batch_size", 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_ids": nn.spec.Tensor([1, "seq_len"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_ids": nn.spec.Tensor([1, 1], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
