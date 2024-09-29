"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Tensor
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode


class PagedKVCache(TVMPagedKVCache):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(  # pylint: disable=too-many-locals
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        rotary_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_ext_factors: Optional[List[int]] = None,
        layer_partition: Optional[List[int]] = None,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = head_dim
        if rope_scaling is None:
            rope_scaling = {}
        if layer_partition is None:
            layer_partition = [0, num_hidden_layers]
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
                rx.ShapeExpr(
                    [
                        max_batch_size,
                        max_total_seq_len,
                        prefill_chunk_size,
                        page_size,
                        support_sliding_window,
                    ]
                ),
                rx.ShapeExpr(layer_partition),
                rx.PrimValue(num_hidden_layers),
                rx.PrimValue(num_attention_heads),
                rx.PrimValue(num_key_value_heads),
                rx.PrimValue(head_dim),
                rx.PrimValue(rope_mode),
                rx.PrimValue(rope_scale),
                rx.PrimValue(rope_theta),
                rx.StringImm(json.dumps(rope_scaling)),
                (
                    rx.const(np.array(rope_ext_factors, "float32"))
                    if rope_ext_factors is not None
                    else rx.PrimValue(0)
                    # NOTE: since relax does not have "Optional" type, we use PrimValue(0)
                    # to represent "undefined".
                ),
                rx.PrimValue(rotary_dim),
                rx.DataTypeImm(dtype),
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )

    def append_kv_with_output(
        self, layer_id: int, k: Tensor, v: Tensor, context_length: int, output: Tensor
    ) -> Tensor:
        """"""
        b, s, length, d = k._expr.struct_info.shape
        k = k.reshape(b * s, length, d)
        v = v.reshape(b * s, length, d)
        bb = rx.BlockBuilder.current()
        return Tensor(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.attention_kv_cache_append_kv_with_output",
                    self._expr,
                    rx.PrimValue(layer_id),
                    k._expr,
                    v._expr,
                    rx.PrimValue(context_length),
                    output._expr,
                    sinfo_args=output._expr.struct_info,
                ),
                name_hint="output",
            )
        )

    def get_compact_kv(
        self,
        seq_id: tir.PrimExpr,
        layer_id: int,
        batch_size: tir.PrimExpr,
        length: tir.PrimExpr,
        num_kv_heads: int,
        head_dim: int,
        dtype: str,
    ) -> Tuple[Tensor, Tensor]:
        """Get the compact full KV of the running sequnece.
        Limitation: it's expected that only one sequence is running.
        """
        bb = rx.BlockBuilder.current()
        kv_sinfo = rx.TensorStructInfo([batch_size, length, num_kv_heads, head_dim], dtype)
        kv = bb.emit(
            rx.call_dps_packed(
                "vm.builtin.attention_kv_cache_debug_get_kv",
                args=[
                    self._expr,
                    rx.PrimValue(seq_id),
                    rx.PrimValue(layer_id),
                    rx.PrimValue(0),
                    length,
                ],
                out_sinfo=[kv_sinfo, kv_sinfo],
            ),
            name_hint="fetched_kv",
        )
        return Tensor(_expr=bb.emit(kv[0])), Tensor(_expr=bb.emit(kv[1]))

    def get_last_qkv(
        self,
        batch_size: tir.PrimExpr,
        length: tir.PrimExpr,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: str,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bb = rx.BlockBuilder.current()
        q_sinfo = rx.TensorStructInfo([batch_size * length, num_qo_heads, head_dim], dtype)
        kv_sinfo = rx.TensorStructInfo([batch_size * length, num_kv_heads, head_dim], dtype)
        qkv = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_debug_get_last_qkv",
                self._expr,
                sinfo_args=[q_sinfo, kv_sinfo, kv_sinfo],
            ),
            name_hint="last_qkv",
        )
        q = Tensor(_expr=bb.emit(qkv[0]))
        k = Tensor(_expr=bb.emit(qkv[1]))
        v = Tensor(_expr=bb.emit(qkv[2]))
        return (
            q.reshape(batch_size, length, num_qo_heads, head_dim),
            k.reshape(batch_size, length, num_kv_heads, head_dim),
            v.reshape(batch_size, length, num_kv_heads, head_dim),
        )
