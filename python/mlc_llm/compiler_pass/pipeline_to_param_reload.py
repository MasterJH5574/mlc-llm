"""A compiler pass that rewrites IR for pipeline parallelism."""

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import relax, tir
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor


@tvm.transform.module_pass(opt_level=0, name="PipelineParallelRewrite")
class PipelineToParamLoadRewrite:  # pylint: disable=too-few-public-methods
    """A compiler pass that rewrites IR for pipeline parallelism."""

    def transform_module(
        self,
        mod: IRModule,
        _ctx: tvm.transform.PassContext,
    ) -> IRModule:
        """IRModule-level transformation"""
        return _PipelineToParamReloadRewriter(mod.clone()).transform()


@mutator
class _PipelineToParamReloadRewriter(PyExprMutator):  # pylint: disable=abstract-method
    def __init__(self, mod: IRModule):
        super().__init__(mod)
        self.mod = mod
        self.is_first_binding: bool
        self.original_packed_params_var: relax.Var
        self.last_stage_packed_params_var: relax.Var
        self.state_cnt: int

    def transform(self) -> IRModule:  # pylint: disable=too-many-locals
        """Entry point of the transformation"""
        for g_var, func in self.mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            if "num_input" not in func.attrs:
                continue

            print(f"processing function {g_var.name_hint}")
            num_input = int(func.attrs["num_input"])
            assert (
                len(func.params) == num_input + 1
                and isinstance(func.params[num_input], relax.Var)
                and func.params[num_input].name_hint == "packed_params"
            ), 'Only the extra "packed_params" parameter is allowed'
            self.original_packed_params_var = func.params[num_input]
            self.last_stage_packed_params_var = self.original_packed_params_var

            assert isinstance(func.body, relax.SeqExpr)
            assert len(func.body.blocks) == 1
            assert isinstance(func.body.blocks[0], relax.DataflowBlock)
            self.is_first_binding = True
            self.state_cnt = 0
            updated_func = self.visit_expr(func)
            self.builder_.update_func(g_var, updated_func)

        return self.builder_.finalize()

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if self.is_first_binding:
            assert self.state_cnt == 0
            new_param_var = self.builder_.emit(
                relax.call_pure_packed(
                    "mlc.fetch_params",
                    sinfo_args=self.original_packed_params_var.struct_info,
                )
            )
            self.last_stage_packed_params_var = new_param_var
            self.set_var_remap(self.original_packed_params_var.vid, new_param_var)
            self.is_first_binding = False
        super().visit_var_binding_(binding)

    def visit_call_(self, call: relax.Call) -> relax.Call:  # pylint: disable=arguments-renamed
        call = super().visit_call_(call)
        if (
            call.op == tvm.ir.Op.get("relax.call_pure_packed")
            and call.args[0].global_symbol == "mlc.pipeline_parallel_stage_boundary"
        ):
            self._insert_param_reload()
            assert len(call.args) >= 2
            ret = call.args[1:]
            return relax.Tuple(ret) if len(ret) > 1 else ret[0]
        return call

    def _insert_param_reload(self) -> None:
        self.state_cnt += 1
        new_param_var = self.builder_.emit(
            relax.call_pure_packed(
                "mlc.reload_params_on_stage",
                relax.PrimValue(self.state_cnt),
                sinfo_args=self.original_packed_params_var.struct_info,
            )
        )
        self.last_stage_packed_params_var = new_param_var
        self.set_var_remap(self.original_packed_params_var.vid, new_param_var)
