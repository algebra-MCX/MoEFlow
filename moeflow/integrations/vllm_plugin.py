"""vLLM general plugin (entry point `vllm.general_plugins`). Never edits the vLLM install.

Two independent switches:
  * MOEFLOW_RUNNER=1  wraps FusedMoEFactory so MoE layers are built with runner_cls=MoEFlowRunner
                     (primary integration point).
  * MOEFLOW_ALL2ALL=1   wraps maybe_make_prepare_finalize so modular EP kernels use
                     MoEFlowPrepareAndFinalize (alternative transport-only integration).
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

logger = logging.getLogger("moeflow")

# Modules that bind `maybe_make_prepare_finalize` by name at import time (vLLM 0.32.0).
_CALL_SITES = (
    "vllm.model_executor.layers.fused_moe.all2all_utils",
    "vllm.model_executor.layers.fused_moe.oracle.fp8",
    "vllm.model_executor.layers.fused_moe.oracle.unquantized",
    "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.fused_moe.oracle.int8",
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.fused_moe.oracle.w4a8",
    "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8",
    "vllm.model_executor.layers.fused_moe.eep_reconfigure",
    "vllm.model_executor.layers.quantization.modelopt_block_quant",
)


def _make_factory(original):
    def maybe_make_prepare_finalize(
        moe,
        quant_config,
        routing_tables=None,
        allow_new_interface=False,
        use_monolithic=False,
        all2all_manager=None,
        input_dtype=None,
    ):
        pcfg = moe.moe_parallel_config
        if pcfg.use_all2all_kernels and not use_monolithic:
            from vllm.distributed import get_ep_group

            from .vllm_pf import MoEFlowPrepareAndFinalize

            logger.info(
                "moeflow: MoEFlowPrepareAndFinalize (ep_size=%d, local experts=%d)", pcfg.ep_size, moe.num_local_experts
            )
            return MoEFlowPrepareAndFinalize(get_ep_group().device_group, moe.num_local_experts)
        return original(
            moe,
            quant_config,
            routing_tables=routing_tables,
            allow_new_interface=allow_new_interface,
            use_monolithic=use_monolithic,
            all2all_manager=all2all_manager,
            input_dtype=input_dtype,
        )

    maybe_make_prepare_finalize._moeflow_wrapped_pf = True
    return maybe_make_prepare_finalize


_FACTORY_SITES = (
    "vllm.model_executor.layers.fused_moe.layer",
    "vllm.model_executor.layers.fused_moe",
)


def _wrap_factory(original):
    def FusedMoEFactory(*args, **kwargs):  # noqa: N802 - mirrors the vLLM name
        if kwargs.get("runner_cls") is None:
            from .vllm_runner import MoEFlowRunner

            kwargs["runner_cls"] = MoEFlowRunner
        return original(*args, **kwargs)

    FusedMoEFactory._moeflow_wrapped = True
    return FusedMoEFactory


def register_runner() -> None:
    base = importlib.import_module(_FACTORY_SITES[0])
    original = base.FusedMoEFactory
    if getattr(original, "_moeflow_wrapped", False):
        return
    wrapped = _wrap_factory(original)
    for name in _FACTORY_SITES:
        mod = importlib.import_module(name)
        if getattr(mod, "FusedMoEFactory", None) is original:
            mod.FusedMoEFactory = wrapped
    # Model modules imported before the plugin ran hold their own binding.
    for name in ("vllm.models.glm5next.common.model",):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "FusedMoEFactory", None) is original:
            mod.FusedMoEFactory = wrapped
    logger.warning("moeflow: FusedMoEFactory wrapped (runner_cls=MoEFlowRunner)")


def register() -> None:
    if os.environ.get("MOEFLOW_RUNNER") == "1":
        register_runner()
    if os.environ.get("MOEFLOW_ALL2ALL") == "1":
        register_prepare_finalize()


def register_prepare_finalize() -> None:
    base = importlib.import_module(_CALL_SITES[0])
    original = base.maybe_make_prepare_finalize
    if getattr(original, "_moeflow_wrapped_pf", False):
        return
    wrapped = _make_factory(original)
    patched = []
    for name in _CALL_SITES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        current = getattr(mod, "maybe_make_prepare_finalize", None)
        if current is original:
            mod.maybe_make_prepare_finalize = wrapped
        if current is original or current is wrapped:
            patched.append(name.rsplit(".", 1)[-1])
    logger.warning("moeflow: patched maybe_make_prepare_finalize in %s", ", ".join(patched))
