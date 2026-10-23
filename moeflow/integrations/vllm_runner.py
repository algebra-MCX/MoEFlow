"""MoEFlowRunner: a vLLM MoERunner subclass injected through FusedMoEFactory(runner_cls=...).

Modes (env MOEFLOW_MODE):
  route_check (default, smoke test): project router instead of vLLM's, cross-checked on live
      data, then vLLM's own modular kernel (including vLLM's dispatch/combine).
  fixed: MoEFlow expert-parallel execution. While the largest per-DP-rank token count of the
      step is <= MOEFLOW_MAX_TOKENS (decode), tokens travel through the fixed-capacity,
      CUDA-graph-capturable path (moeflow.fixed: 2 all-to-alls, no host sync) and the experts run
      through this layer's own vLLM fused-experts object (FP8/BF16) wrapped in a local,
      no-communication Prepare/Finalize. Larger steps (prefill) use vLLM's path unchanged.
The path decision uses DPMetadata.num_tokens_across_dp_cpu, which is identical on all ranks, so
every rank issues the same collectives. Monolithic expert kernels are never intercepted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from ..config import MoEConfig
from ..router import route

_MAX_LOGGED_CALLS_PER_LAYER = 8
_BUFFER_POOL: dict = {}  # Compatible layers run one at a time and share fixed communication buffers.


def _shared_fixed_layer(cfg: MoEConfig, rank: int, world: int, capacity: int, device, group, fused_dispatch=False):
    from ..fixed import FixedEPLayer

    key = (cfg, rank, world, capacity, str(device), group, fused_dispatch)
    if key not in _BUFFER_POOL:
        _BUFFER_POOL[key] = FixedEPLayer(
            cfg, rank, world, None, None, capacity=capacity, group=group, device=device, fused_dispatch=fused_dispatch
        )
    return _BUFFER_POOL[key]


def _cfg_from_vllm() -> MoEConfig:
    hf = get_current_vllm_config().model_config.hf_text_config
    return MoEConfig(
        hidden_size=hf.hidden_size,
        moe_intermediate_size=hf.moe_intermediate_size,
        n_routed_experts=hf.n_routed_experts,
        n_shared_experts=hf.n_shared_experts,
        top_k=hf.num_experts_per_tok,
        n_group=hf.n_group,
        topk_group=hf.topk_group,
        norm_topk_prob=hf.norm_topk_prob,
        routed_scaling_factor=float(hf.routed_scaling_factor),
        scoring_func=hf.scoring_func,
        topk_method=hf.topk_method,
        swiglu_limit=getattr(hf, "swiglu_limit", None),
    )


class MoEFlowRunner(MoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moeflow_cfg = _cfg_from_vllm()
        self._moeflow_calls = 0
        self._moeflow_fixed_calls = 0
        self.moeflow_mode = os.environ.get("MOEFLOW_MODE", "route_check")
        self.moeflow_max_tokens = int(os.environ.get("MOEFLOW_MAX_TOKENS", "256"))
        self.moeflow_combine_mode = os.environ.get("MOEFLOW_COMBINE_MODE", "fp32")
        self.moeflow_fused_dispatch = os.environ.get("MOEFLOW_FUSED_DISPATCH") == "1"
        self.moeflow_fused_expert_prep = os.environ.get("MOEFLOW_FUSED_EXPERT_PREP") == "1"
        self.moeflow_direct_expert_output = os.environ.get("MOEFLOW_DIRECT_EXPERT_OUTPUT") == "1"
        if self.moeflow_direct_expert_output and not self.moeflow_fused_expert_prep:
            raise ValueError("Direct expert output requires MOEFLOW_FUSED_EXPERT_PREP=1")
        if self.moeflow_combine_mode not in ("fp32", "bf16_ring", "bf16_ring_fused"):
            raise ValueError("MOEFLOW_COMBINE_MODE must be fp32, bf16_ring or bf16_ring_fused")
        self._local_kernel = None
        trace_dir = os.environ.get("MOEFLOW_ROUTING_TRACE_DIR")
        self._routing_trace_dir = Path(trace_dir) if trace_dir else None
        self._routing_trace_calls = 0
        self._fixed_probe = None
        if os.environ.get("MOEFLOW_VALIDATE_FIXED") == "1":
            from .vllm_diagnostics import FixedLayerProbe

            if self._routing_trace_dir is None:
                raise ValueError("Fixed diagnostics require MOEFLOW_ROUTING_TRACE_DIR and eager execution")
            self._fixed_probe = FixedLayerProbe(self._routing_trace_dir)
        log_dir = os.environ.get("MOEFLOW_SMOKE_DIR")
        self._moeflow_log = Path(log_dir) / f"runner_pid{os.getpid()}.jsonl" if log_dir else None

    def _record(self, entry: dict) -> None:
        calls = self._moeflow_fixed_calls if entry.get("mode") == "fixed" else self._moeflow_calls
        if self._moeflow_log is not None and calls <= _MAX_LOGGED_CALLS_PER_LAYER:
            entry.update(layer=self.layer_name, call=self._moeflow_calls)
            with open(self._moeflow_log, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _record_routing_trace(self, ids, weights) -> None:
        """Sample actual eager requests only after the launcher enables collection."""
        directory = self._routing_trace_dir
        if directory is None or self._routing_trace_calls >= 32 or not (directory / "enabled").is_file():
            return
        if ids.shape[0] == 0 or torch.cuda.is_current_stream_capturing():
            return
        from vllm.distributed import get_ep_group
        from vllm.forward_context import get_forward_context

        group = get_ep_group()
        execution_rows = ids.shape[0]
        padding = get_forward_context().is_padding
        if padding is not None:
            if padding.shape != (execution_rows,):
                raise ValueError("Routing trace padding mask does not match local rows")
            ids, weights = ids[~padding], weights[~padding]
        count = min(ids.shape[0], 128)
        row = {
            "layer": self.layer_name,
            "call": self._moeflow_calls,
            "mode": self.moeflow_mode,
            "ep_rank": group.rank_in_group,
            "world": group.world_size,
            "execution_rows": execution_rows,
            "padding_known": padding is not None,
            "tokens": ids.shape[0],
            "tokens_recorded": count,
            "expert_ids": ids[:count].detach().cpu().tolist(),
            "weights": weights[:count].detach().float().cpu().tolist(),
        }
        with (directory / f"routes_pid{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        self._routing_trace_calls += 1

    def _apply_quant_method(
        self, hidden_states, router_logits, shared_experts_input, input_ids=None, shared_experts_overlapping=False
    ):
        self._moeflow_calls += 1
        if self.moeflow_mode == "fixed" and not self.routed_experts.quant_method.is_monolithic:
            cap = self._fixed_capacity(hidden_states.shape[0])
            if cap is not None:
                return self._apply_fixed(
                    hidden_states, router_logits, shared_experts_input, input_ids, shared_experts_overlapping, cap
                )
        if self.moeflow_mode == "fixed" or self.routed_experts.quant_method.is_monolithic:
            # Fixed mode declines large prefills as a whole, including routing.
            reason = (
                "monolithic expert kernel" if self.routed_experts.quant_method.is_monolithic else "fixed token limit"
            )
            self._record({"intercepted": False, "reason": reason})
            return super()._apply_quant_method(
                hidden_states, router_logits, shared_experts_input, input_ids, shared_experts_overlapping
            )

        self._maybe_apply_shared_experts(shared_experts_input, _no_overlap_order())
        bias = getattr(self.gate, "e_score_correction_bias", None) if self.gate is not None else None
        weights, ids = route(router_logits, bias, self.moeflow_cfg)
        self._record_routing_trace(ids, weights)

        if self._moeflow_log is not None and self._moeflow_calls <= _MAX_LOGGED_CALLS_PER_LAYER:
            v_w, v_ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
            self._record(_compare_routing(ids, weights, v_ids, v_w))

        ids = ids.to(self._quant_method.topk_indices_dtype or torch.int32)
        fused_out = self.routed_experts.forward_modular(
            x=hidden_states,
            topk_weights=weights,
            topk_ids=ids,
            shared_experts=self._shared_experts,
            shared_experts_input=shared_experts_input,
        )
        return self._shared_output(shared_experts_overlapping, fused_out)

    def _shared_output(self, overlapping, routed_out):
        if overlapping:
            self._shared_experts.wait()
        return (self._shared_experts.output if self._shared_experts is not None else None, routed_out)

    def _fixed_capacity(self, local_tokens: int) -> int | None:
        from vllm.forward_context import get_forward_context

        dp = get_forward_context().dp_metadata
        max_tokens = int(dp.num_tokens_across_dp_cpu.max()) if dp is not None else local_tokens
        if max_tokens > self.moeflow_max_tokens:
            return None
        capacity = 1
        while capacity < max(max_tokens, local_tokens, 1):
            capacity *= 2
        return capacity

    def _local_experts_kernel(self):
        if self._local_kernel is None:
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
                MoEPrepareAndFinalizeNoDPEPModular,
            )

            experts = self._quant_method.moe_kernel.fused_experts
            self._local_kernel = mk.FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)
        return self._local_kernel

    def _apply_fixed(self, hidden_states, router_logits, shared_experts_input, input_ids, overlapping, capacity):
        from vllm.distributed import get_ep_group

        from .vllm_runner_ops import local_to_global_ids, make_combine_fn, make_prepare_fn

        self._moeflow_fixed_calls += 1
        self._maybe_apply_shared_experts(shared_experts_input, _no_overlap_order())
        weights, ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=self._quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
        self._record_routing_trace(ids, weights)
        ep_group = get_ep_group()
        layer = _shared_fixed_layer(
            self.moeflow_cfg,
            ep_group.rank_in_group,
            ep_group.world_size,
            capacity,
            hidden_states.device,
            ep_group.device_group,
            fused_dispatch=self.moeflow_fused_dispatch,
        )
        routed_experts = self.routed_experts
        kernel = self._local_experts_kernel()
        offset = ep_group.rank_in_group * layer.L

        def compute_local_experts(hidden_rows, expert_ids, routing_weights):
            if self.moeflow_fused_expert_prep:
                global_ids, local_weights = expert_ids, routing_weights
            else:
                global_ids, local_weights = local_to_global_ids(
                    expert_ids, routing_weights, offset, routed_experts.global_num_experts
                )
            return kernel.apply(
                hidden_rows,
                routed_experts.w13_weight,
                routed_experts.w2_weight,
                local_weights,
                global_ids.to(ids.dtype),
                activation=routed_experts.activation,
                global_num_experts=routed_experts.global_num_experts,
                expert_map=routed_experts.expert_map,
                apply_router_weight_on_input=False,
            )

        if self.moeflow_combine_mode != "fp32" and hidden_states.dtype != torch.bfloat16:
            raise ValueError("BF16 Ring compatibility requires BF16 hidden states")
        combine_fn = make_combine_fn(self.moeflow_combine_mode, ep_group.rank_in_group)
        out = layer.forward_routed(
            hidden_states,
            weights,
            ids,
            expert_fn=compute_local_experts,
            combine_fn=combine_fn,
            prepare_fn=make_prepare_fn(self.moeflow_cfg, offset, routed_experts.global_num_experts, ids.dtype)
            if self.moeflow_fused_expert_prep
            else None,
            direct_expert_output=self.moeflow_direct_expert_output,
        )
        if self._fixed_probe is not None:
            self._fixed_probe.compare(
                routed_experts,
                hidden_states,
                weights,
                ids,
                out,
                self.layer_name,
                capacity,
                owner_partials=layer.ret_recv,
                experts_per_rank=layer.L,
            )
        if (
            self._moeflow_log is not None
            and self._moeflow_fixed_calls <= _MAX_LOGGED_CALLS_PER_LAYER
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._record(
                {
                    "intercepted": True,
                    "mode": "fixed",
                    "tokens": hidden_states.shape[0],
                    "capacity": capacity,
                    "combine_mode": self.moeflow_combine_mode,
                    "fused_dispatch": self.moeflow_fused_dispatch,
                    "fused_expert_prep": self.moeflow_fused_expert_prep,
                    "direct_expert_output": self.moeflow_direct_expert_output,
                }
            )
        return self._shared_output(overlapping, out)


def _no_overlap_order():
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

    return SharedExpertsOrder.NO_OVERLAP


def _compare_routing(ids, w, v_ids, v_w) -> dict:
    """Slot order may differ between implementations; compare as per-token {expert: weight} maps."""
    T = ids.shape[0]
    if T == 0:
        return {"intercepted": True, "tokens": 0}
    a_sorted, a_idx = ids.long().sort(-1)
    b_sorted, b_idx = v_ids.long().sort(-1)
    same = (a_sorted == b_sorted).all(-1)
    wa = w.float().gather(1, a_idx)
    wb = v_w.float().gather(1, b_idx)
    wdiff = (wa - wb).abs()[same]
    return {
        "intercepted": True,
        "tokens": T,
        "token_expert_set_match": same.float().mean().item(),
        "max_abs_weight_diff_on_match": wdiff.max().item() if wdiff.numel() else None,
    }
