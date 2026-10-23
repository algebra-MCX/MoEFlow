"""vLLM integration: moeflow dispatch/combine as a modular-kernel Prepare/Finalize.

vLLM's own fused expert kernels (Triton / FP8 / NVFP4, chosen by vLLM's oracle) run
unchanged between our prepare() and finalize(), so end-to-end comparisons isolate the
communication path.

The returned top-k ids follow the same convention as vLLM's DeepEP high-throughput
Prepare/Finalize: one row per (token, destination rank), local hits mapped back to global
expert ids, and invalid slots pointed at an expert this rank does not own so that
`expert_map` turns them into -1. See PROVENANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

from .. import dispatch as D
from ..comm import NcclRowExchange, pack_meta, unpack_meta


@dataclass
class _InFlight:
    plan: D.DispatchPlan
    recv_counts: torch.Tensor
    num_tokens: int


class MoEFlowPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Naive-NCCL moeflow transport (naive baseline). Not re-entrant: no DBO micro-batching."""

    def __init__(self, ep_group: torch.distributed.ProcessGroup, num_local_experts: int):
        super().__init__()
        self.comm = NcclRowExchange(ep_group)
        self.num_local_experts = num_local_experts
        self.rank_expert_offset = self.comm.rank * num_local_experts
        self._inflight: _InFlight | None = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return self.comm.world

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            raise NotImplementedError("moeflow applies router weights after the experts")
        plan = D.plan_dispatch(topk_ids, topk_weights, self.comm.world, self.num_local_experts)
        recv_counts = self.comm.exchange_counts(plan.send_counts)
        recv_x = self.comm.exchange(D.pack_send(a1, plan), plan.send_counts, recv_counts)
        recv_meta = self.comm.exchange(pack_meta(plan.slot_local_id, plan.slot_weight), plan.send_counts, recv_counts)
        self._inflight = _InFlight(plan, recv_counts, a1.shape[0])

        local_id, weight = unpack_meta(recv_meta)
        invalid = local_id == D.INVALID
        filler = num_experts - 1 if self.rank_expert_offset == 0 else 0
        global_ids = torch.where(invalid, torch.full_like(local_id, filler), local_id + self.rank_expert_offset)
        global_ids = global_ids.to(topk_ids.dtype)

        valid_local = local_id[~invalid].long()
        counts = torch.bincount(valid_local, minlength=self.num_local_experts).to(torch.int32)
        meta = mk.ExpertTokensMetadata(expert_num_tokens=counts, expert_num_tokens_cpu=None)

        scale = None
        if quant_config.quant_dtype is not None and not defer_input_quant and recv_x.numel():
            recv_x, scale = moe_kernel_quantize_input(
                recv_x,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
            )
        return recv_x, scale, meta, global_ids, weight

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        st, self._inflight = self._inflight, None
        assert st is not None, "finalize() without a matching prepare()"
        if fused_expert_output.numel():
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            partial = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        else:
            partial = fused_expert_output.new_empty((0, output.shape[-1]))
        back = self.comm.exchange(partial.to(torch.bfloat16), st.recv_counts, st.plan.send_counts)
        output.copy_(D.combine(back, st.plan, st.num_tokens, self.comm.world, output.dtype))
