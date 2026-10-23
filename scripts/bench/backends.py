"""Layer backends under one boundary: router logits in -> weighted routed-expert sum out.

Each backend is built per rank after `init_dist` and exposes `run(x, logits) -> out`.

* `MoEFlowBackend`: the project pipeline (`EPMoELayer`) with a selectable expert backend.
* `VllmModularBackend`: vLLM's modular kernel, i.e. vLLM's grouped-top-k router followed by
  `FusedMoEKernel(prepare_finalize, experts)`. The Prepare/Finalize comes from vLLM's own
  factory for the configured `all2all_backend` (allgather_reducescatter, deepep_high_throughput,
  deepep_low_latency, ...) or is MoEFlow's (`moeflow`), and the experts are vLLM's
  Triton / batched-Triton experts — the communication x GEMM cross matrix.

Distributed set-up mirrors vLLM v0.32.0 tests/kernels/moe/modular_kernel_tools/parallel_utils.py
(see PROVENANCE.md).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist

from moeflow.comm import NcclRowExchange
from moeflow.config import MoEConfig
from moeflow.ep_moe import EPMoELayer
from moeflow.experts import EXPERT_BACKENDS
from moeflow.fixed import FixedEPLayer
from moeflow.integrations.vllm_runner_ops import make_combine_fn, make_prepare_fn

VLLM_P2F = ("allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency", "moeflow", "moeflow_fixed")


@dataclass
class DistInfo:
    rank: int
    world: int
    local_rank: int
    device: torch.device


def init_dist(all2all_backend: str | None = None) -> tuple[DistInfo, object | None]:
    """Initialise torch.distributed from torchrun env; optionally vLLM's DP=EP groups."""
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=device)
    info = DistInfo(rank, world, local, device)
    if all2all_backend is None:
        return info, None
    return info, _init_vllm(info, all2all_backend)


def _init_vllm(info: DistInfo, all2all_backend: str):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(
        enforce_eager=True, is_moe=True, sleep_mode_offload_cudagraph=False, enable_sleep_mode=False
    )
    cfg.parallel_config.data_parallel_size = info.world
    cfg.parallel_config.enable_expert_parallel = True
    # MoEFlow's Prepare/Finalize is slotted in later; vLLM still needs an all-to-all mode that
    # creates the EP groups, the reference AG/RS one is used for that.
    cfg.parallel_config.all2all_backend = (
        "allgather_reducescatter" if all2all_backend in ("moeflow", "moeflow_fixed") else all2all_backend
    )
    cfg.parallel_config.data_parallel_rank = info.rank
    cfg.parallel_config.rank = 0
    store = tempfile.mkstemp(prefix=f"moeflow_vllm_{os.environ.get('MASTER_PORT', '0')}_")[1]
    with set_current_vllm_config(cfg):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{store}", local_rank=info.local_rank, backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(info.device)  # as in vLLM's profile_modular_kernel.py
    return cfg


class MoEFlowBackend:
    name = "moeflow"

    def __init__(self, cfg: MoEConfig, info: DistInfo, w13_local, w2_local, bias, experts: str = "triton_grouped"):
        self.layer = EPMoELayer(cfg, info.rank, info.world, w13_local, w2_local, experts=EXPERT_BACKENDS[experts]())
        self.comm = NcclRowExchange()
        self.bias = bias
        self.label = f"moeflow_naive+{experts}"

    def run(self, x, logits):
        return self.layer.forward(x, logits, self.bias, self.comm)


class MoEFlowFixedBackend:
    """Fixed-capacity, graph-capturable path; capacity = tokens per rank of the working point."""

    name = "moeflow_fixed"

    def __init__(self, cfg: MoEConfig, info: DistInfo, w13_local, w2_local, bias, capacity: int):
        self.layer = FixedEPLayer(cfg, info.rank, info.world, w13_local, w2_local, capacity=capacity)
        self.bias = bias
        self.label = "moeflow_fixed+triton_grouped"

    def run(self, x, logits):
        return self.layer.forward(x, logits, self.bias)


class VllmModularBackend:
    def __init__(
        self,
        cfg: MoEConfig,
        info: DistInfo,
        vllm_config,
        all2all_backend: str,
        w13_local,
        w2_local,
        bias,
        max_tokens_per_rank: int,
        tokens_per_rank: list[int],
    ):
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.config import set_current_vllm_config
        from vllm.distributed import get_dp_group, get_pcp_group, get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            FusedMoEQuantConfig,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import BatchedTritonExperts
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        from vllm.utils.math_utils import next_power_of_2

        self.cfg, self.info, self.vllm_config, self.bias = cfg, info, vllm_config, bias
        self.w13, self.w2 = w13_local, w2_local
        L = cfg.experts_per_rank(info.world)
        with set_current_vllm_config(vllm_config):
            pcfg = FusedMoEParallelConfig.make(
                tp_size_=get_tensor_model_parallel_world_size(),
                pcp_size_=get_pcp_group().world_size,
                dp_size_=get_dp_group().world_size,
                sp_size_=1,
                vllm_parallel_config=vllm_config.parallel_config,
            )
            moe = FusedMoEConfig(
                num_experts=cfg.n_routed_experts,
                experts_per_token=cfg.top_k,
                hidden_dim=cfg.hidden_size,
                intermediate_size=cfg.moe_intermediate_size,
                num_local_experts=L,
                num_logical_experts=cfg.n_routed_experts,
                moe_parallel_config=pcfg,
                in_dtype=torch.bfloat16,
                max_num_tokens=next_power_of_2(max(max_tokens_per_rank, 1)),
                activation=MoEActivation.SILU,
                device=vllm_config.device_config.device,
                routing_method=RoutingMethodType.DeepSeekV3,
                swiglu_limit=cfg.swiglu_limit,
            )
            self.quant_config = FusedMoEQuantConfig.make(None)
            if all2all_backend == "moeflow":
                from vllm.distributed import get_ep_group

                from moeflow.integrations.vllm_pf import MoEFlowPrepareAndFinalize

                pf = MoEFlowPrepareAndFinalize(get_ep_group().device_group, L)
            else:
                pf = maybe_make_prepare_finalize(moe=moe, quant_config=self.quant_config, allow_new_interface=True)
            assert pf is not None, f"vLLM built no Prepare/Finalize for {all2all_backend}"
            if pf.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
                experts = BatchedTritonExperts(
                    moe_config=moe,
                    quant_config=self.quant_config,
                    max_num_tokens=moe.max_num_tokens,
                    num_dispatchers=pf.num_dispatchers(),
                )
            else:
                experts = TritonExperts(moe_config=moe, quant_config=self.quant_config)
            self.kernel = mk.FusedMoEKernel(prepare_finalize=pf, fused_experts=experts)
        self.label = f"{type(pf).__name__}+{type(experts).__name__}"
        self.activation = MoEActivation.SILU
        emap = torch.full((cfg.n_routed_experts,), -1, dtype=torch.int32)
        emap[info.rank * L : (info.rank + 1) * L] = torch.arange(L, dtype=torch.int32)
        self.expert_map = emap.to(info.device)
        # vLLM keeps per-DP-rank token counts on the host (DPMetadata.num_tokens_across_dp_cpu).
        self.tokens_across_dp = torch.tensor(tokens_per_rank, dtype=torch.int32)
        self.topk_dtype = pf.topk_indices_dtype()

    def route(self, logits):
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import grouped_topk

        c = self.cfg
        return grouped_topk(
            hidden_states=logits,
            gating_output=logits,
            topk=c.top_k,
            renormalize=c.norm_topk_prob,
            num_expert_group=c.n_group,
            topk_group=c.topk_group,
            scoring_func=c.scoring_func,
            routed_scaling_factor=c.routed_scaling_factor,
            e_score_correction_bias=self.bias,
        )

    def run(self, x, logits):
        from vllm.config import set_current_vllm_config
        from vllm.forward_context import get_forward_context, set_forward_context

        w, ids = self.route(logits)
        if self.topk_dtype is not None:
            ids = ids.to(self.topk_dtype)
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                None, self.vllm_config, num_tokens=x.shape[0], num_tokens_across_dp=self.tokens_across_dp
            ),
        ):
            # Same context vLLM's MoERunner._sequence_parallel_context() opens around the kernel.
            with get_forward_context().dp_metadata.sp_local_sizes(1, pcp_size=1, use_ep=True):
                return self._apply(x, w, ids)

    def _apply(self, x, w, ids):
        return self.kernel.apply(
            hidden_states=x,
            w1=self.w13,
            w2=self.w2,
            topk_weights=w,
            topk_ids=ids,
            activation=self.activation,
            global_num_experts=self.cfg.n_routed_experts,
            expert_map=self.expert_map,
            apply_router_weight_on_input=False,
        )


class MoEFlowFixedVllmBackend(VllmModularBackend):
    """Fixed transport with the same vLLM router and BF16 experts as the native baseline."""

    def __init__(
        self,
        cfg,
        info,
        vllm_config,
        w13_local,
        w2_local,
        bias,
        capacity,
        max_tokens_per_rank,
        fused_combine=False,
        combine_mode="fp32",
        fused_dispatch=False,
        fused_expert_prep=False,
        direct_expert_output=False,
    ):
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular

        if fused_combine and combine_mode != "fp32":
            raise ValueError("Choose either the FP32 fused switch or a BF16 Ring combine mode")
        if direct_expert_output and not fused_expert_prep:
            raise ValueError("Direct expert output requires fused expert preparation")
        self.direct_expert_output = direct_expert_output
        self.combine_fn = make_combine_fn(combine_mode, info.rank)
        super().__init__(
            cfg, info, vllm_config, "moeflow", w13_local, w2_local, bias, max_tokens_per_rank, [capacity] * info.world
        )
        experts = self.kernel.fused_experts
        self.kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)
        self.fixed = FixedEPLayer(
            cfg,
            info.rank,
            info.world,
            None,
            None,
            capacity=capacity,
            device=info.device,
            fused_combine=fused_combine,
            fused_dispatch=fused_dispatch,
        )
        self.label = f"moeflow_fixed+{type(experts).__name__}" + ("+fused_combine" if fused_combine else "")
        if combine_mode != "fp32":
            self.label += f"+{combine_mode}"
        if fused_dispatch:
            self.label += "+fused_dispatch"
        self.fused_expert_prep = fused_expert_prep
        self.prepare_fn = None
        if fused_expert_prep:
            self.prepare_fn = make_prepare_fn(
                cfg, info.rank * self.fixed.L, cfg.n_routed_experts, self.topk_dtype or torch.int32
            )
            self.label += "+fused_expert_prep"
        if direct_expert_output:
            self.label += "+direct_expert_output"

    def _apply(self, x, weights, ids):
        return self.fixed.forward_routed(
            x,
            weights,
            ids,
            expert_fn=self._local_experts,
            combine_fn=self.combine_fn,
            prepare_fn=self.prepare_fn,
            direct_expert_output=self.direct_expert_output,
        )

    def _local_experts(self, hidden_rows, expert_ids, weights):
        from moeflow.integrations.vllm_runner_ops import local_to_global_ids

        if self.fused_expert_prep:
            global_ids, local_weights = expert_ids, weights  # prepared IDs are already global
        else:
            global_ids, local_weights = local_to_global_ids(
                expert_ids, weights, self.info.rank * self.fixed.L, self.cfg.n_routed_experts
            )
        # Packed communication rows include metadata, so the hidden view has a wider row stride.
        # BF16 Triton experts require contiguous input; include this compaction in layer timing.
        return super()._apply(hidden_rows.contiguous(), local_weights, global_ids.to(self.topk_dtype or torch.int32))
