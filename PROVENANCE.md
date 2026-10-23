# Provenance

Code adapted from, or modelled on, third-party projects. Nothing listed here is claimed as
original work.

| File | Source | License | What was taken |
|---|---|---|---|
| `moeflow/integrations/vllm_pf.py` | vLLM v0.32.0 `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` (`DeepEPHTPrepareAndFinalize._receiver`, `_finalize`) | Apache-2.0 | Convention of mapping invalid top-k slots to a non-local expert id so `expert_map` yields −1; delegate→contiguous weight-and-reduce before the return exchange; quantize-after-dispatch call shape |
| `moeflow/integrations/vllm_runner.py` | vLLM v0.32.0 `fused_moe/runner/moe_runner.py` (`MoERunner._apply_quant_method`) | Apache-2.0 | Subclass override; body mirrors the parent's modular branch (shared-expert ordering, `forward_modular` call, return tuple) |
| `moeflow/router.py`, `moeflow/reference.py` (routing semantics only) | vLLM v0.32.0 `fused_moe/router/grouped_topk_router.py` (`grouped_topk`) | Apache-2.0 | Semantics of noaux_tc selection (biased selection, unbiased weights, renormalize, scale). Code written independently |
| `moeflow/dispatch.py` (`reduce_local` rounding mode, `combine` order) | vLLM `csrc/moe/moe_align_sum_kernels.cu` (`moe_sum_vec_kernel`), vLLM `topk_weight_and_reduce.py`; DeepEP `d4f41e4` `csrc/kernels/legacy/intranode.cu` combine | Apache-2.0; MIT | Rounding/accumulation order replicated for bitwise comparability. Code written independently |
| `moeflow/integrations/vllm_runner_ops.py`, `moeflow/kernels/reduce.py` (`combine_bf16_ring`) | Measured native NCCL 2.29.7 EP4 Ring reduction behavior, recorded in `results/stage1/full_diagnostic_9229/` | Independent implementation; no third-party code copied | Source-relative owner order and intermediate BF16 rounding reproduced for the explicit compatibility experiment; not assumed portable across NCCL algorithms |
| `scripts/make_trimmed_checkpoint.py` | — | — | Original |

Third-party sources kept for reading only (not shipped, git-ignored): `third_party/DeepEP`
(github.com/deepseek-ai/DeepEP @ d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb, MIT),
`third_party/vllm_v0.32.0_install_python_libraries.sh` (vLLM, Apache-2.0).
