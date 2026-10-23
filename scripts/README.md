# scripts

Entry points that run, verify and report the experiments; the library itself is `moeflow/`.
Every script reads its site settings from [`lib/config.sh`](lib/config.sh), so nothing is tied to one cluster.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MOEFLOW_PYTHON` / `MOEFLOW_TORCHRUN` / `MOEFLOW_VLLM` | `.venv/bin/{python,torchrun,vllm}` | interpreter and launchers (the `scripts/quickstart.sh` environment) |
| `MOEFLOW_MODEL_PATH` | `checkpoints/GLM-5.3-Flash` | full model snapshot directory |
| `MOEFLOW_TRIMMED_MODEL_PATH` | `checkpoints/GLM-5.3-Flash-4L` | layer-trimmed checkpoint, built with `make_trimmed_checkpoint.py` |
| `MOEFLOW_JOBID` | unset | existing Slurm allocation; GPU wrappers `srun --jobid=$MOEFLOW_JOBID --overlap` into it |

Slurm partition and account are not hard-coded: `sbatch -p <partition> -A <account> scripts/<job>.sbatch`.
Run `sbatch` from the repository root; each job writes `results/<stage>/<name>_<jobid>/commands.log` with the git SHA, job description and GPU occupancy.

## Contents

| Group | Scripts |
|---|---|
| Quick start and tests | `quickstart.sh`, `run_all_tests.sh`, `run_multigpu_check.sh` |
| Layer benchmarks | `bench_layer.py`, `bench_stages.py`, `bench_ring_combine.py`, `run_baselines.sh`, `profile_rank.sh` |
| Paired A/B measurements | `ab_fixed.py`, `ab_native_fixed.py`, `paired_*.sbatch` |
| Serving and end to end | `smoke_vllm.sh`, `e2e_bench.sh`, `smoke_client.py`, `logits_client.py`, `measure_w1.sbatch` |
| Verification gates | `verify_*.sbatch`, `validate_serving_result.py` |
| Profiling and traces | `profile_*.sbatch`, `trace_full_model.sbatch`, `diagnose_full_model.sbatch` |
| Report generation | `summarize_*.py` |
| Utilities | `make_trimmed_checkpoint.py`, `probes/` (Triton and NCCL sanity checks), `bench/` (shared timing, statistics and record helpers), `lib/` (shared shell) |
