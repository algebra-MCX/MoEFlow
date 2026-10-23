#!/usr/bin/env bash
# One command from a fresh clone to a verified install: creates .venv, installs the pinned vLLM and
# MoEFlow, then runs the CPU regression suite (reference, routing, dispatch/restore, simulators).
# Inside a Slurm allocation, MOEFLOW_JOBID=<alloc> additionally runs the GPU tests (see run_all_tests.sh).
# Usage: scripts/quickstart.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.32.0" -e ".[dev]"
if [ -n "${MOEFLOW_JOBID:-}" ]; then
  exec scripts/run_all_tests.sh
fi
exec .venv/bin/python -m pytest -q -p no:cacheprovider -m "not gpu and not multigpu and not vllm"
