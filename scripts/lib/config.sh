# Site configuration shared by every script. Override any value from the environment.
#   MOEFLOW_PYTHON / _TORCHRUN / _VLLM  interpreter and launchers (default: the quickstart .venv)
#   MOEFLOW_MODEL_PATH                  full GLM-5.3-Flash snapshot directory
#   MOEFLOW_TRIMMED_MODEL_PATH          layer-trimmed checkpoint (see scripts/make_trimmed_checkpoint.py)
# Slurm partition and account are not set here: pass them to sbatch (e.g. `sbatch -p <partition>`).
PYTHON="${MOEFLOW_PYTHON:-.venv/bin/python}"
TORCHRUN="${MOEFLOW_TORCHRUN:-.venv/bin/torchrun}"
VLLM="${MOEFLOW_VLLM:-.venv/bin/vllm}"
export MOEFLOW_MODEL_PATH="${MOEFLOW_MODEL_PATH:-checkpoints/GLM-5.3-Flash}"
export MOEFLOW_TRIMMED_MODEL_PATH="${MOEFLOW_TRIMMED_MODEL_PATH:-checkpoints/GLM-5.3-Flash-4L}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"  # the shared benchmark helpers live in scripts/bench
