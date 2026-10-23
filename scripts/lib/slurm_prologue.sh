# Shared Slurm job prologue: command log plus the provenance every result directory carries.
# Caller sets OUT (result directory) and GPUS (cards allocated to this job), then sources this file.
mkdir -p "$OUT"
exec > >(tee -a "$OUT/commands.log") 2>&1
set -x
git rev-parse HEAD
scontrol show job "$SLURM_JOB_ID"
srun --overlap -n1 --gres=gpu:"$GPUS" nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
