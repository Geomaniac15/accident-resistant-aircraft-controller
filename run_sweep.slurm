#!/bin/bash
# Massive failure sweep for the 747-8 controller.
# sweep.py runs the cascade autopilot against the full cartesian product of
#   phase x failure x affected unit/side x failure timing x weight x wind x gains
# and records, per run, whether the aircraft survives plus outcome metrics
# (impact time, peak/min altitude, min stall margin, max bank, max AoA).
# Submit with: sbatch run_sweep.slurm
#
# Array: NSHARDS tasks, one CSV each -> results/shard_<id>.csv.
#   Aggregate afterwards with: python sweep.py --summarize 'results/shard_*.csv'
#
# OUTDIR defaults to results/ here; override at submit time:
#   OUTDIR=/abs/path sbatch run_sweep.slurm

#SBATCH --partition=cpu-6h
#SBATCH -J flight_sweep
#SBATCH -t 5:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-63
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

source /etc/profile
module add miniforge
source activate "${FLIGHT_ENV:-$global_storage/flight_env}"   # needs python + numpy

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# NSHARDS must match the --array size above (0..NSHARDS-1).
NSHARDS=64
OUTDIR="${OUTDIR:-results}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
SHARD=$SLURM_ARRAY_TASK_ID
CSV="$OUTDIR/shard_${SHARD}.csv"

echo "[shard $SHARD/$NSHARDS] host=$(hostname) workers=$WORKERS started=$(date)"
echo "[shard $SHARD/$NSHARDS] writing -> $CSV"

python sweep.py \
    --shard ${SHARD}/${NSHARDS} \
    --workers ${WORKERS} \
    --out "$CSV"

echo "[shard $SHARD/$NSHARDS] finished=$(date)"
