#!/bin/bash
#SBATCH --job-name=sim_stats
#SBATCH --time=5:00:00
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --output=/dev/null
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mail-user=wkt5100@psu.edu

set -euo pipefail

log_name=$(date +"%m%d%H%M")
log_file="log/${log_name}.log"

mkdir -p log results

module load anaconda
conda activate tensorflow

# Redirect all output to the log file
exec > "$log_file" 2>&1

# -----------------------------
# Sweep knobs (edit as needed)
# -----------------------------
METRICS_CSV="${METRICS_CSV:-results/metrics.csv}"
NUM_RUNS="${NUM_RUNS:-500}"
USE_GPU="${USE_GPU:-0}"
MAX_CASES="${MAX_CASES:-0}"  # 0 => no cap

MODELS=(gnn dirgnn)
GRAPH_MODELS=(rgg)
SEEDS=(123)
SAMPLE_SIZES=(1500)
CLIPS=(0.0005)
LAYERS=(2 3)
OUTPUT_DIMS=(6)
DIR_DROPOUT_RATES=(0.1)

VARIANCE_TYPES=(skeleton directed)
SKELETON_METHODS=(max)
DIRECTED_METHODS=(dir_max)

run_case() {
  local model="$1"
  local graph="$2"
  local seed="$3"
  local n="$4"
  local clip="$5"
  local L="$6"
  local out_dim="$7"
  local dropout_rate="${8}"
  local vtype="${9}"
  local vmethod="${10}"

  local cmd=(
    python sim.py
    --model "$model"
    --DGP spillover
    --num_runs "$NUM_RUNS"
    --n "$n"
    --seed "$seed"
    --gen_graph "$graph"
    --tau_dir_true -2.0
    --tau_in_true 1.0
    --tau_out_true 1.0
    --features nodes
    --clip "$clip"
    --L "$L"
    --output_dim "$out_dim"
    --dir_dropout_rate "$dropout_rate"
    --metrics_csv "$METRICS_CSV"
    --variance_type "$vtype"
    --use_gpu "$USE_GPU"
  )

  if [[ -n "$vmethod" ]]; then
    cmd+=(--variance_method "$vmethod")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

case_count=0

for model in "${MODELS[@]}"; do
  for graph in "${GRAPH_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for n in "${SAMPLE_SIZES[@]}"; do
        for clip in "${CLIPS[@]}"; do
          for L in "${LAYERS[@]}"; do
            for out_dim in "${OUTPUT_DIMS[@]}"; do
              for dropout_rate in "${DIR_DROPOUT_RATES[@]}"; do
                for vtype in "${VARIANCE_TYPES[@]}"; do
                  # GNN should not run directed variance.
                  if [[ "$model" == "gnn" && "$vtype" == "directed" ]]; then
                    continue
                  fi
                  # DirGNN should run only directed variance.
                  if [[ "$model" == "dirgnn" && "$vtype" != "directed" ]]; then
                    continue
                  fi

                  methods=("")
                  if [[ "$vtype" == "skeleton" ]]; then
                    methods=("${SKELETON_METHODS[@]}")
                  elif [[ "$vtype" == "directed" ]]; then
                    methods=("${DIRECTED_METHODS[@]}")
                  fi

                  for vmethod in "${methods[@]}"; do
                    run_case "$model" "$graph" "$seed" "$n" "$clip" "$L" "$out_dim" "$dropout_rate" "$vtype" "$vmethod"
                    case_count=$((case_count + 1))
                    if [[ "$MAX_CASES" -gt 0 && "$case_count" -ge "$MAX_CASES" ]]; then
                      echo "Reached MAX_CASES=${MAX_CASES}; stopping early."
                      echo "Completed cases: ${case_count}"
                      # Add a blank separator line after this SLURM run.
                      printf '\n' >> "$METRICS_CSV"
                      echo "Job ID: ${SLURM_JOB_ID:-N/A}"
                      exit 0
                    fi
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Completed cases: ${case_count}"

# Add a blank separator line after this SLURM run.
printf '\n' >> "$METRICS_CSV"

echo "Job ID: ${SLURM_JOB_ID}"
