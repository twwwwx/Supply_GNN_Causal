#!/bin/bash
#SBATCH --job-name=sim_gnn          # Job name
#SBATCH --time=3:30:00          # Time limit hrs:min:sec
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --output=/dev/null      # Prevent SLURM from handling output directly
#SBATCH --mem=16G                # Memory demand (e.g., 4GB)
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1

log_name=$(date +"%m%d%H%M")
log_file="log/${log_name}.log"

module load anaconda
conda activate tensorflow

# Redirect all output to the log file
exec > "$log_file" 2>&1

# Run all model/DGP combos in one job (one row per command appended to metrics.csv)
for DGP in dir; do
  for model in gnn dirgnn; do
    python sim.py \
      --model "$model" \
      --DGP "$DGP" \
      --num_runs 500 \
      --n 2000 \
      --seed 123 \
      --gen_graph er \
      --tau_true 2.0 \
      --p_treat 0.5 \
      --features nodes \
      --clip 0.001 \
      --L 2 \
      --output_dim 6 \
      --metrics_csv results/metrics.csv \
      --variance_type directed \
      --use_gpu 1
  done
done

# Output the job number to the log file
echo "Job ID: ${SLURM_JOB_ID}"
