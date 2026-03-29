#!/bin/bash
#SBATCH --job-name=sim_py          # Job name
#SBATCH --time=2:30:00          # Time limit hrs:min:sec
#SBATCH --account=one_sc_default
#SBATCH --output=/dev/null      # Prevent SLURM from handling output directly
#SBATCH --mem=1G                # Memory demand (e.g., 4GB)
#SBATCH --cpus-per-task=1

log_name=$(date +"%m%d%H%M")
log_file="log/${log_name}.log"

module load anaconda
conda activate deep_learning

# Redirect all output to the log file
exec > "$log_file" 2>&1

# Run all model/DGP combos in one job (one row per command appended to metrics.csv)
for model in gnn; do
  for DGP in dir undir; do
    python sim.py \
      --model "$model" \
      --DGP "$DGP" \
      --num_runs 100 \
      --n 200 \
      --seed 123 \
      --gen_graph rgg \
      --tau_true 2.0 \
      --p_treat 0.5 \
      --features nodes \
      --clip 0.001 \
      --L 2 \
      --output_dim 6 \
      --metrics_csv results/metrics.csv \
      --variance_type skeleton \
      --variance_method max 
  done
done

# Output the job number to the log file
echo "Job ID: ${SLURM_JOB_ID}"
