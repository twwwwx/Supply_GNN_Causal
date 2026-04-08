#!/bin/bash
#SBATCH --job-name=sim_gnn          # Job name
#SBATCH --time=1:30:00          # Time limit hrs:min:sec
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

# Run both GNN variants for the spillover DGP (1 wide row per command in metrics.csv)
for DGP in spillover; do
  for model in gnn dirgnn; do
    python sim.py \
      --model "$model" \
      --DGP "$DGP" \
      --num_runs 100 \
      --n 2000 \
      --seed 123 \
      --gen_graph er \
      --tau_dir_true 2.0 \
      --tau_in_true 1.0 \
      --tau_out_true 1.0 \
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
