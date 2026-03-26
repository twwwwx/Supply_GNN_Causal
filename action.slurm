#!/bin/bash
#SBATCH --job-name=sim          # Job name
#SBATCH --time=0:10:00             # Time limit hrs:min:sec
#SBATCH --account=one_sc_default
#SBATCH --output=/dev/null      # Prevent SLURM from handling output directly
#SBATCH --mem=1G                # Memory demand (e.g., 4GB)
#SBATCH --cpus-per-task=1

log_name=$(date +"%m%d%H%M")
log_file="log/${log_name}.log"

module load python

# Redirect all output to the log file
exec > "$log_file" 2>&1

python test/test_load.py
# Output the job number to the log file
echo "Job ID: ${SLURM_JOB_ID}"

