#!/bin/bash
#SBATCH --job-name=sim-study
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
cd "$(dirname "$0")"

python3 sim.py --num-runs 5 --sample-sizes 100,500,1000
