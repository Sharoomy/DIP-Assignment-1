#!/bin/bash
# ======================
# HDBSCAN.sh
# ======================
#SBATCH --job-name=HDBSCAN
#SBATCH --partition=teach_cpu
#SBATCH --account=phys040684
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --mem=2G

source ~/initMamba.sh
mamba activate projectenv1

cd $SLURM_SUBMIT_DIR

python HDBSCAN.py