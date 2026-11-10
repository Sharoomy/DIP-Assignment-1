#!/bin/bash
# ======================
# data_vis.sh
# ======================
#SBATCH --job-name=data_vis
#SBATCH --partition=teach_cpu
#SBATCH --account=dl23921
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem=100M

source ~/initMamba.sh
mamba activate projectenv

cd $SLURM_SUBMIT_DIR

python data_vis.py