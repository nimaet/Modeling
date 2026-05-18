#!/bin/bash
#SBATCH -J FE_piezoBeam
#SBATCH --account=gts-aerturk3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t120
#SBATCH --array=0-49
#SBATCH -q embers
#SBATCH -o ./reports/Report-%A_%a.out

module load anaconda3

# Prevent nested thread oversubscription inside each joblib worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python generic_nD_sweep_SLURMarray.py