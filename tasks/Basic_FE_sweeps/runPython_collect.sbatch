#!/bin/bash
#SBATCH -J FE_collect
#SBATCH --account=gts-aerturk3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t10
#SBATCH -q embers
#SBATCH -o ./Report-collect-%j.out

set -euo pipefail

module load anaconda3

python -u collect_nD_sweep_results.py
