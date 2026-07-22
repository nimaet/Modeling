#!/bin/bash
#SBATCH -J FE_reduce
#SBATCH --account=gts-aerturk3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t10
#SBATCH -q embers
#SBATCH -o ./Report-reduce-%j.out

set -euo pipefail

module load anaconda3

# Must be submitted from Basic_FE_sweeps/ (e.g. `sbatch reduced_pipeline/runPython_reduce.sbatch`),
# same convention as runPython_collect.sbatch -- reduce_nD_sweep_results.py
# resolves sim_dat/ relative to the submission directory, not this script's location.
python -u reduced_pipeline/reduce_nD_sweep_results.py
