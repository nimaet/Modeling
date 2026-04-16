#!/bin/bash
#SBATCH -J FE_collect
#SBATCH --account=gts-aerturk3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t10
#SBATCH -q embers
#SBATCH -o ./Report-collect-%j.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch runPython_collect.sbatch <array_job_id>"
  exit 1
fi

ARRAY_JOB_ID="$1"

module load anaconda3

python -u collect_nD_sweep_results.py --array-job-id "${ARRAY_JOB_ID}"
