#!/bin/bash
#SBATCH --job-name=FE_sweep_amp_kc
#SBATCH --array=0-13               # 14 cases (7 amps x 2 Kc cases)
#SBATCH --ntasks=1                 # 1 task per array task
#SBATCH --cpus-per-task=1          # Adjust based on available cores
#SBATCH --mem=8G                  # Memory per task
#SBATCH -t120           # Adjust as needed
#SBATCH --output=logs/slurm_%A_%a.log
#SBATCH --error=logs/slurm_%A_%a.err

# =========================================================
# SLURM Job Array Script for FE Sweep
# =========================================================
# This script submits N parallel jobs (one per sweep case).
# Each job runs run_single_sweep_case.py with a different case index.
# All results are saved to the same intermediate folder for post-processing.

# =========================================================
# Configuration
# =========================================================
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd ../.. && pwd )"
FE_STUDIES_DIR="${PROJECT_DIR}/Modeling/tasks/FE_studies"
RUN_NAME="Alternating_Kc_sweep_$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${FE_STUDIES_DIR}/sim_dat/${RUN_NAME}"
LOGS_DIR="${FE_STUDIES_DIR}/logs"

# =========================================================
# Setup
# =========================================================
mkdir -p "${OUTPUT_BASE}/intermediate_npz"
mkdir -p "${LOGS_DIR}"

# Activate conda/module environment if needed
# Example: module load python/3.10
# Example: conda activate myenv

# =========================================================
# Run single case
# =========================================================
echo "Starting job array task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX}"
echo "Output directory: ${OUTPUT_BASE}"
echo "Job ID: ${SLURM_JOB_ID}, Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Start time: $(date)"

cd "${FE_STUDIES_DIR}"

python run_single_sweep_case.py ${SLURM_ARRAY_TASK_ID} "${OUTPUT_BASE}"
EXIT_CODE=$?

echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
