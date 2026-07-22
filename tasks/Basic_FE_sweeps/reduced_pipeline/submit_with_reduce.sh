#!/bin/bash
set -euo pipefail

# Run from Basic_FE_sweeps/ (same convention as submit_with_collector.sh).
# NOTE: before running, set ARRAY_JOB_ID in reduce_nD_sweep_results.py to
# match the array job id this script submits (printed below) -- same manual
# step submit_with_collector.sh already requires for collect_nD_sweep_results.py.

mkdir -p reports

# Submit the array workers first.
array_submit_output=$(sbatch runPython_SLURMarray.sbatch)
array_job_id=$(echo "${array_submit_output}" | awk '{print $4}')

echo "Submitted array job: ${array_job_id}"
echo "Set ARRAY_JOB_ID = \"${array_job_id}\" in reduced_pipeline/reduce_nD_sweep_results.py if not already set."

# Submit one reduce job that starts only after array completion.
reduce_submit_output=$(sbatch --dependency=afterany:${array_job_id} reduced_pipeline/runPython_reduce.sbatch)
reduce_job_id=$(echo "${reduce_submit_output}" | awk '{print $4}')

echo "Submitted reduce job: ${reduce_job_id}"
echo "Reduce dependency: afterany:${array_job_id}"
