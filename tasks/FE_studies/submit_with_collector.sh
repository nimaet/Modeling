#!/bin/bash
set -euo pipefail

mkdir -p reports

# Submit the array workers first.
array_submit_output=$(sbatch runPython_SLURMarray.sbatch)
array_job_id=$(echo "${array_submit_output}" | awk '{print $4}')

echo "Submitted array job: ${array_job_id}"

# Submit one collector job that starts only after array completion.
collect_submit_output=$(sbatch --dependency=afterany:${array_job_id} runPython_collect.sbatch "${array_job_id}")
collect_job_id=$(echo "${collect_submit_output}" | awk '{print $4}')

echo "Submitted collector job: ${collect_job_id}"
echo "Collector dependency: afterany:${array_job_id}"
