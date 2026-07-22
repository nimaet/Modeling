"""Download only the small reduced-pipeline outputs for a run.

Companion to reduce_nD_sweep_results.py: run the reduction on the cluster
first (see runPython_reduce.sbatch), where it reads the raw npz's in place,
then use this script to pull just its output locally -- config.json,
reduction_manifest.json, and reduced_results__*.pkl. The raw npz/ folder is
never transferred.

No sharding / parallel rsync streams needed here (unlike
download_npz_results.py) -- this is a handful of small files, not thousands
of per-sim npz's.
"""
import subprocess
import shlex

REMOTE_USER = "setemadi3"
REMOTE_HOST = "login-phoenix.pace.gatech.edu"
dirnaeme = "Duffing_gain_sweep_11043124"
REMOTE_DIR = (
    f"/storage/home/hcoda1/4/setemadi3/scratch/Modeling/tasks/Basic_FE_sweeps/sim_dat/{dirnaeme}/"
)

DEST_WSL = (
    "/mnt/e/GaTech Dropbox/Seyednima Etemadi/"
    "Projects/Metamaterial beam/"
    f"Modeling/tasks/Basic_FE_sweeps/sim_dat/{dirnaeme}"
)

INCLUDE_PATTERNS = ["config.json", "reduction_manifest.json", "reduced_results__*.pkl"]

list_cmd = (
    f"ssh {REMOTE_USER}@{REMOTE_HOST} "
    f"'find {shlex.quote(REMOTE_DIR)} -maxdepth 1 -type f "
    f"\\( -name config.json -o -name reduction_manifest.json -o -name \"reduced_results__*.pkl\" \\) "
    f"| sort'"
)

print("Remote directory contents:")
list_result = subprocess.run(["wsl", "bash", "-lc", list_cmd], capture_output=True, text=True)
print(f"returncode: {list_result.returncode}")
print("\n--- stdout ---")
print(list_result.stdout if list_result.stdout else "<none>")
print("\n--- stderr ---")
print(list_result.stderr if list_result.stderr else "<none>")

include_args = " ".join(f"--include='{pattern}'" for pattern in INCLUDE_PATTERNS)

cmd = (
    f"mkdir -p {shlex.quote(DEST_WSL)} && "
    f"rsync -r --progress "
    f"--no-times --no-perms "
    f"{include_args} "
    f"--exclude='*' "
    f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR} "
    f"{shlex.quote(DEST_WSL)}"
)

print(cmd)
result = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True)
print(f"returncode: {result.returncode}")
print("\n--- stdout ---")
print(result.stdout if result.stdout else "<none>")
print("\n--- stderr ---")
print(result.stderr if result.stderr else "<none>")
