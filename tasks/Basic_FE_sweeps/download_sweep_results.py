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
	f"Modeling/tasks/FE_studies/sim_dat/{dirnaeme}"
)

FILENAME = "config.json"
INCLUDE_PATTERNS = ["config.json", "*.pkl"]

list_cmd = (
	f"ssh {REMOTE_USER}@{REMOTE_HOST} "
	f"'find {shlex.quote(REMOTE_DIR)} -maxdepth 2 -type f | sort'"
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
	f"--include='*/' "
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