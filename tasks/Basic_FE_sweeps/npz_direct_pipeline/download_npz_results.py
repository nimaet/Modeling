"""Direct-npz variant of download_sweep_results.py.

Skips collect_nD_sweep_results.py entirely: pulls `config.json` plus every
per-simulation `npz/sim_*.npz` file straight from the cluster, instead of a
single combined `results.pkl`. See README.md in this folder for the full
workflow.

Downloading is network/round-trip-latency bound, not CPU bound, so it's not
"multicore" in the usual sense -- but with many small npz files, a single
rsync stream leaves most of that latency unhidden. This version launches
N_WORKERS concurrent rsync streams (via WSL), each restricted to sim indices
whose last digit falls in its own bucket, so transfers overlap instead of
queuing behind one another. config.json is fetched once, separately, since
it's tiny and unrelated to the per-sim sharding.
"""
import subprocess
import shlex

REMOTE_USER = "setemadi3"
REMOTE_HOST = "login-phoenix.pace.gatech.edu"
dirnaeme = "Duffing_softening_10928098"
REMOTE_DIR = (
f"/storage/home/hcoda1/4/setemadi3/scratch/Modeling/tasks/Basic_FE_sweeps/sim_dat/{dirnaeme}/"
)

DEST_WSL = (
    "/mnt/e/GaTech Dropbox/Seyednima Etemadi/"
	"Projects/Metamaterial beam/"
	f"Modeling/tasks/FE_studies/sim_dat/{dirnaeme}"
)

# Number of concurrent rsync streams for the npz transfer. Try 4-8; higher
# isn't necessarily better -- it's bounded by your network/VPN bandwidth and
# by how many concurrent sessions the login node/SSH server will tolerate.
# Must be between 1 and 10 (sims are sharded by the last digit of their
# zero-padded index, sim_00000.npz .. sim_99999.npz).
N_WORKERS = 6


def digit_groups(n_workers):
    if not (1 <= n_workers <= 10):
        raise ValueError("N_WORKERS must be between 1 and 10.")
    groups = [[] for _ in range(n_workers)]
    for digit in range(10):
        groups[digit % n_workers].append(digit)
    return groups


def npz_shard_pattern(digits):
    # rsync/wildmatch bracket expression: matches sim_*<d>.npz for d in digits.
    return "*[" + "".join(str(d) for d in digits) + "].npz"


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

# --- 1. config.json: one small file, fetched once, no sharding needed. ---
config_cmd = (
	f"mkdir -p {shlex.quote(DEST_WSL)} && "
	f"rsync --no-times --no-perms "
	f"{REMOTE_USER}@{REMOTE_HOST}:{shlex.quote(REMOTE_DIR + 'config.json')} "
	f"{shlex.quote(DEST_WSL)}/"
)

print("\nFetching config.json...")
print(config_cmd)
config_result = subprocess.run(["wsl", "bash", "-lc", config_cmd], capture_output=True, text=True)
print(f"returncode: {config_result.returncode}")
print("\n--- stdout ---")
print(config_result.stdout if config_result.stdout else "<none>")
print("\n--- stderr ---")
print(config_result.stderr if config_result.stderr else "<none>")

# --- 2. npz files: sharded across N_WORKERS concurrent rsync streams. ---
shards = digit_groups(N_WORKERS)
shard_cmds = []
for digits in shards:
	pattern = npz_shard_pattern(digits)
	cmd = (
		f"rsync -r --progress "
		f"--no-times --no-perms "
		f"--include='*/' "
		f"--include='{pattern}' "
		f"--exclude='*' "
		f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR} "
		f"{shlex.quote(DEST_WSL)}"
	)
	shard_cmds.append((digits, cmd))

print(f"\nLaunching {len(shard_cmds)} parallel rsync workers for npz files...")
processes = []
for digits, cmd in shard_cmds:
	print(f"[worker digits={digits}] {cmd}")
	proc = subprocess.Popen(
		["wsl", "bash", "-lc", cmd],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	processes.append((digits, proc))

for digits, proc in processes:
	stdout, stderr = proc.communicate()
	print(f"\n=== worker digits={digits} (returncode={proc.returncode}) ===")
	print("--- stdout ---")
	print(stdout if stdout else "<none>")
	print("--- stderr ---")
	print(stderr if stderr else "<none>")

failed = [digits for digits, proc in processes if proc.returncode != 0]
if failed:
	print(f"\nWorkers with non-zero exit code: {failed}")
else:
	print("\nAll npz workers completed successfully.")
