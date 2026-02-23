# SLURM Job Array Setup for FE Sweep

This directory contains scripts for running the amplitude/Kc sweep on a SLURM HPC cluster.

## Files

- **`run_single_sweep_case.py`** — Runs a single (amplitude, Kc) simulation by index.
- **`submit_sweep.sbatch`** — SLURM job array submission script.
- **`aggregate_sweep_results.py`** — Post-processing: aggregates intermediate `.npz` files into final pickle & JSON.
- **`README_SLURM.md`** — This file.

## Workflow

### 1. Customize the SBATCH Script

Edit `submit_sweep.sbatch` to match your cluster's resources and environment:

```bash
# Adjust these for your cluster:
#SBATCH --cpus-per-task=4          # Cores per job
#SBATCH --mem=16G                  # Memory per job
#SBATCH --time=02:00:00            # Max runtime per job
#SBATCH --array=0-13               # Number of cases (14 = 7 amps × 2 Kc cases)

# Uncomment/add any module/conda commands:
# module load python/3.10
# conda activate myenv
```

### 2. Submit the Job Array

```bash
cd Modeling/tasks/FE_studies/
sbatch submit_sweep.sbatch
```

This submits 14 parallel jobs (one per sweep case). Progress is logged to `logs/slurm_JOBID_*.log`.

Monitor:
```bash
squeue -u $USER
sacct -j <JOBID>
```

### 3. Post-Process Results

After all jobs complete, aggregate the intermediate files:

```bash
python aggregate_sweep_results.py sim_dat/Alternating_Kc_sweep_20260223_120000
```

This creates:
- `sim_dat/Alternating_Kc_sweep_20260223_120000/Alternating_Kc_.pkl`
- `sim_dat/Alternating_Kc_sweep_20260223_120000/Alternating_Kc_.json`

### 4. Load and Plot in Notebook (Cell 2)

Back in the notebook, update `RUN_FOLDER` and run the loader:

```python
RUN_FOLDER = "Alternating_Kc_sweep_20260223_120000"
```

Then run Cell 2 to load and plot. The plot is automatically saved to the same folder.

## Testing Locally

To verify the scripts work before submitting to SLURM, run a single case locally:

```bash
python run_single_sweep_case.py 0 sim_dat/test_local
```

This should save intermediate results to `sim_dat/test_local/intermediate_npz/`.

## Example: Full Workflow

```bash
# 1. Submit array job
cd ~/Projects/Metamaterial\ beam/Modeling/tasks/FE_studies/
sbatch submit_sweep.sbatch
# Output: Submitted batch job 12345678

# 2. Monitor
squeue -u $USER
# watch -n 5 'squeue -u $USER'  # For continuous monitoring

# 3. Once all jobs finish (~30 min to 2 hours depending on your cluster)
python aggregate_sweep_results.py sim_dat/Alternating_Kc_sweep_20260223_120000

# 4. Load in Jupyter
# Open notebook, Cell 2, change RUN_FOLDER, run it
```

## Customization

### Different Sweep Configuration

If you change `amp_list` or `kc_magnitudes` in the notebook:

1. Update the same arrays in `run_single_sweep_case.py` (lines ~50-60)
2. Update `#SBATCH --array=0-N` to match total count = `len(amp_list) * len(kc_magnitudes)`
3. Update the same arrays in `aggregate_sweep_results.py` (lines ~20-40)

### Running Subset of Cases

To run only cases 0-6 (first Kc case with all amplitudes):

```bash
sbatch --array=0-6 submit_sweep.sbatch
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Jobs fail with "module not found" | Add proper module load or activate environment in `.sbatch` before running Python |
| Output directory already exists | Script creates nested `_1`, `_2` etc. automatically; or delete old run folder first |
| Intermediate files not found after job completes | Check `logs/slurm_*.err` for runtime errors; verify `sim_dat/<run>/intermediate_npz/` exists |
| Python import errors | Ensure `Modeling/` module is importable; verify path setup in `run_single_sweep_case.py` line ~18 |

## Notes

- Each job is **independent** and parallelizable—no communication between tasks.
- Results are **streamed directly to disk** (no memory accumulation).
- Failed jobs log errors in `logs/slurm_*.err`; aggregator skips them gracefully.
- After aggregation, you can safely delete the `intermediate_npz/` folder to save space.
