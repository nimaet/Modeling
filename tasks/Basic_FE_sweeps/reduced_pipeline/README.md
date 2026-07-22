# Reduced-npz Sweep Workflow

Third alternative alongside `collect_nD_sweep_results.py` and
`npz_direct_pipeline/`. **Nothing in either of those is modified** -- this
folder is a parallel, opt-in path.

## Why this exists

`generic_nD_sweep_SLURMarray.py` writes one full time-series `npz` per sim
(`t`, `u_dot`, `v`, `v_exc`). Two things you might want from that:

- `collect_nD_sweep_results.py` loads every sim's full `data` into memory to
  build one combined `results.pkl` -- expensive locally, and you still need
  the raw npz's present to run it.
- `npz_direct_pipeline/` avoids materializing everything in memory, but its
  `download_npz_results.py` step still transfers **every raw npz** to local
  disk. For a large sweep with long time traces, that's the actual blocker:
  it doesn't fit on local storage, and downloading it isn't a memory problem
  you can code around -- it's a "the bytes don't fit" problem.

This folder solves that by never downloading the raw npz's at all. The
reduction happens **on the cluster**, where the npz's already live:

```
generic_nD_sweep_SLURMarray.py  -->  npz/sim_*.npz (stays on cluster)
                                          |
                                          v  reduce_nD_sweep_results.py (runs on cluster)
                                          |
                                          v
                          reduced_results__v001.pkl (tiny, one per version)
                                          |
                                          v  download_reduced_results.py (local)
                                          |
                                          v
                          plot_bifurcation_from_reduced.py (local)
```

The only thing that crosses the network is a handful of scalars/small arrays
per sim, not full time series.

## Design: why reduction lives here, not in the SLURM array script

Baking reduction directly into `generic_nD_sweep_SLURMarray.py` (compute
metrics right after `solve_newmark`, skip writing raw arrays) would be
cheaper, but it's rigid: realizing you want one more metric means re-running
every expensive FE solve. Keeping reduction as its own decoupled step means
you can add a metric to `reduction_spec.py` and re-run
`reduce_nD_sweep_results.py` in seconds against the existing raw npz's --
no re-simulation. The raw npz's on the cluster remain the source of truth
for anything you didn't think to extract yet.

## Files

- **`reduction_spec.py`** -- the only file you should need to edit to change
  *what* gets extracted. `REDUCTION_SPEC` maps a name to a function
  `fn(data, params, sweep_entry) -> scalar or small array`. Which DOFs get
  reduced is explicit, not auto-detected: `MECH_DOFS`/`ELEC_DOFS` are editable
  lists of column indices into `u_dot`/`v` (mechanical/electrical DOFs
  respectively). Ships with a starting example for a frequency sweep:
  - `response_amplitude`: array shaped `(len(MECH_DOFS),)` -- steady-state
    peak-to-peak/2 of `u_dot` at each selected mechanical DOF, for a
    backbone/FRF curve.
  - `poincare_samples`: array shaped `(len(MECH_DOFS), n_periods)` -- a
    stroboscopic (Poincare) section of `u_dot` at each selected mechanical
    DOF, one sample per forcing period over the last `N_POINCARE_PERIODS`
    periods of steady state. Multiple distinct values at one frequency
    indicate period-doubling/chaos; this is what the bifurcation diagram
    scatters.
  - `voltage_amplitude`: array shaped `(len(ELEC_DOFS),)` -- electrical
    analog of `response_amplitude`, for the selected piezo-interface DOFs.
  - `excitation_freq`: the x-axis for the plots above.
- **`reduce_nD_sweep_results.py`** -- runs on the cluster next to `npz/`.
  Applies `REDUCTION_SPEC` to every successful sim (parallelized with
  joblib), and writes one consolidated `reduced_results__vNNN.pkl` per run,
  in the same `meta`/`sweep`/`results`/`failed` shape documented in
  `../FE_SWEEP_DATA_FORMAT.md`, except `results[i]["reduced"]` holds the
  small reducer outputs instead of `results[i]["data"]`'s full arrays.
  - Each run gets the next sequential version number for that run directory
    (`v001`, `v002`, ...) -- editing `reduction_spec.py` (reducers, DOFs, or
    window settings) and re-running never silently overwrites an older
    reduction. `reduction_manifest.json` (written alongside) records every
    version produced for that run, with timestamps, reducer names, and the
    `MECH_DOFS`/`ELEC_DOFS`/window settings used.
- **`download_reduced_results.py`** -- like `download_sweep_results.py`, but
  only pulls `config.json`, `reduction_manifest.json`, and
  `reduced_results__*.pkl`. No parallel-rsync sharding needed (unlike
  `download_npz_results.py`) -- these are a handful of small files.
- **`plot_bifurcation_from_reduced.py`** -- loads the (small) reduced pickle
  and plots the backbone amplitude curve and the bifurcation diagram
  side by side, for one `MECH_DOFS` entry at a time (`DOF_INDEX`, a position
  in that list, not an absolute node index). Defaults to the most recent
  version per `reduction_manifest.json`; set `VERSION` explicitly to pin an
  older one.
- **`runPython_reduce.sbatch`** / **`submit_with_reduce.sh`** -- cluster-side
  job plumbing, mirroring `runPython_collect.sbatch` /
  `submit_with_collector.sh`.

## Usage

1. Run `generic_nD_sweep_SLURMarray.py` on the cluster as usual (via
   `submit_with_reduce.sh`, or your existing submission flow) -- unchanged.
2. Edit `reduction_spec.py` if you want different/additional metrics.
3. On the cluster, from `Basic_FE_sweeps/`, run:
   `sbatch reduced_pipeline/runPython_reduce.sbatch`
   (after setting `ARRAY_JOB_ID`/`RUN_DIR` at the top of
   `reduce_nD_sweep_results.py` to match the sweep run, same as
   `collect_nD_sweep_results.py`'s config block).
4. Locally, edit `download_reduced_results.py`'s settings block
   (`REMOTE_USER`, `REMOTE_HOST`, `dirnaeme`, `REMOTE_DIR`, `DEST_WSL`) and
   run it -- pulls only the small reduced outputs.
5. Open `plot_bifurcation_from_reduced.py`, set `RUN_DIR` to the downloaded
   run folder, and run it in the VS Code Interactive Window.

## Notes

- If you later want a metric that requires data no reducer captured, you
  still need the raw npz's for that run -- this pipeline trades some
  flexibility for never moving the big arrays over the network. The
  `npz_direct_pipeline/` workflow is still there for cases where you
  genuinely need full time/frequency-domain data locally for a few Kc/freq
  groups.
- `MECH_DOFS`/`ELEC_DOFS` (in `reduction_spec.py`) are explicit column-index
  lists, not auto-detected -- but they index into the *already-reduced*
  `u_dot`/`v` arrays FE_helpers.solve_newmark writes into the npz, not raw
  ODE state indices. `MECH_DOFS` indexes free nodes' transverse
  velocity/displacement only (rotational DOFs are dropped by a `::2` stride
  before this file ever sees the data); `ELEC_DOFS` indexes the *free*
  (non-excited) piezo channels, i.e. `j_exc` is a different, electrical-only
  index space with the excited channel(s) already removed. If you change the
  mesh or `j_exc` in `generic_nD_sweep_SLURMarray.py`, re-check what these
  indices actually land on before trusting them.
- `STEADY_STATE_FRACTION` controls how much of the start of each time trace
  is discarded as transient before computing amplitude; `N_POINCARE_PERIODS`
  controls how many stroboscopic samples are kept per sim for the
  bifurcation diagram. Both are module-level constants in
  `reduction_spec.py`.
