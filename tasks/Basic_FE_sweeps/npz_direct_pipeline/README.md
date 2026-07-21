# Direct-npz Duffing Sweep Workflow

Alternative to the `collect_nD_sweep_results.py` -> `results.pkl` pipeline.
Skips the collection step entirely: downloads and loads the raw
per-simulation `npz` files that `generic_nD_sweep_SLURMarray.py` already
writes, and only decompresses the (large) time/frequency-domain arrays for
the Kc groups you actually plot.

**Nothing in the original pipeline is modified.** `collect_nD_sweep_results.py`,
`download_sweep_results.py`, and
`results/Duffing_paper_visuals/sim_figs/plot_duffing.py` /
`duffing_plot_helpers.py` are untouched and still work exactly as before —
this folder is a parallel, opt-in path.

## Why

- `collect_nD_sweep_results.py` loads every sim's full `params` + `data`
  into memory just to write one combined `results.pkl`.
- `plot_duffing.py` then loads that entire pickle back into memory, even
  though `KC_SELECTION` only plots 2-3 of the Kc groups.
- This workflow never materializes a combined file. It reads only the small
  `params` dict from every sim (to figure out Kc/amp grouping), and reads
  the large `data` arrays only for the Kc groups you select.

## Files

- **`download_npz_results.py`** — like `download_sweep_results.py`, but
  rsyncs `config.json` + every `npz/*.npz` instead of `*.pkl`, and does the
  npz transfer as `N_WORKERS` concurrent rsync streams instead of one (see
  "Parallel download" below).
- **`npz_direct_helpers.py`** — the loading logic:
  - `npz_kc_index(run_dir)` — Pass 1. Opens every `npz`, reads only
    `params["K_c"]` / `params["amp"]`, and groups sims by Kc in native
    (ascending sim-index) order. Cheap — never touches `data`.
  - `collect_unique_amps_from_index(kc_index)` — amplitude list for the
    colorbar, from the same cheap pass.
  - `choose_kc_labels(labels, selection)` — same semantics as the original
    `duffing_plot_helpers.choose_kc_labels`.
  - `load_npz_results_by_kc(kc_index, selected_labels)` — Pass 2. Loads
    `data` only for the selected Kc labels. Returns the same
    `{label: {kc_label, kc_vec, amps, data}}` shape as
    `duffing_plot_helpers.to_results_by_kc`, so it's a drop-in replacement.
- **`plot_duffing_from_npz.py`** — copy of `plot_duffing.py` with only the
  data-loading section changed (see below); all plotting/colorbar/MATLAB
  export code is identical.

## Usage

1. Run `generic_nD_sweep_SLURMarray.py` on the cluster as usual — unchanged.
2. **Skip `collect_nD_sweep_results.py`.** Don't run it.
3. Edit the settings block at the top of `download_npz_results.py`
   (`REMOTE_USER`, `REMOTE_HOST`, `dirnaeme`, `REMOTE_DIR`, `DEST_WSL`) to
   match your run, then execute it (same prerequisites as the original
   script: WSL + `rsync` + SSH key access to the cluster). This downloads
   `config.json` and `npz/sim_*.npz` into `DEST_WSL`.
4. Open `plot_duffing_from_npz.py` in the VS Code Interactive Window. Set
   `RUN_DIR` to the downloaded run folder (the directory that contains
   `config.json` and `npz/` — **not** a `results.pkl` path), adjust
   `KC_SELECTION` / frequency limits / other settings as needed, and run it.

## Parallel download

Downloading is network/round-trip-latency bound, not CPU bound, so more
cores don't speed it up directly. But with thousands of small npz files, a
single rsync stream leaves most of that per-file round-trip latency
unhidden — it transfers one file, waits, transfers the next. Running several
rsync streams concurrently overlaps those round trips instead of queuing
them.

`download_npz_results.py` shards the npz transfer by the **last digit of
each sim's zero-padded index** (`sim_00000.npz` ... `sim_99999.npz`) into
`N_WORKERS` buckets (default 4), and launches that many concurrent `rsync`
processes via WSL, each restricted to its own bucket with an
`--include='*[<digits>].npz'` pattern. `config.json` is fetched once,
separately, since it's a single small file unrelated to per-sim sharding.

- Raise `N_WORKERS` (up to 10, since sharding is by one decimal digit) if
  the cluster/network tolerates more concurrent SSH sessions and the run has
  enough sims to benefit; lower it if you see connection throttling or the
  login node complaining about too many sessions.
- This doesn't help (and can slightly hurt) for a run with very few sims —
  the overlap only pays off once there are enough files that per-file
  latency, not raw bandwidth, is the bottleneck.
- rsync's `-z` (compression) is deliberately not used: `npz` files are
  already `savez_compressed`, so compressing them again just burns CPU for
  no size reduction.

## Notes

- **`KC_SELECTION` indices are unchanged** from the `results.pkl` workflow:
  `npz_kc_index` walks `npz/sim_*.npz` in ascending sim-index order, which
  is the same order `collect_nD_sweep_results.py`'s `results` list ended up
  in (it iterates sorted, zero-padded `status/result_*.json` files). Native
  Kc-group order is therefore identical either way.
- A sim whose `npz` is missing, corrupt, or missing `K_c`/`amp` in `params`
  is silently skipped in `npz_kc_index` — mirroring how the original
  `to_results_by_kc` silently skips records it can't find those keys for.
- `config.json`'s `sweep_spec` (with the `target` field, including the
  multi-target list form used for frequency sweeps) is not needed for
  grouping — grouping reads resolved values straight out of each sim's
  `params`. `npz_direct_helpers.load_config(run_dir)` is provided if you
  need `config.json`'s metadata (e.g. `time`/`fe_params`/`base_params`) for
  something else.
- This only helps if you actually restrict `KC_SELECTION` to fewer Kc groups
  than exist in the run — if you select all of them, memory use converges
  to the same as the `results.pkl` flow (all `data` gets loaded either way).
