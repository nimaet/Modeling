# Refactored piezo patch optimizer bundle

Files:

- `Modeling/models/beam_properties.py` — cleaned physical parameter class and section property utility.
- `Modeling/models/FE3.py` — cleaned FE model, geometry builders, ODE builders, tolerance-based piezo node matching.
- `Modeling/models/FE_helpers.py` — cleaned FRF/Newmark helpers with optional progress/debug output.
- `Modeling/models/piezo_patch_optimizer.py` — new standalone optimizer module.
- `run_refactored_optimizer_example.py` — minimal driver script showing the intended notebook usage pattern.

The optimizer design vector is generalized to `Np` patches:

```python
z = [L1, g12, L2, g23, ..., g(Np-1,Np), LNp]
```

The generated region sequence is:

```python
['piezo', 'substrate', 'piezo', 'substrate', ...]
```

For the current workflow, use `Np=3` and `phase_mode='binary'` to match the previous notebook behavior.
