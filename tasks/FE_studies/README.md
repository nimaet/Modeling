# Finite Element Analysis - Piezoelectric Metamaterial Beam

This directory contains Finite Element (FE) analysis scripts for the piezoelectric metamaterial beam with nonlinear shunt circuits, complementing the Reduced Order Model (ROM) approach.

## Files

### Analysis Scripts

1. **`FE_2D_sweep_amp_Kc.py`**: 
   - 2D parameter sweep over excitation amplitude and K_c (cubic stiffness)
   - Parallel execution using joblib
   - Nonlinear Duffing shunt circuit analysis
   - Similar to the ROM version but using FE discretization
   - Outputs: FRF plots for each K_c value and NPZ data file

### Notebooks

1. **`Kc_amp_2d_sweep.ipynb`**: 
   - Interactive notebook for 2D sweep studies
   - Includes comparison with ROM and experimental data
   - Animation capabilities for visualizing beam motion
   - Contains both 1D amplitude sweep and full 2D sweep sections

## Key Differences from ROM

### Advantages of FE:
- More accurate spatial discretization
- No assumption on mode shapes
- Can capture local effects and discontinuities
- Better for very short beams or complex geometries

### Computational Cost:
- FE simulations are slower than ROM
- Time domain FE: ~10-30 minutes per simulation (depending on resolution)
- ROM: ~1-5 minutes per simulation
- Use parallel execution to compensate (joblib)

## Usage

### Running the 2D Sweep Script

```bash
python FE_2D_sweep_amp_Kc.py
```

This will:
1. Run simulations for all amplitude and K_c combinations
2. Generate FRF plots organized by K_c
3. Save results to `sim_dat/FE_2D_sweep_softening_Kp=*.npz`
4. Save figure to `sim_dat/FE_2D_sweep_amp_Kc_*.png`

### Using the Notebook

Open `Kc_amp_2d_sweep.ipynb` in Jupyter/VS Code and run cells sequentially.

## Parameters

### FE Model Setup
- **n_el_patch**: Number of elements per piezo patch (default: 3)
- **n_el_gap**: Number of elements per gap between patches (default: 1)
- Higher values = more accuracy but slower

### Shunt Circuit Parameters
- **K_c**: Cubic stiffness (Duffing nonlinearity) [N/m³]
  - Negative values: softening behavior
  - Positive values: hardening behavior
- **K_p**: Proportional gain (resistance-like)
- **K_i**: Integral gain (inductance-like)
- **R_c**: Series resistance [Ω]

### Simulation Parameters
- **dt**: Time step size [s] - typically 1/(f1*20)
- **t_end**: Simulation duration [s]
- **f0, f1**: Chirp frequency range [Hz]
- **j_exc**: Excitation patch index (0-30 for 31-patch system)

### Newmark-β Time Integration
- **beta**: 0.25 (unconditionally stable, 2nd order accurate)
- **gamma**: 0.5 (no numerical damping)
- **newton_tol**: Convergence tolerance (1e-8)
- **newton_maxiter**: Max Newton iterations per time step (8)

## Output

### Generated Files in `sim_dat/`:
- PNG figures with 300 DPI showing FRF vs frequency
- NPZ files containing:
  - `amp_list`: Array of excitation amplitudes
  - `Kc_list`: Array of K_c values
  - `K_p`, `K_i`: Shunt parameters used
  - `results_by_Kc`: Dictionary with FRF data organized by K_c

## Comparison with ROM

To compare FE and ROM results:
```python
import numpy as np

# Load FE results
fe_data = np.load('sim_dat/FE_2D_sweep_softening_Kp=0.030_Ki=2100.npz', allow_pickle=True)

# Load ROM results
rom_data = np.load('../Reduced_order/sim_dat/2D_sweep_softening_Kp=0.030_Ki=2100.npz', allow_pickle=True)

# Compare FRFs
fe_results = fe_data['results_by_Kc'].item()
rom_results = rom_data['results_by_Kc'].item()
```

## Dependencies

```python
numpy
scipy
matplotlib
joblib       # for parallel execution
pathlib      # for path handling
```

## Performance Tips

1. **Parallel Execution**: 
   - Adjust `n_jobs` in `Parallel()` based on your CPU cores
   - Default is 8, increase if you have more cores
   - Don't exceed physical core count

2. **Time Step Selection**:
   - Rule of thumb: dt ≤ 1/(20*f_max)
   - Smaller dt = more accurate but slower
   - For stability: dt < T_min/10 where T_min is smallest period

3. **Mesh Refinement**:
   - Start with n_el_patch=3, n_el_gap=1
   - Increase if convergence study shows insufficient accuracy
   - Each doubling approximately quadruples computation time

## Notes

1. **Nonlinear Analysis**: 
   - FE time domain required when K_c ≠ 0
   - Newton-Raphson iteration at each time step
   - Monitor convergence (newton_maxiter warnings)

2. **Memory Usage**:
   - Full 2D sweep stores all time histories
   - For 63 simulations: ~1-5 GB RAM depending on t_end
   - Results dictionary can be compressed after processing

3. **Frequency Resolution**:
   - FFT resolution: Δf = 1/t_end
   - Longer t_end = better frequency resolution
   - Trade-off with computation time

## Troubleshooting

### Newton-Raphson Not Converging
- Reduce time step dt
- Increase newton_maxiter
- Check if K_c value is too large
- Verify initial conditions

### Memory Issues
- Reduce number of parallel jobs
- Process results in batches
- Don't store full time histories if not needed

### Slow Execution
- Check parallel execution is working (verbose=10)
- Reduce t_end for testing
- Use coarser mesh for initial exploration

## Contact

For questions about FE implementation or comparison with ROM results, refer to:
- ROM scripts in `../Reduced_order/`
- Main FE1.py model in `../../models/FE1.py`
