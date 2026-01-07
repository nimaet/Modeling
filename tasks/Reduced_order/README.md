# Piezoelectric Metamaterial Beam - Modular Analysis Scripts

This directory contains modular Python scripts for analyzing a piezoelectric metamaterial beam with Duffing nonlinear shunt circuits using a reduced order model (ROM).

## File Structure

### Core Module
- **`ROM.py`**: Base reduced order model containing:
  - Geometry and material properties
  - Modal analysis (eigenvalues, mode shapes)
  - Damping parameters (Rayleigh damping)
  - ODE system definition (`odefun`)
  - Frequency domain analysis (`modal_simulation`)
  - Time domain simulation helper (`time_domain_simulation`)

### Analysis Scripts
Each script imports from `ROM.py` and performs a specific analysis:

1. **`kp_sweep_freq_domain.py`**: 
   - Frequency domain K_p sweep
   - Compares Open Circuit (K_p=0) vs Short Circuit (K_p→∞)
   - Fast linear analysis using frequency response

2. **`modal_validation.py`**: 
   - Validates ROM against experimental data
   - Compares with COMSOL results (if available)
   - Plots FRF comparisons

3. **`time_domain_simulation.py`**: 
   - Single time domain simulation with chirp excitation
   - FFT analysis and comparison with frequency domain
   - Plots time signals and FRF

4. **`kp_sweep_time_domain.py`**: 
   - Time domain K_p sweep with nonlinearity (K_c ≠ 0)
   - FFT-based FRF extraction
   - Slower but captures nonlinear effects

5. **`amplitude_sweep.py`**: 
   - Nonlinear amplitude-dependent response
   - Investigates Duffing nonlinearity effects
   - Variable excitation amplitude

## Usage

### Basic Usage
Run any analysis script directly:
```bash
python kp_sweep_freq_domain.py
python modal_validation.py
python time_domain_simulation.py
python kp_sweep_time_domain.py
python amplitude_sweep.py
```

### Importing in Custom Scripts
```python
from ROM import modal_simulation, time_domain_simulation, N, S, L_b

# Frequency domain analysis
freq, vel_mag, disp_mag = modal_simulation(K_c=0, K_p=0.01, K_i=2100)

# Time domain analysis
sol, x_eval, disp, veloc = time_domain_simulation(
    K_c=5e10, K_p=0.01, K_i=2100, A_exc=50, f0=1000, f1=3000, t_end=1.0
)
```

## Key Parameters

### Shunt Circuit Parameters
- **K_c**: Cubic stiffness (Duffing nonlinearity) [N/m³]
- **K_p**: Proportional gain (resistance-like)
- **K_i**: Integral gain (inductance-like)
- **R_c**: Series resistance [Ω]

### Simulation Parameters
- **N**: Number of modes (40)
- **S**: Number of piezoelectric patches (31)
- **j_exc**: Excitation patch index (default: 30)
- **A_exc**: Excitation amplitude [V]
- **f0, f1**: Chirp frequency range [Hz]
- **t_end**: Simulation time [s]

## Output

All scripts save figures to `sim_dat/` directory:
- PNG figures with 300 DPI
- NPZ files with simulation results (for sweep scripts)

## Dependencies
```python
numpy
scipy
matplotlib
pandas  # only for modal_validation.py
```

## Notes

1. **Frequency vs Time Domain**:
   - Use frequency domain (`modal_simulation`) for fast linear analysis
   - Use time domain when K_c ≠ 0 (nonlinear Duffing shunt)

2. **Data Paths**:
   - Experimental data paths in `modal_validation.py` may need updating
   - COMSOL CSV files should be in `../../comsol/` directory

3. **Computational Cost**:
   - Frequency domain: ~seconds
   - Time domain single sim: ~minutes
   - Time domain sweeps: ~hours (depending on parameter range)

## Contact
For questions about the model or scripts, refer to the original notebook:
`Duffing_time_domain.ipynb`
