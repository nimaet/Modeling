"""
Flexible 2D Parameter Sweep for ROM simulations
Easily configure which two parameters to sweep over
Results are saved in a simple flat format for easy post-processing
"""

from ROM import run_time_sim, N, S, L_b
import numpy as np
from joblib import Parallel, delayed
import itertools
from datetime import datetime

# ======================================================================
# CONFIGURATION: Edit this section to change sweep parameters
# ======================================================================

# Fixed parameters (not being swept)
FIXED_PARAMS = {
    'K_i': 0.001,
    'K_p': 0.03,
    'K_c': 5e11,
    't_end': 1.0,
    'f0': 1000,
    'f1': 4500,
}

# Define the two parameters to sweep
# Each sweep parameter is defined as: (param_name, array_of_values)
SWEEP_PARAM1 = ('amp', np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125)
SWEEP_PARAM2 = ('K_p', np.arange(0.02, 0.18, 0.01))

# Alternative sweep configurations (comment/uncomment as needed):
# SWEEP_PARAM1 = ('amp', np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]) * 125)
# SWEEP_PARAM2 = ('K_c', np.array([2e9, 4e9, 8e9, 1e10, 1.2e10, 1.6e10, 2e10, 2.25e10, 2.5e10]))

# SWEEP_PARAM1 = ('K_p', np.arange(0.01, 0.2, 0.01))
# SWEEP_PARAM2 = ('K_c', np.array([0, 1e10, 5e10, 1e11]))

# Parallel processing settings
N_JOBS = 8
VERBOSE = 10

# Output file name (will be auto-generated if None)
OUTPUT_FILE = None  # e.g., 'sim_dat/custom_sweep.npz'

# ======================================================================
# END CONFIGURATION
# ======================================================================

# Unpack sweep parameter names and values
param1_name, param1_values = SWEEP_PARAM1
param2_name, param2_values = SWEEP_PARAM2

print("="*60)
print("2D PARAMETER SWEEP CONFIGURATION")
print("="*60)
print(f"Sweep Parameter 1: {param1_name}")
print(f"  Values: {param1_values}")
print(f"Sweep Parameter 2: {param2_name}")
print(f"  Values: {param2_values}")
print(f"\nFixed Parameters:")
for key, val in FIXED_PARAMS.items():
    print(f"  {key}: {val}")
print(f"\nTotal simulations: {len(param1_values) * len(param2_values)}")
print("="*60)

# Setup
t_eval = np.arange(0, FIXED_PARAMS['t_end'], 1/FIXED_PARAMS['f1']/20)
x_eval = np.linspace(0, L_b, 100)


def run_single_simulation(param1_val, param2_val):
    """
    Run a single simulation with specified parameter values
    
    Parameters:
        param1_val: Value for first sweep parameter
        param2_val: Value for second sweep parameter
    
    Returns:
        Dictionary containing simulation results and parameter values
    """
    # Build complete parameter dictionary
    params = FIXED_PARAMS.copy()
    params[param1_name] = param1_val
    params[param2_name] = param2_val
    
    # Extract amplitude (might be a sweep parameter or fixed)
    amp = params.get('amp', 50)  # Default amplitude if not specified
    
    print(f"  {param1_name}={param1_val:.3g}, {param2_name}={param2_val:.3g}")
    
    # Define excitation function
    def v_exc(t, A_exc=amp, f0=params['f0'], f1=params['f1'], t_end=params['t_end']):
        return A_exc * np.sin(2*np.pi*(f0 + t*(f1-f0)/t_end) * t)
    
    # Run simulation
    res = run_time_sim(
        v_exc=v_exc,
        K_c=params['K_c'],
        K_p=params['K_p'],
        K_i=params['K_i'],
        t_end=params['t_end'],
        x_eval=x_eval,
        t_eval=t_eval
    )
    
    # Return results with parameter information
    return {
        param1_name: param1_val,
        param2_name: param2_val,
        'freq': res['freq'],
        'FRF': res['FRF'],
        't': res['t'],
        'veloc': res['veloc'],
        'Y': res['Y'],
        'X': np.fft.fft(v_exc(res['t'], A_exc=amp)),
        'all_params': params  # Store all parameters for reference
    }


# ======= Run 2D sweep in parallel =======
print("\nRunning 2D parameter sweep in parallel...")
param_pairs = list(itertools.product(param1_values, param2_values))

sim_results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
    delayed(run_single_simulation)(p1, p2) 
    for p1, p2 in param_pairs
)

print(f"\nCompleted {len(sim_results)} simulations")

# ======= Save results =======
# Generate output filename if not provided
if OUTPUT_FILE is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f'sim_dat/sweep_{param1_name}_{param2_name}_{timestamp}.npz'

# Extract parameter arrays for easy access
param1_array = np.array([res[param1_name] for res in sim_results])
param2_array = np.array([res[param2_name] for res in sim_results])

# Save with descriptive key names
np.savez(
    OUTPUT_FILE,
    # Parameter information
    param1_name=param1_name,
    param2_name=param2_name,
    param1_values=param1_values,
    param2_values=param2_values,
    param1_array=param1_array,  # Flattened array matching results order
    param2_array=param2_array,  # Flattened array matching results order
    fixed_params=FIXED_PARAMS,
    # Simulation results (as flat arrays)
    results=sim_results,
    # Metadata
    n_param1=len(param1_values),
    n_param2=len(param2_values),
    timestamp=timestamp if OUTPUT_FILE is None else datetime.now().strftime("%Y%m%d_%H%M%S")
)

print(f"\n{'='*60}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"{'='*60}")
print(f"Sweep 1: {param1_name} [{param1_values.min():.3g} - {param1_values.max():.3g}]")
print(f"Sweep 2: {param2_name} [{param2_values.min():.3g} - {param2_values.max():.3g}]")
print(f"Fixed parameters: {', '.join([f'{k}={v}' for k, v in FIXED_PARAMS.items()])}")
print(f"Total simulations: {len(sim_results)}")
print(f"{'='*60}")
