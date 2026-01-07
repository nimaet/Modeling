"""
Plot results from 2D parameter sweep
Automatically detects sweep parameters and organizes plots accordingly
Supports grouping by either parameter for flexible visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import os
from ROM import modal_simulation

# ======================================================================
# CONFIGURATION
# ======================================================================

# Path to sweep results file
# NPZ_FILE = 'sim_dat/2D_sweep_Kp=0.030_Ki=2100.npz'
# NPZ_FILE = 'sim_dat/sweep_amp_K_p_20251220_183931.npz'
NPZ_FILE = 'sim_dat\sweep_amp_K_p_20251220_200148.npz'
NPZ_FILE = "sim_dat\sweep_amp_K_p_20251220_211429.npz"
NPZ_FILE = "sim_dat\sweep_amp_K_p_20251220_223004.npz"
# Grouping: which parameter to use for separate figures
# Options: 'param1', 'param2', or 'auto' (uses param2 by default)
GROUP_BY = 'param2'

# Plot settings
PLOT_CONFIG = {
    'figsize': (8, 5),
    'freq_range': [1000, 4500],
    'frf_range': [1e-5, 8e-4],
    'colormap': 'viridis',
    'dpi': 300,
}

# Reference lines to add (set to True to compute and plot)
ADD_REFERENCES = {
    'short_circuit': True,      # K_p=100, K_c=0
    'linear': True,             # K_c=0, same K_p as fixed
    'linear_lowdamp': True,     # K_c=0, K_p=0.01
}

# Output directory (None = auto-generate from input file location)
OUTPUT_DIR = None

# Generate separate legend file
SEPARATE_LEGEND = True

# ======================================================================
# END CONFIGURATION
# ======================================================================


def load_sweep_data(npz_file):
    """
    Load and organize sweep data from NPZ file
    Returns structured data regardless of which parameters were swept
    """
    print(f"Loading data from: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    # Check if this is old format (results_by_Kc) or new format (flat results)
    if 'results' in data.files:
        # New format: flat results array
        param1_name = str(data['param1_name'])
        param2_name = str(data['param2_name'])
        results = data['results']
        fixed_params = data['fixed_params'].item()
        
        print(f"  Sweep parameters: {param1_name} × {param2_name}")
        print(f"  Total results: {len(results)}")
        
        return {
            'param1_name': param1_name,
            'param2_name': param2_name,
            'param1_values': data['param1_values'],
            'param2_values': data['param2_values'],
            'results': results,
            'fixed_params': fixed_params,
            'format': 'new'
        }
    
    elif 'results_by_Kc' in data.files or 'results_by_param2' in data.files:
        # Old format: nested dictionary structure
        results_key = 'results_by_Kc' if 'results_by_Kc' in data.files else 'results_by_param2'
        results_dict = data[results_key].item()
        
        # Try to infer parameter names
        if 'amp_list' in data.files:
            param1_name = 'amp'
            param1_values = data['amp_list']
        elif 'param1_list' in data.files:
            param1_name = 'param1'
            param1_values = data['param1_list']
        else:
            param1_name = 'param1'
            param1_values = None
        
        if 'Kc_list' in data.files:
            param2_name = 'K_c'
            param2_values = data['Kc_list']
        elif 'param2_list' in data.files:
            param2_name = 'param2'
            param2_values = data['param2_list']
        else:
            param2_name = 'param2'
            param2_values = np.array(list(results_dict.keys()))
        
        # Extract fixed parameters if available
        fixed_params = {}
        for key in ['K_p', 'K_i', 'K_c']:
            if key in data.files:
                fixed_params[key] = data[key].item() if hasattr(data[key], 'item') else data[key]
        
        print(f"  Detected old format")
        print(f"  Inferred parameters: {param1_name} × {param2_name}")
        
        return {
            'param1_name': param1_name,
            'param2_name': param2_name,
            'param1_values': param1_values,
            'param2_values': param2_values,
            'results_dict': results_dict,
            'fixed_params': fixed_params,
            'format': 'old'
        }
    
    else:
        raise ValueError("Unrecognized NPZ file format")


def organize_data_for_plotting(sweep_data, group_by='param2'):
    """
    Organize sweep results into groups for plotting
    
    Parameters:
        sweep_data: Dictionary from load_sweep_data()
        group_by: 'param1' or 'param2' - which parameter defines separate figures
    
    Returns:
        Dictionary with organized data ready for plotting
    """
    if sweep_data['format'] == 'old':
        # Old format: already grouped by param2
        results_dict = sweep_data['results_dict']
        
        # Create standardized structure
        organized = {
            'groups': {},
            'param1_name': sweep_data['param1_name'],
            'param2_name': sweep_data['param2_name'],
            'group_param': sweep_data['param2_name'],
            'series_param': sweep_data['param1_name'],
        }
        
        for param2_val, data in results_dict.items():
            organized['groups'][param2_val] = {
                'series_values': data['amps'],
                'freq': data['freq'],
                'FRFs': data['FRFs'],
            }
        
        return organized
    
    # New format: flat results, need to group
    results = sweep_data['results']
    param1_name = sweep_data['param1_name']
    param2_name = sweep_data['param2_name']
    
    # Create DataFrame for easy grouping
    df_data = {
        param1_name: [r[param1_name] for r in results],
        param2_name: [r[param2_name] for r in results],
        'result_idx': range(len(results))
    }
    df = pd.DataFrame(df_data)
    
    # Determine grouping parameter
    if group_by == 'auto':
        group_by = 'param2'
    
    group_param_name = param2_name if group_by == 'param2' else param1_name
    series_param_name = param1_name if group_by == 'param2' else param2_name
    
    # Group and sort
    grouped = df.groupby(group_param_name)
    
    organized = {
        'groups': {},
        'param1_name': param1_name,
        'param2_name': param2_name,
        'group_param': group_param_name,
        'series_param': series_param_name,
    }
    
    for group_val, group_df in grouped:
        # Sort by series parameter
        group_df_sorted = group_df.sort_values(series_param_name)
        
        # Extract data
        series_vals = group_df_sorted[series_param_name].values
        indices = group_df_sorted['result_idx'].values
        
        FRFs = [results[idx]['FRF'] for idx in indices]
        freq = results[indices[0]]['freq']
        
        organized['groups'][group_val] = {
            'series_values': series_vals,
            'freq': freq,
            'FRFs': FRFs,
        }
    
    return organized


def compute_reference_lines(fixed_params, add_refs):
    """Compute reference FRF curves"""
    refs = {}
    
    K_i = fixed_params.get('K_i', 2100)
    K_p = fixed_params.get('K_p', 0.03)
    
    if add_refs.get('short_circuit', False):
        print("Computing short circuit reference (K_p=100, K_c=0)...")
        freq, FRF, _ = modal_simulation(K_c=0, K_p=100, K_i=K_i)
        refs['short_circuit'] = {'freq': freq, 'FRF': FRF, 
                                  'label': 'Short circuit', 'color': 'purple', 'linestyle': '--'}
    
    if add_refs.get('linear', False):
        print(f"Computing linear reference (K_c=0, K_p={K_p})...")
        freq, FRF, _ = modal_simulation(K_c=0, K_p=0.2, K_i=0)
        refs['linear'] = {'freq': freq, 'FRF': FRF,
                         'label': 'Linear', 'color': 'r', 'linestyle': '--'}
    
    if add_refs.get('linear_lowdamp', False):
        print("Computing linear low damping reference (K_c=0, K_p=0.01)...")
        freq, FRF, _ = modal_simulation(K_c=0, K_p=0.001, K_i=K_i)
        refs['linear_lowdamp'] = {'freq': freq, 'FRF': FRF,
                                  'label': 'Linear low damping', 'color': 'c', 'linestyle': '--'}
    
    return refs


def plot_sweep_results(organized_data, references, config, output_dir, separate_legend=True):
    """
    Generate plots for sweep results
    
    Parameters:
        organized_data: Dictionary from organize_data_for_plotting()
        references: Dictionary of reference lines from compute_reference_lines()
        config: Plot configuration dictionary
        output_dir: Directory to save plots
        separate_legend: Whether to create a separate legend figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    groups = organized_data['groups']
    series_param = organized_data['series_param']
    group_param = organized_data['group_param']
    
    # Get all unique series values for consistent coloring
    all_series_values = []
    for group_data in groups.values():
        all_series_values.extend(group_data['series_values'])
    unique_series = np.sort(np.unique(all_series_values))
    
    # Create color mapping
    cmap = cm.get_cmap(config['colormap'], len(unique_series))
    series_to_color = {val: cmap(i / max(len(unique_series)-1, 1)) 
                      for i, val in enumerate(unique_series)}
    
    # Plot each group
    for group_val in sorted(groups.keys()):
        group_data = groups[group_val]
        
        plt.figure(figsize=config['figsize'])
        
        # Plot series lines
        for series_val, FRF in zip(group_data['series_values'], group_data['FRFs']):
            color = series_to_color[series_val]
            line, = plt.semilogy(
                group_data['freq'], FRF, '-',
                color=color,
                label=f"{series_param}={series_val:.2f}"
            )
            # if np.isclose(series_val, 0.08, atol=1e-3):
            #            color = series_to_color[series_val]
            #            line, = plt.semilogy(
            #     group_data['freq'], FRF, '-',
            #     color=color,
            #     label=f"{series_param}={series_val:.2f}"
            # )
                # line.set_linestyle('None')
                # line.set_visible(False)
                

        
        # Add reference lines
        for ref_data in references.values():
            plt.semilogy(ref_data['freq'], ref_data['FRF'],
                        linestyle=ref_data['linestyle'],
                        color=ref_data['color'],
                        linewidth=1.5,
                        label=ref_data['label'])
        
        plt.xlabel("Frequency [Hz]")
        plt.ylabel('Average FRF Velocity / Voltage (m/s/V)')
        plt.title(f"{group_param} = {group_val:.3g}")
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.xlim(config['freq_range'])
        plt.ylim(config['frf_range'])
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f"{group_param}_{group_val:.3e}.png")
        plt.savefig(save_path, dpi=config['dpi'])
        print(f"Saved: {save_path}")
        plt.legend(loc='best', fontsize=8)
        plt.show()
    
    # Create separate legend
    if separate_legend:
        series_handles = [mpatches.Patch(color=series_to_color[val], 
                                        label=f"{series_param}={val:.2f}") 
                         for val in unique_series]
        ref_handles = [Line2D([], [], color=ref['color'], linestyle=ref['linestyle'],
                             label=ref['label']) 
                      for ref in references.values()]
        
        fig_leg = plt.figure(figsize=(5, 3))
        fig_leg.legend(handles=series_handles + ref_handles, loc='center', 
                      ncol=2, fontsize=9, frameon=True)
        plt.axis('off')
        plt.tight_layout()
        legend_path = os.path.join(output_dir, 'legend.png')
        plt.savefig(legend_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Saved legend: {legend_path}")
        plt.show()


# ======================================================================
# MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":
    print("="*60)
    print("2D PARAMETER SWEEP PLOTTER")
    print("="*60)
    
    # Load data
    sweep_data = load_sweep_data(NPZ_FILE)
    
    # Organize for plotting
    organized = organize_data_for_plotting(sweep_data, group_by=GROUP_BY)
    
    print(f"\nOrganization:")
    print(f"  Groups ({organized['group_param']}): {len(organized['groups'])}")
    print(f"  Series per group ({organized['series_param']}): "
          f"{len(list(organized['groups'].values())[0]['series_values'])}")
    
    # Compute reference lines
    references = compute_reference_lines(sweep_data['fixed_params'], ADD_REFERENCES)
    
    # Determine output directory
    if OUTPUT_DIR is None:
        output_dir = os.path.join(os.path.dirname(NPZ_FILE), 'plots')
    else:
        output_dir = OUTPUT_DIR
    
    # Generate plots
    print(f"\nGenerating plots...")
    plot_sweep_results(organized, references, PLOT_CONFIG, output_dir, SEPARATE_LEGEND)
    
    print(f"\n{'='*60}")
    print(f"Plotting complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
