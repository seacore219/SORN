################################################################################
# Script for the 1st paper figure                                              #
# Modified to use spiking_data.h5 files from SORN simulations                 #
# Python 3 version                                                             #
# A: 3 Network phases;                                                         #
# B: activity thresholding example                                             #
################################################################################

from pylab import *
import h5py
import os
from tempfile import TemporaryFile
import glob
import tables  # Add this import for .h5 files created by PyTables

import matplotlib.ticker as mtick
import matplotlib.patches as patches
from matplotlib import gridspec

# Configuration - set your file path here
SORN_H5_PATH = r'C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\test_single\peptember2\2025-09-10 23-51-01\common\result.h5'

def find_first_sorn_h5():
    """Use configured SORN h5 file path"""
    if not os.path.exists(SORN_H5_PATH):
        raise ValueError(f"File not found: {SORN_H5_PATH}")
    
    print(f"Using h5 file: {SORN_H5_PATH}")
    return SORN_H5_PATH, os.path.dirname(SORN_H5_PATH)

# Find the h5 file to use
h5_path, output_dir = find_first_sorn_h5()

# Load data from SORN h5 file
print('Loading SORN data...')

# First, let's check if this is an HDF5 or PyTables file
try:
    # Try h5py first
    with h5py.File(h5_path, 'r') as f:
        print("File opened with h5py")
        print("File structure:")
        print("Attributes:", list(f.attrs.keys()))
        print("Datasets/Groups:", list(f.keys()))
        use_h5py = True
except:
    # If h5py fails, try PyTables
    print("h5py failed, trying PyTables...")
    use_h5py = False

if use_h5py:
    with h5py.File(h5_path, 'r') as f:
        # Get network parameters
        # Your file seems to be PyTables format opened with h5py
        if 'c' in f:
            # Access PyTables-style data
            c_group = f['c']
            N_E = c_group['N_e'][0] if 'N_e' in c_group else 1000
            # Get timesteps from stats
            if 'stats' in c_group and 'only_last_spikes' in c_group['stats']:
                n_timesteps = c_group['stats']['only_last_spikes'][0]
            else:
                # Get from spike data shape
                n_timesteps = f['Spikes'].shape[2]
        else:
            N_E = 1000
            n_timesteps = f['Spikes'].shape[2]
        
        print(f"N_E: {N_E}, n_timesteps: {n_timesteps}")
        
        # Load connection fraction data
        if 'connection_fraction' in f:
            connec_frac_data = f['connection_fraction'][:]
            print(f"Connection fraction shape: {connec_frac_data.shape}")
            print(f"Connection fraction first few values: {connec_frac_data[:5]}")
            
            # Handle different possible formats
            if connec_frac_data.ndim == 1:
                connec_frac = connec_frac_data
            else:
                connec_frac = connec_frac_data.flatten()
        else:
            print("Warning: connection_fraction not found, using dummy data")
            connec_frac = np.ones(100) * 0.1
            
        record_interval = n_timesteps // len(connec_frac) if len(connec_frac) > 1 else n_timesteps
        
        # Load spike data
        plot_last_steps = 150
        
        # Get actual dimensions of spike data
        spike_shape = f['Spikes'].shape
        print(f"Spikes shape: {spike_shape}")
        
        # Adjust indices based on actual data
        actual_timesteps = spike_shape[2]
        start_idx = max(0, actual_timesteps - plot_last_steps - 100)
        end_idx = actual_timesteps - 100
        
        # Load spikes - format is [batch, neurons, time]
        spikes = f['Spikes'][0, :N_E, start_idx:end_idx].T
        activity = np.sum(spikes, axis=1)

# Convert to actual timesteps
connec_frac_timesteps = np.arange(len(connec_frac)) * record_interval

# Interpolate to all timesteps for plotting
from scipy.interpolate import interp1d

# Determine the target length for the plot (first 4M steps or available data)
plot_length = min(n_timesteps, int(4e6))

if len(connec_frac) > 1 and len(connec_frac_timesteps) > 1:
    # Ensure we don't extrapolate beyond available data
    max_time = min(connec_frac_timesteps[-1], plot_length)
    time_points = np.arange(0, max_time, 1000)  # Sample every 1000 steps for efficiency
    
    f_interp = interp1d(connec_frac_timesteps[connec_frac_timesteps <= max_time], 
                       connec_frac[:len(connec_frac_timesteps[connec_frac_timesteps <= max_time])], 
                       kind='linear', fill_value='extrapolate', bounds_error=False)
    connec_frac_full = f_interp(time_points)
    time_axis_plot = time_points
else:
    # If only one value, create constant array
    time_axis_plot = np.arange(0, plot_length, 1000)
    connec_frac_full = np.ones_like(time_axis_plot) * connec_frac.flat[0]

print(f"connec_frac_full shape: {connec_frac_full.shape}")
print(f"time_axis_plot shape: {time_axis_plot.shape}")

# Set threshold for avalanche detection
Theta = 10

### figure parameters
width  =  10
height = 3
fig = figure(1, figsize=(width, height))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.7])
letter_size = 13
letter_size_panel = 15
line_width = 1.0

c_size = '#B22400'
c_duration = '#006BB2'
c_stable = '#2E4172'
c_notstable = '#7887AB'

################################################################################
# Fig. 1A: CONNECTION FRACTION
fig_1a = subplot(gs[0])

# Determine phases based on time
phase1_end = int(2e6)
phase2_end = int(4e6)

# Plot different phases using the corrected time axis
mask1 = time_axis_plot < phase1_end
mask2 = (time_axis_plot >= phase1_end) & (time_axis_plot < phase2_end)

plot(time_axis_plot[mask1], connec_frac_full[mask1]*100, c_notstable, linewidth=line_width)
plot(time_axis_plot[mask2], connec_frac_full[mask2]*100, c_stable, linewidth=line_width)

# Find closest indices for annotations
early_time = 2e5
mid_time = 8e5
late_time = 2.6e6

early_idx = np.argmin(np.abs(time_axis_plot - early_time))
mid_idx = np.argmin(np.abs(time_axis_plot - mid_time))
late_idx = np.argmin(np.abs(time_axis_plot - late_time))

if early_idx < len(connec_frac_full):
    text(early_time, connec_frac_full[early_idx]*100 - 2, r'decay', 
         fontsize=letter_size, color=c_notstable)
if mid_idx < len(connec_frac_full):
    text(mid_time, connec_frac_full[mid_idx]*100 + 1, r'growth', 
         fontsize=letter_size, color=c_notstable)
if late_idx < len(connec_frac_full):
    text(late_time, connec_frac_full[late_idx]*100 + 0.5, r'stable', 
         fontsize=letter_size, color=c_stable)

# axis stuff
xlim([0, min(4e6, time_axis_plot[-1])])
ylim([0, 20])

fig_1a.spines['right'].set_visible(False)
fig_1a.spines['top'].set_visible(False)
fig_1a.spines['left'].set_visible(False)
fig_1a.spines['bottom'].set_visible(False)
fig_1a.yaxis.set_ticks_position('left')
fig_1a.xaxis.set_ticks_position('bottom')
fig_1a.tick_params(axis='both', which='both',length=0)
fig_1a.grid()

xticks(arange(0, 4.1e6, 1e6), ['0', '1', '2', '3', '4'])
yticks([5, 10, 15], ['5%', '10%', '15%'])

xlabel(r'$10^6$ time steps', fontsize=letter_size)
ylabel(r'Active Connections', fontsize=letter_size)
tick_params(axis='both', which='major', labelsize=letter_size)

################################################################################
# Fig. 1B: AVALANCHE DEFINITION
fig_1b = subplot(gs[1])

### plot stuff
boundary = Theta*np.ones(plot_last_steps)
plot(activity, 'k', label='network activity', linewidth=line_width)
plot(boundary, '--k', label='$\\theta$', linewidth=line_width)
fill_between(np.arange(plot_last_steps), activity, boundary, \
      alpha = 0.5, where=activity>=boundary, facecolor=c_size, interpolate=True)

### annotate stuff
text(20, 45, r'avalanches', fontsize=letter_size, color='k')
text(70, 4, r'duration', fontsize=letter_size, color=c_duration)
text(80, 12, r'size', fontsize=letter_size, color=c_size)
text(62, -4, r'100 time steps', fontsize=letter_size, color='k')

plot((58, 122), (8, 8) , c_duration, linewidth=2.0)
plot((50, 150), (0, 0) , 'k', linewidth=2.5)

### arrow stuff
arrow1 = patches.FancyArrowPatch((35,44), (12,29), arrowstyle='-|>', \
                                                fc='k', lw=1, mutation_scale=10)
fig_1b.add_patch(arrow1)
arrow2 = patches.FancyArrowPatch((55,44), (40,35), arrowstyle='-|>', \
                                                fc='k', lw=1, mutation_scale=10)
fig_1b.add_patch(arrow2)
arrow3 = patches.FancyArrowPatch((65,44), (75,38), arrowstyle='-|>', \
                                                fc='k', lw=1, mutation_scale=10)
fig_1b.add_patch(arrow3)

arrow4 = patches.FancyArrowPatch((60,44), (54.5,15), arrowstyle='-|>', \
                                                fc='k', lw=1, mutation_scale=10)
fig_1b.add_patch(arrow4)

### axis stuff
xlim([0, plot_last_steps])
ylim([0, max(50, activity.max() * 1.2)])

fig_1b.spines['right'].set_visible(False)
fig_1b.spines['top'].set_visible(False)
fig_1b.spines['bottom'].set_visible(False)
fig_1b.yaxis.set_ticks_position('left')
fig_1b.axes.get_xaxis().set_visible(False)

yticks([0, Theta, 20, 40], ['0', r'$\theta$', '20', '40'])

ylabel(r'$a(t)$' + r' [# neurons]', fontsize=letter_size)
tick_params(axis='both', which='major', labelsize=letter_size)

################################################################################
### Panel label stuff
fig.text(0.01, 0.9, "A", weight="bold", fontsize=letter_size_panel,
         horizontalalignment='left', verticalalignment='center')
fig.text(0.55, 0.9, "B", weight="bold", fontsize=letter_size_panel,
         horizontalalignment='left', verticalalignment='center')
         
gcf().subplots_adjust(bottom=0.17)
fig.subplots_adjust(wspace=.4)

# Save figure
print('Saving figure...', end=' ')
result_name = 'Fig1_SORN_analysis.pdf'
savefig(os.path.join(output_dir, result_name), format='pdf')
print('Saved to:', os.path.join(output_dir, result_name))
print('done\n\n')