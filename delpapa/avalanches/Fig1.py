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

import matplotlib.ticker as mtick
import matplotlib.patches as patches
from matplotlib import gridspec

def find_first_sorn_h5():
    """Find the first spiking_data.h5 file from SORN runs"""
    # Start from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to common directory where SORN results are stored
    common_dir = os.path.join(script_dir, '..', '..', 'common')
    common_dir = os.path.abspath(common_dir)
    
    print(f"Looking for SORN results in: {common_dir}")
    
    # Find the most recent h_ip_sweep directory
    sweep_dirs = glob.glob(os.path.join(common_dir, 'h_ip_sweep_eigenvalues_*'))
    if not sweep_dirs:
        raise ValueError(f"No h_ip_sweep_eigenvalues directories found in {common_dir}")
    
    # Use the most recent one
    latest_sweep = sorted(sweep_dirs)[-1]
    print(f"Using sweep directory: {latest_sweep}")
    
    # Navigate to the h_ip_0.1 runs
    h_ip_dir = os.path.join(latest_sweep, 'sweep_eigenvalues_*', 'h_ip_0.1')
    h_ip_dirs = glob.glob(h_ip_dir)
    
    if not h_ip_dirs:
        raise ValueError(f"No h_ip_0.1 directory found in {latest_sweep}")
    
    h_ip_dir = h_ip_dirs[0]
    
    # Find first run_seed directory
    run_dirs = sorted(glob.glob(os.path.join(h_ip_dir, 'run_seed_*')))
    if not run_dirs:
        raise ValueError(f"No run_seed directories found in {h_ip_dir}")
    
    # Use the first run
    first_run = run_dirs[0]
    h5_path = os.path.join(first_run, 'spiking_data.h5')
    
    if not os.path.exists(h5_path):
        raise ValueError(f"No spiking_data.h5 found in {first_run}")
    
    print(f"Using h5 file: {h5_path}")
    return h5_path, h_ip_dir

# Find the h5 file to use
h5_path, output_dir = find_first_sorn_h5()

# Load data from SORN h5 file
print('Loading SORN data...')
with h5py.File(h5_path, 'r') as f:
    # Get network parameters
    N_E = f.attrs['N_E']
    n_timesteps = f.attrs['n_timesteps']
    
    # Load connection fraction data
    connec_frac = f['connection_fraction'][:]
    record_interval = f.attrs.get('record_interval', 1000)
    
    # Convert to actual timesteps
    connec_frac_timesteps = np.arange(len(connec_frac)) * record_interval
    
    # Interpolate to all timesteps for plotting
    from scipy.interpolate import interp1d
    if len(connec_frac) > 1:
        f_interp = interp1d(connec_frac_timesteps, connec_frac, 
                           kind='linear', fill_value='extrapolate')
        connec_frac_full = f_interp(np.arange(n_timesteps))
    else:
        connec_frac_full = np.ones(n_timesteps) * connec_frac[0]
    
    # Load activity data for the last part
    plot_last_steps = 150
    start_idx = n_timesteps - plot_last_steps - 100
    end_idx = n_timesteps - 100
    
    # Calculate activity from spikes
    spikes = f['spikes_E'][start_idx:end_idx]
    activity = np.sum(spikes, axis=1)  # Number of active neurons per timestep

# Set threshold for avalanche detection
Theta = 10

### figure parameters
width  =  10
height = 3
fig = figure(1, figsize=(width, height))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.7])
letter_size = 13
letter_size_panel = 15  # Added this line
line_width = 1.0

c_size = '#B22400'
c_duration = '#006BB2'
c_stable = '#2E4172'
c_notstable = '#7887AB'

################################################################################
# Fig. 1A: CONNECTION FRACTION
fig_1a = subplot(gs[0])

# Determine phases based on time (you may need to adjust these based on your data)
phase1_end = int(2e6)  # First 2M steps for growth/decay
phase2_end = int(4e6)  # Next 2M steps for stable

# Plot different phases
time_axis = np.arange(len(connec_frac_full))
mask1 = time_axis < phase1_end
mask2 = (time_axis >= phase1_end) & (time_axis < phase2_end)

plot(time_axis[mask1], connec_frac_full[mask1]*100, c_notstable, linewidth=line_width)
plot(time_axis[mask2], connec_frac_full[mask2]*100, c_stable, linewidth=line_width)

### annotate stuff
# Find appropriate positions for annotations based on actual data
early_idx = int(2e5)
mid_idx = int(8e5)
late_idx = int(2.6e6)

if early_idx < len(connec_frac_full):
    text(2e5, connec_frac_full[early_idx]*100 - 2, r'decay', 
         fontsize=letter_size, color=c_notstable)
if mid_idx < len(connec_frac_full):
    text(8e5, connec_frac_full[mid_idx]*100 + 1, r'growth', 
         fontsize=letter_size, color=c_notstable)
if late_idx < len(connec_frac_full):
    text(2.6e6, connec_frac_full[late_idx]*100 + 0.5, r'stable', 
         fontsize=letter_size, color=c_stable)

# axis stuff
xlim([0, min(4e6, n_timesteps)])
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
# Set ylim based on actual activity range
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