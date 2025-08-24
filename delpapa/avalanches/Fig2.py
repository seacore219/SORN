################################################################################
# Script for the 2nd paper figure                                              #
# Modified to use spiking_data.h5 files from SORN simulations                 #
# Python 3 version                                                             #
################################################################################

from pylab import *
import h5py
import os
from tempfile import TemporaryFile
from matplotlib import gridspec
import sys
import glob

# Add parent directory to path to import data_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_analysis as analysis
import powerlaw as pl

def find_sorn_h5_files():
    """Find spiking_data.h5 files from SORN runs"""
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
    print(f"Using h_ip directory: {h_ip_dir}")
    
    # Find all run_seed_* directories
    run_dirs = sorted(glob.glob(os.path.join(h_ip_dir, 'run_seed_*')))
    
    # Get h5 files
    h5_files = []
    for run_dir in run_dirs:
        h5_path = os.path.join(run_dir, 'spiking_data.h5')
        if os.path.exists(h5_path):
            h5_files.append(h5_path)
            print(f"Found: {h5_path}")
    
    return h5_files, h_ip_dir

def load_sorn_data(h5_path, start_idx, end_idx):
    """Load spike data from SORN h5 file and convert to activity"""
    with h5py.File(h5_path, 'r') as f:
        # Get network size
        N_E = f.attrs['N_E']
        
        # Load spike data
        spikes = f['spikes_E'][start_idx:end_idx]
        
        # Convert to activity (number of active neurons per timestep)
        activity = np.sum(spikes, axis=1)
        
    return activity, N_E

### Files to run
number_of_files = 10  # Use 10 runs as shown in your directory

### figure parameters
width  =  8
height = width / 1.718 # fix this to the golden ratio
fig = figure(1, figsize=(width, height))
gs = gridspec.GridSpec(2, 3)
letter_size = 10
letter_size_panel = 12
line_width = 1.5
line_width_fit = 2.0
subplot_letter = (-0.25, 1.15)

### color parameters
c_size = '#B22400'
c_duration = '#006BB2'
c_rawdata = 'gray'
c_expcut = 'k'

########################################################################
# Fig. 2A and 2B SORN size and duration with exponents (A and B)

exp_name = 'N200'
start_idx = int(5e6)  # steps to use: after transient
end_idx = int(6e6)  
stable_steps = end_idx - start_idx

THETA = 'half'

# Find h5 files
print('Finding SORN h5 files...')
h5_files, output_dir = find_sorn_h5_files()

if len(h5_files) < number_of_files:
    print(f"Warning: Only found {len(h5_files)} files, requested {number_of_files}")
    number_of_files = len(h5_files)

# Load the data
print('Loading experiment files...')
data_all = zeros((number_of_files, stable_steps))
N_e_val = None

for i, h5_path in enumerate(h5_files[:number_of_files]):
    print(f"Loading file {i+1}/{number_of_files}: {os.path.basename(os.path.dirname(h5_path))}")
    activity, N_E = load_sorn_data(h5_path, start_idx, end_idx)
    data_all[i] = activity
    if N_e_val is None:
        N_e_val = N_E

# Calculate order parameter and susceptibility
rho = data_all / N_e_val
avg_rho = rho.mean()
print("Average Order Parameter (rho):", avg_rho)
avg_rho_sq = np.mean(rho ** 2)
chi = avg_rho_sq - (avg_rho ** 2)
print("Average Susceptibility: ", chi)

# calculates avalanches
T_data, S_data = analysis.avalanches(data_all, \
                             exp_name[0], exp_name[1:], Threshold=THETA)

########################################
### duration
fig_2a = subplot(gs[0])
print('Fig. 2A...')

# raw data
T_x, inverse = unique(T_data, return_inverse=True)
y_freq = bincount(inverse)
T_y = y_freq / float(y_freq.sum()) # normalization
plot(T_x, T_y, '.', color=c_rawdata, markersize=2, zorder=1)

# power law fit
T_fit = pl.Fit(T_data, xmin=6, xmax=60, discrete=True)
T_alpha = T_fit.alpha
T_sigma = T_fit.sigma
T_xmin = T_fit.xmin
T_fit.power_law.plot_pdf(color=c_duration, \
            label = r'$ \alpha = $' + str(round(T_alpha, 2)), \
            linewidth=line_width_fit, zorder=3)

# exp cutoff calculation
T_fit = pl.Fit(T_data, xmin=6, discrete=True)
T_trunc_alpha = T_fit.truncated_power_law.parameter1
T_trunc_beta = T_fit.truncated_power_law.parameter2
T_fit.truncated_power_law.plot_pdf(color=c_expcut, \
     label = r'$ \alpha^* = $' + str(round(T_trunc_alpha, 2)) + ', ' + \
     r'$ \beta_{\alpha}^* = $' + str(round(T_trunc_beta, 3)), \
     linewidth=line_width, zorder=2)

### axis stuff
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$f(T)$', fontsize=letter_size)

fig_2a.spines['right'].set_visible(False)
fig_2a.spines['top'].set_visible(False)
fig_2a.xaxis.set_ticks_position('bottom')
fig_2a.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 300])
ylim([0.0001, 1])
xticks([1, 10, 100], ['$10^0$', '$10^{1}$', '$10^{2}$'])
yticks([1, 0.01, 0.0001], ['$10^0$', '$10^{-2}$', '$10^{-4}$'])

# legend stuff
legend(loc=(0.0, 0.85), prop={'size':letter_size}, \
                                 title='Fit parameters', frameon=False)
fig_2a.get_legend().get_title().set_fontsize(letter_size)

#########################################
### size stuff
fig_2b = subplot(gs[1])
print('Fig. 2B...')

# raw data
S_x, inverse = unique(S_data, return_inverse=True)
y_freq = bincount(inverse)
S_y = y_freq / float(y_freq.sum())
plot(S_x, S_y, '.', color=c_rawdata, markersize=2, zorder=1)

# power law fit
S_fit = pl.Fit(S_data, xmin=10, xmax=1500, discrete=True)
S_alpha = S_fit.alpha
S_sigma = S_fit.sigma
S_xmin = S_fit.xmin
S_fit.power_law.plot_pdf(color=c_size, \
            label = r'$ \tau = $' + str(round(S_alpha, 2)), \
            linewidth=line_width_fit, zorder=3)

# exp cutoff calculation
S_fit = pl.Fit(S_data, xmin=10, discrete=True)
S_trunc_alpha = S_fit.truncated_power_law.parameter1
S_trunc_beta = S_fit.truncated_power_law.parameter2
S_fit.truncated_power_law.plot_pdf(color=c_expcut,\
          label = r'$ \tau^* = $' + str(round(S_trunc_alpha, 2)) +\
          '; ' + r'$\beta_{\tau}^* = $' + str(round(S_trunc_beta, 3)), \
          linewidth=line_width, zorder=2)

########################################################################
# S1 - comparison to distribution with 1 parameter
print('\n\n Duration (T): \n Exp: ', \
       T_fit.distribution_compare('power_law','exponential', \
                                                normalized_ratio=True),\
       '\n Str. Exp: ', \
       T_fit.distribution_compare('power_law','stretched_exponential',\
                                                  normalized_ratio=True))

print('\n\n Size (S): \n Exp: ', \
       S_fit.distribution_compare('power_law','exponential',\
                                                normalized_ratio=True),\
       '\n Str. Exp: ',\
       S_fit.distribution_compare('power_law','stretched_exponential',\
                                                  normalized_ratio=True))

########################################################################

### axis stuff
xscale('log'); yscale('log')
xlabel(r'$S$', fontsize=letter_size)
ylabel(r'$f(S)$', fontsize=letter_size)

fig_2b.spines['right'].set_visible(False)
fig_2b.spines['top'].set_visible(False)
fig_2b.xaxis.set_ticks_position('bottom')
fig_2b.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

# ticks name
xlim([1, 3000])
ylim([0.00001, 0.1])
xticks([1, 10, 100, 1000], \
     ['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
yticks([0.1, 0.001, 0.00001],\
            ['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])

# legend stuff
legend(loc=(0.0, 0.85), prop={'size':letter_size}, \
                                 title='Fit parameters', frameon=False)
fig_2b.get_legend().get_title().set_fontsize(letter_size)

########################################################################
# Fig. 1C: ratio between exponents

print('Fig 2C...')
fig_2c = subplot(gs[2])

# plot experimental ratio
a_dur_avg, a_area_avg = analysis.area_X_duration(T_data, S_data)
plot(a_dur_avg, a_area_avg / a_area_avg.sum(), '.', color=c_rawdata, \
                     markersize=2, zorder=1, label = r'$\gamma_{\rm data}$')

# plot theoretical ratio (from calculated exponents)
x_range = arange(a_dur_avg.max())
gamma= (T_alpha-1)/(S_alpha-1)
plot(x_range, (a_area_avg/a_area_avg.sum()).min()*x_range**gamma, 'r', \
                label = r'$ \frac{\alpha-1}{\tau-1} $ = ' + str(round(gamma, 2)), \
                linewidth=line_width)

# plot a good ratio for comparison
plot(x_range, (a_area_avg/a_area_avg.sum()).min()*x_range**1.3, '--k', \
                label = r'$\gamma = $' + str(1.3), linewidth=line_width)

# axis stuff
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$ \langle S \rangle (T)$', fontsize=letter_size)

fig_2c.spines['right'].set_visible(False)
fig_2c.spines['top'].set_visible(False)
fig_2c.xaxis.set_ticks_position('bottom')
fig_2c.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 200])
ylim([0.000001, 0.1])
xticks([1, 10, 100], \
       ['$10^0$', '$10^{1}$', '$10^{2}$'])
yticks([0.1, 0.001, 0.00001], \
       ['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])

legend(loc=(0.0, 0.65), prop={'size':letter_size}, \
                                 title='Exponent ratio', \
                                 frameon=False, numpoints=1)
fig_2c.get_legend().get_title().set_fontsize(letter_size)

del a_dur_avg, a_area_avg, T_data, S_data # to save memory
########################################################################

########################################################################
# Fig. 1D and 1E - Using same data for different network sizes
# Note: This would require running SORN with different network sizes
# For now, we'll skip these panels or use the same data

print('Fig 2D and 2E skipped - would require different network sizes')

# Create empty subplots for now
fig_2d = subplot(gs[3])
fig_2d.text(0.5, 0.5, 'Panel D\n(Different network sizes\nrequired)', 
            ha='center', va='center', fontsize=letter_size)
fig_2d.set_xlim([0, 1])
fig_2d.set_ylim([0, 1])

fig_2e = subplot(gs[4])
fig_2e.text(0.5, 0.5, 'Panel E\n(Different network sizes\nrequired)', 
            ha='center', va='center', fontsize=letter_size)
fig_2e.set_xlim([0, 1])
fig_2e.set_ylim([0, 1])

########################################################################
# Add panel labels
fig_2a.annotate('A', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_2b.annotate('B', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_2c.annotate('C', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_2d.annotate('D', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_2e.annotate('E', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')

fig.subplots_adjust(wspace=.5)
fig.subplots_adjust(hspace=.65)

# saving figures
print('Saving figures...')
# Save to the output directory where the h5 files are
result_name = 'Fig2_SORN_analysis.png'
savefig(os.path.join(output_dir, result_name))
print('Saved to:', os.path.join(output_dir, result_name))
print('done\n\n')