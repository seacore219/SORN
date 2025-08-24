################################################################################
# Script for the 2nd paper figure                                              #
# A: N200 duration distribution                                                #
# B: N200 size distribution                                                    #
# C: Average size X duration                                                   #
# D: N50~800 duration distribution                                             #
# E: N50~800 size distribution                                                 #
# F: Power-law range X network size                                            #
#                                                                              #
# Script also used for comparison between power-law fit and other              #
# 1-parameter distributions for Table S1 (from Fig2B)                          #
################################################################################

from pylab import *
import tables
import os
from tempfile import TemporaryFile
from matplotlib import gridspec

import data_analysis as analysis
import powerlaw as pl

def test_hdf5_file(filepath):
    """Test if HDF5 file is readable"""
    try:
        h5 = tables.open_file(filepath, 'r')
        h5.close()
        return True
    except:
        return False

def get_backup_folders():
    """Get list of timestamped sanity_check backup folders with valid HDF5 files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    backup_root = os.path.join(root_dir, 'backup', 'test_single', 'sanity_check')  # added 'sanity_check'

    folders = sorted([d for d in os.listdir(backup_root) if d.startswith('202') and 
                      os.path.isdir(os.path.join(backup_root, d))])
    
    # Test and filter for valid folders
    valid_folders = []
    print "Testing backup folders..."
    for folder in folders:
        result_path = os.path.join(backup_root, folder, 'common', 'result.h5')
        if os.path.exists(result_path) and test_hdf5_file(result_path):
            print "Valid backup found:", folder
            valid_folders.append(folder)
    
    print "Found %d valid backup folders" % len(valid_folders)
    return valid_folders

def get_result_path(base_num):
    """Get path to result.h5 from a specific sanity_check backup folder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    backup_folders = get_backup_folders()
    if base_num > len(backup_folders):
        raise ValueError("Not enough backup folders available")
    folder = backup_folders[-base_num]  # Use newest folders first
    path = os.path.join(root_dir, 'backup', 'test_single', 'sanity_check', folder, 'common', 'result.h5')
    print "Using backup from:", folder
    return path



### Files to run
number_of_files = 25

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
start_idx = int(5e6)  # Convert to integer # steps to use: after transient (2e6)
end_idx = int(6e6)  
stable_steps = end_idx - start_idx  # Total number of steps in windowv

THETA = 'half'

#count files
print 'Loading experiment files...'
exper_path = ''

#load the data in 'data_all'
data_all = zeros((number_of_files, stable_steps))
for result_file in range(number_of_files):
        result_path = get_result_path(result_file + 1)
        print "Loading:", result_path
        h5 = tables.open_file(result_path, 'r')
        data = h5.root

        if end_idx is None:
              selected_activity = data.activity[0][start_idx:]
        else:
              selected_activity = data.activity[0][start_idx:end_idx]

        data_all[result_file] = around(selected_activity * data.c.N_e)
        N_e_val = data.c.N_e[...]
        h5.close()

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
print 'Fig. 2A...'

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
# pl.plot_pdf(T_data, color=c_duration, linewidth=line_width_SORN)
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
print 'Fig. 2B...'

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
# pl.plot_pdf(S_data, linewidth=line_width)
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
# S1 - comparison to distribution with 1 parameter:
#      exponential and streched_exponential
#      The distributions with 2 parameters have, of course,
#      a better fit: lognormal, truncated_exponential

print '\n\n Duration (T): \n Exp: ', \
       T_fit.distribution_compare('power_law','exponential', \
                                                normalized_ratio=True),\
       '\n Str. Exp: ', \
       T_fit.distribution_compare('power_law','stretched_exponential',\
                                                  normalized_ratio=True)

print '\n\n Size (S): \n Exp: ', \
       S_fit.distribution_compare('power_law','exponential',\
                                                normalized_ratio=True),\
       '\n Str. Exp: ',\
       S_fit.distribution_compare('power_law','stretched_exponential',\
                                                  normalized_ratio=True)

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

print 'Fig 2C...'
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
# Fig. 1D and 1E SORN size and duration with exponents (A and B)
variable = 'N'; values = ['50', '100', '400', '800', '200']

print 'Fig 2D...'
fig_2d = subplot(gs[3])
for v in values:
    data_all = np.zeros((number_of_files, stable_steps))
    for result_file in range(number_of_files):
        result_path = get_result_path(result_file + 1)
        h5 = tables.open_file(result_path, 'r')
        data = h5.root
        data_all[result_file] = \
              np.around(data.activity[0][-stable_steps:]*data.c.N_e)
        h5.close()

    # calculate duration and area of avalanches
    a_dur, a_area = analysis.avalanches(data_all, variable, v, \
                                                       Threshold=THETA)
    pl.plot_pdf(a_dur, linewidth=line_width)

x_range = arange(a_dur.max())
if T_alpha != 0:  # Add check for zero
    plot(x_range, x_range**(-T_alpha), '--', color=c_duration, linewidth=line_width_fit)
# axis stuff
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$f(T)$', fontsize=letter_size)

fig_2d.spines['right'].set_visible(False)
fig_2d.spines['top'].set_visible(False)
fig_2d.xaxis.set_ticks_position('bottom')
fig_2d.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 1000])
ylim([0.000001, 1])
xticks([1, 10, 100, 1000, 10000], \
       ['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'])
yticks([1, 0.01, 0.0001, 0.000001], \
       ['$10^{0}$', '$10^{-2}$', '$10^{-4}$', '$10^{-6}$'])


############################3
print 'Fig 2E...'
fig_2e = subplot(gs[4])
for v in values:
    data_all = np.zeros((number_of_files, stable_steps))
    for result_file in range(number_of_files):
        result_path = get_result_path(result_file + 1)  # Use same function as before
        h5 = tables.open_file(result_path, 'r')
        data = h5.root
        data_all[result_file] = \
                  np.around(data.activity[0][-stable_steps:]*data.c.N_e)
h5.close()

    # calculate duration and area of avalanches
a_dur, a_area = analysis.avalanches(data_all, variable, v, \
                                                       Threshold=THETA)
pl.plot_pdf(a_area, label= v, linewidth=line_width)

x_range = arange(a_area.max())
if S_alpha != 0:  # Add check for zero
    plot(x_range, x_range**(-S_alpha), '--', color=c_size, linewidth=line_width_fit)

# axis stuff
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$f(T)$', fontsize=letter_size)

fig_2e.spines['right'].set_visible(False)
fig_2e.spines['top'].set_visible(False)
fig_2e.xaxis.set_ticks_position('bottom')
fig_2e.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 100000])
ylim([0.000001, 1])

xticks([1, 100, 10000], \
       ['$10^0$', '$10^{2}$', '$10^{4}$'])
yticks([1, 0.001, 0.000001], \
       ['$10^{0}$', '$10^{-3}$', '$10^{-6}$'])

xlabel(r'$S$', fontsize=letter_size)
ylabel(r'$f(S)$', fontsize=letter_size)


legend(loc=(0.45, 0.45), prop={'size':letter_size}, \
                                 title='Network size', frameon=False)
fig_2e.get_legend().get_title().set_fontsize(letter_size)

del a_dur, a_area # just to save memory
########################################################################

########################################################################
# Fig. 2F - Window of power-laws X network size
# print 'Fig 2F...'
# fig_2f = subplot(gs[5])
# net_sizes = np.array([200])

# duration_window = np.array([np.log10(10)-np.log10(2), \
#                             np.log10(30)-np.log10(4), \
#                             np.log10(60)-np.log10(6), \
#                             np.log10(100)-np.log10(8),\
#                             np.log10(150)-np.log10(10)])

# size_window = np.array([np.log10(100)-np.log10(3), \
#                             np.log10(200)-np.log10(4), \
#                             np.log10(1500)-np.log10(10), \
#                             np.log10(3000)-np.log10(30),\
#                             np.log10(7000)-np.log10(70)])

# plot(net_sizes, size_window, '-o', color=c_size, \
#                        linewidth=line_width, label='size')
# plot(net_sizes, duration_window, '-o', color=c_duration, \
#                        linewidth=line_width, label='duration')

# # axis stuff
# xlabel(r'Network Size', fontsize=letter_size)
# ylabel(r'Power-law scale range', fontsize=letter_size)
# tick_params(axis='both', which='major', labelsize=letter_size)

# fig_2f.spines['right'].set_visible(False)
# fig_2f.spines['top'].set_visible(False)
# fig_2f.yaxis.set_ticks_position('left')
# fig_2f.xaxis.set_ticks_position('bottom')

# # ticks name
# xscale('log')
# xlim([20, 2000])
# ylim([0, 3])


# xticks([100, 1000], \
#        ['$10^{2}$', '$10^{3}$'])
# yticks([0, 1, 2, 3], \
#        ['$0$', '$1$', '$2$', '$3$'])

# legend(loc=(0.15, 0.8), prop={'size':letter_size}, \
#                          frameon=False, numpoints=1)
########################################################################
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
# fig_2f.annotate('F', xy=subplot_letter, xycoords='axes fraction', \
#                 fontsize=letter_size_panel ,  fontweight='bold', \
#                 horizontalalignment='right', verticalalignment='bottom')

fig.subplots_adjust(wspace=.5)
fig.subplots_adjust(hspace=.65)

# saving figures
print 'Saving figures...'
# Create plots directory if it doesn't exist
plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Save figure
result_name = 'Fig2.png'  # Changed to PNG format
savefig(os.path.join(plots_dir, result_name))
# print 'Saved to:', os.path.join(plots_dir, result_name)
print 'done\n\n'