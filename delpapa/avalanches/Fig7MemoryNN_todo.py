####
# Script for the sixth paper figure
# Includes: noise-ABCD-noise
####

from pylab import *

import tables
import os
from tempfile import TemporaryFile

import data_analysis as analysis

# work around to run powerlaw package [Alstott et al. 2014]
import powerlaw as pl

### figure parameters
width  =  7
height = 14
fig_7 = figure(1, figsize=(width, height))

fig_7a = subplot(421)
fig_7b = subplot(422)
fig_7c = subplot(412)
fig_7d = subplot(425)
fig_7e = subplot(426)
fig_7f = subplot(414)

letter_size = 16
letter_size_panel = 16
line_width = 1.5
line_width_fit = 2.0
subplot_letter = (-0.1, 0.9)
subplot_letter1 = (-0.3 , 0.95)

number_of_files = 1

########################################################################
# Avalanches - CM Task                                           #
########################################################################
print 'Avalanche distributions for the Counting Task...'

avalanches_steps = 200000
number_of_files_trans = number_of_files

#for experiment_folder in ['n4_aval', 'n20_aval']:
for experiment_folder in ['MyCMTask']:

    n_trials = number_of_files
    data_all = zeros((n_trials, avalanches_steps))

    for result_file in xrange(n_trials):

        exper = 'result.h5'
        #exper_path =  ''
        exper_path = experiment_folder
        h5 = tables.open_file(os.path.join(exper_path,exper),'r')
        data = h5.root
        data_all[result_file] = np.around(data.activity[0] \
                                        [-avalanches_steps:]*data.c.N_e)
        h5.close()

    Thres_normal = int(data_all.mean()/2.) + 1 # rounding purposes
    Thres_end = Thres_normal
    T_data, S_data = analysis.avalanches(data_all, 'N', '200', \
                                                    Threshold=Thres_end)

    fig_7a = subplot(421)
   #pl.plot_pdf(T_data, linewidth = line_width_fit)

    fig_7b = subplot(422)
    if experiment_folder == 'MyCMTask':
        exp_label = r'$n = 4$'
    elif experiment_folder == 'n20_aval':
        exp_label = r'$n = 20$'
    #pl.plot_pdf(S_data, linewidth = line_width_fit, label = exp_label)

subplot(421)
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$f(T)$', fontsize=letter_size)
fig_7a.spines['right'].set_visible(False)
fig_7a.spines['top'].set_visible(False)
fig_7a.xaxis.set_ticks_position('bottom')
fig_7a.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 300])
ylim([0.0001, 1])
xticks([1, 10, 100], ['$10^0$', '$10^{1}$', '$10^{2}$'])
yticks([1, 0.01, 0.0001], ['$10^0$', '$10^{-2}$', '$10^{-4}$'])

subplot(422)
xscale('log'); yscale('log')
xlabel(r'$S$', fontsize=letter_size)
ylabel(r'$f(S)$', fontsize=letter_size)
fig_7b.spines['right'].set_visible(False)
fig_7b.spines['top'].set_visible(False)
fig_7b.xaxis.set_ticks_position('bottom')
fig_7b.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 3000])
ylim([0.00001, 0.1])
xticks([1, 10, 100, 1000], \
     ['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
yticks([0.1, 0.001, 0.00001],\
            ['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
legend(loc=(0.5, 0.8), prop={'size':letter_size}, frameon=False)

########################################################################

########################################################################
# CM Task - Performance                                          #
########################################################################
print '\nCalculating performance for the Counting Task...'

final_performance_mean = []
final_performance_std = []
final_performance_5p = []
final_performance_95p = []
final_sequence_lengh = [4]


# Add this parameter near the top of your code
tp = 2  # Time shift parameter - adjust as needed

for experiment_folder in ['MyCMTask']:
    partial_performance = np.zeros(number_of_files)

    for file_number in range(number_of_files):
        # Read data files (unchanged)
        exper = 'result.h5'
        exper_path = experiment_folder
        h5 = tables.open_file(os.path.join(exper_path, exper), 'r')
        data = h5.root

        # Create training and test arrays
        train_steps = data.c.steps_readouttrain[0]
        print("Number of training steps: ", train_steps)
        test_steps = data.c.steps_readouttest[0]
        print("Number of testing steps: ", test_steps)

        # Get the letter presentations
        y_train_orig = data.countingletter[0][:train_steps]
        y_test_orig = data.countingletter[0][train_steps:train_steps+test_steps]

        # Apply time shift to get future targets
        # For training data: use later values as targets
        y_train = np.zeros_like(y_train_orig)
        y_train[:-tp] = y_train_orig[tp:]  # Shift by tp steps
        # The last tp steps don't have future values, options:
        y_train[-tp:] = y_train_orig[-tp:]  # Option 1: Use unshifted values
        # Alternatively: y_train[-tp:] = -1  # Mark as special/invalid

        # For test data
        y_test = np.zeros_like(y_test_orig)
        y_test[:-tp] = y_test_orig[tp:]
        y_test[-tp:] = y_test_orig[-tp:]  # Same handling for last tp steps

        print("y_train dtype:", y_train.dtype)
        print("unique values in y_train:", np.unique(y_train))

        # Create one-hot encoded targets
        n_classes = len(np.unique(y_train_orig))  # Based on original data
        print("Number of classes to distinguish: ", n_classes)
        y_read_train = np.zeros((n_classes, len(y_train)))
        y_read_test = np.zeros((n_classes, len(y_test)))

        # Convert to one-hot (now using our time-shifted y values)
        for i, y in enumerate(y_train):
            if y >= 0:  # Only encode valid indices
                y_read_train[int(y), i] = 1
                
        for i, y in enumerate(y_test):
            if y >= 0:
                y_read_test[int(y), i] = 1

        # The rest of the code remains the same
        target = np.argmax(y_read_test, axis=0)

        # Internal state
        X_train = (data.countingactivity[0][:,:train_steps] >= 0) + 0.
        X_test = (data.countingactivity[0][:,train_steps:train_steps+test_steps] >= 0) + 0.

        h5.close()

        # Readout training
        print("Training the readout model...")
        X_train_pinv = np.linalg.pinv(X_train)
        W_trained = np.dot(y_read_train, X_train_pinv)
        print("Readout model trained. Now running inference (prediction)...")
        
        y_predicted = np.dot(W_trained, X_test)
        prediction = np.argmax(y_predicted, axis=0)
        perf_all = (prediction == target).sum()/float(len(y_test))
        print("Prediction done, performance: ", perf_all)

        # reduced performance
        except_first = np.where(np.logical_or(\
                            np.logical_or(y_test == 1, y_test == 2),\
                            np.logical_or(y_test == 4, y_test == 5)))[0]
        y_test_red = y_test[except_first]
        y_pred_red = prediction[except_first]
        perf_red = (y_test_red == y_pred_red).sum()/float(len(y_pred_red))
        print("Reduced performance (only on \"Counting\" symbols)",  perf_red)

        partial_performance[file_number] = perf_red

    final_performance_mean.append(partial_performance.mean())
    final_performance_std.append(partial_performance.std())
    final_performance_5p.append(np.percentile(partial_performance, 16))
    final_performance_95p.append(np.percentile(partial_performance, 84))

subplot(412)
plot(final_sequence_lengh, final_performance_mean, '-b', label=r'$\sigma = 0.05$')
low_err = np.array(final_performance_mean)-np.array(final_performance_5p)
hig_err = np.array(final_performance_95p)-np.array(final_performance_mean)
errorbar(final_sequence_lengh, final_performance_mean, \
                                     yerr=[low_err, hig_err], color='b')
title('CM Task')

ylim([0.4, 1.1])
tight_layout()

########################################################################
fig_7a.annotate('A', xy=subplot_letter1, xycoords='axes fraction', \
                fontsize=letter_size_panel,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_7b.annotate('B', xy=subplot_letter1, xycoords='axes fraction', \
                fontsize=letter_size_panel,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_7c.annotate('C', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_7d.annotate('D', xy=subplot_letter1, xycoords='axes fraction', \
                fontsize=letter_size_panel,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_7e.annotate('E', xy=subplot_letter1, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')
fig_7f.annotate('F', xy=subplot_letter, xycoords='axes fraction', \
                fontsize=letter_size_panel ,  fontweight='bold', \
                horizontalalignment='right', verticalalignment='bottom')

print 'Saving figures...',
result_path = '../../plots/'
result_name_png = 'Fig7.pdf'
savefig(os.path.join(result_path, result_name_png), format = 'pdf')

show()
