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

import numpy as np

class ReadoutNetwork(object):
    def __init__(self, input_size, hidden_size=128):
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size/2) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros(hidden_size/2)
        self.W3 = np.random.randn(hidden_size/2, 6) * np.sqrt(2.0/(hidden_size/2))
        self.b3 = np.zeros(6)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.d1 = np.random.binomial(1, 0.8, self.a1.shape) # dropout
        self.a1 *= self.d1
        
        # Second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.d2 = np.random.binomial(1, 0.8, self.a2.shape) # dropout
        self.a2 *= self.d2
        
        # Output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        
        return self.a3
    
    def backward(self, X, y, learning_rate=0.001):
        batch_size = X.shape[0]
        
        # Output layer
        delta3 = self.a3 - y
        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0)
        
        # Second hidden layer
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(self.z2) * self.d2
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        
        # First hidden layer
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1) * self.d1
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        
        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def train_readout_network(X_train, y_read_train, X_test, y_read_test, 
                         batch_size=32, epochs=100, learning_rate=0.001):
    model = ReadoutNetwork(input_size=X_train.shape[0])
    
    # Transpose data to match expected shape
    X_train = X_train.T
    y_train = y_read_train.T
    X_test = X_test.T
    
    n_batches = int(np.ceil(float(X_train.shape[0]) / batch_size))
    
    # Training loop
    for epoch in xrange(epochs):
        # Shuffle training data
        shuffle_idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        total_loss = 0
        for batch_idx in xrange(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, X_train.shape[0])
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Forward and backward passes
            output = model.forward(batch_X)
            model.backward(batch_X, batch_y, learning_rate)
            
            # Calculate loss (cross-entropy)
            loss = -np.mean(np.sum(batch_y * np.log(np.clip(output, 1e-10, 1.0)), axis=1))
            total_loss += loss
            
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, total_loss/n_batches))
    
    # Get predictions for test set
    test_output = model.forward(X_test)
    predictions = np.argmax(test_output, axis=1)
    
    return model, predictions


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
# Avalanches - Counting Task                                           #
########################################################################
print 'Avalanche distributions for the Counting Task...'

avalanches_steps = 200000
number_of_files_trans = number_of_files

#for experiment_folder in ['n4_aval', 'n20_aval']:
for experiment_folder in ['MySmallCountingTask']:

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
    if experiment_folder == 'MySmallCountingTask':
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
# Counting Task - Performance                                          #
########################################################################
print '\nCalculating performance for the Counting Task...'

final_performance_mean = []
final_performance_std = []
final_performance_5p = []
final_performance_95p = []
final_sequence_lengh = [4]
for experiment_folder in ['MySmallCountingTask']:

    partial_performance = np.zeros(number_of_files)

    for file_number in range(number_of_files):

        # read data files
        exper = 'result.h5'
        exper_path = experiment_folder
        h5 = tables.open_file(os.path.join(exper_path,exper),'r')
        data = h5.root
        
        # create training and test arrays
        train_steps = int(data.c.steps_readouttrain[0])
        test_steps = int(data.c.steps_readouttest[0])
        
        # Get the current letters (input) and next letters (target)
        current_letters_train = data.countingletter[0][:train_steps-1]  # all but last
        next_letters_train = data.countingletter[0][1:train_steps]      # all but first
        
        current_letters_test = data.countingletter[0][train_steps:train_steps+test_steps-1]
        next_letters_test = data.countingletter[0][train_steps+1:train_steps+test_steps]
        
        # Create one-hot vectors for input (X) and output (y)
        # Assuming 6 possible letters (0-5)
        n_letters = 6
        
        # One-hot encode input letters
        X_train = np.zeros((len(current_letters_train), n_letters))
        for i, letter in enumerate(current_letters_train):
            X_train[i, int(letter)] = 1
            
        X_test = np.zeros((len(current_letters_test), n_letters))
        for i, letter in enumerate(current_letters_test):
            X_test[i, int(letter)] = 1
        
        # One-hot encode target letters
        y_train = np.zeros((len(next_letters_train), n_letters))
        for i, letter in enumerate(next_letters_train):
            y_train[i, int(letter)] = 1
            
        y_test = np.zeros((len(next_letters_test), n_letters))
        for i, letter in enumerate(next_letters_test):
            y_test[i, int(letter)] = 1
        
        # Transpose to match the expected format (features x samples)
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T
        
        # Get target for performance calculation
        target = np.argmax(y_test, axis=0)
        
        h5.close()

        # Readout training
        # Readout training
        print("Training the readout model...")
        # Prepare next-letter prediction data
        X_train_current = X_train[:, :-1]  # all but last letter
        y_train_next = y_train[:, 1:]      # all but first letter

        # Network prediction with trained weights
        X_test_current = X_test[:, :-1]  # remove last letter as we can't verify its prediction
        y_test_next = y_test[:, 1:]        # Remove first timestep

        model, prediction = train_readout_network(X_train_current, y_train_next, 
                                        X_test_current, y_test_next)
        
        # Convert one-hot test targets back to indices
        target = np.argmax(y_test_next, axis=0)  # actual next letters
        perf_all = (prediction == target).sum()/float(len(target))
        print("Prediction done, performance: ", perf_all)

        # reduced performance
        # Find positions where the target is 1,2,4,or 5
        except_first = np.where(np.logical_or(
                            np.logical_or(target == 1, target == 2),
                            np.logical_or(target == 4, target == 5)))[0]

        # Get the reduced set of predictions and targets
        y_test_red = target[except_first]
        y_pred_red = prediction[except_first]

        # Calculate reduced performance
        perf_red = (y_test_red == y_pred_red).sum()/float(len(y_pred_red))
        print("Reduced performance (only on \"Counting\" symbols)", perf_red)

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
title('Counting Task')


########################################################################
# Random Task - Avalanches                                             #
########################################################################
print 'Calculating avalanche distributions for the Random Task...'

avalanches_steps = 30000
number_of_files_trans = number_of_files

for experiment_folder in ['MySmallRandomTask']:

    n_trials = number_of_files
    data_all = zeros((n_trials, avalanches_steps))

    for result_file in xrange(n_trials):

        exper = 'result.h5'
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

    fig_7d = subplot(425)
    if experiment_folder == 'MySmallRandomTask':
        exp_label = r'$L=10$'
        colorl = 'b'
    if experiment_folder == 'L20':
        exp_label = r'$L=20$'
        colorl = 'g'
    elif experiment_folder == 'L100':
        exp_label = r'$L=100$'
        colorl = 'c'


    pl.plot_pdf(T_data, color=colorl, linewidth = line_width_fit)

    fig_7e = subplot(426)
    pl.plot_pdf(S_data, color=colorl, linewidth = line_width_fit, label = exp_label)

subplot(425)
xscale('log'); yscale('log')
xlabel(r'$T$', fontsize=letter_size)
ylabel(r'$f(T)$', fontsize=letter_size)
fig_7d.spines['right'].set_visible(False)
fig_7d.spines['top'].set_visible(False)
fig_7d.xaxis.set_ticks_position('bottom')
fig_7d.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 300])
ylim([0.0001, 1])
xticks([1, 10, 100], ['$10^0$', '$10^{1}$', '$10^{2}$'])
yticks([1, 0.01, 0.0001], ['$10^0$', '$10^{-2}$', '$10^{-4}$'])


subplot(426)
xscale('log'); yscale('log')
xlabel(r'$S$', fontsize=letter_size)
ylabel(r'$f(S)$', fontsize=letter_size)
fig_7e.spines['right'].set_visible(False)
fig_7e.spines['top'].set_visible(False)
fig_7e.xaxis.set_ticks_position('bottom')
fig_7e.yaxis.set_ticks_position('left')
tick_params(labelsize=letter_size)

xlim([1, 3000])
ylim([0.00001, 0.1])
xticks([1, 10, 100, 1000], \
     ['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
yticks([0.1, 0.001, 0.00001],\
            ['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
legend(loc=(0.5, 0.75), prop={'size':letter_size}, frameon=False)

########################################################################


########################################################################
# Random Task - Performance                                            #
########################################################################
print '\nCalculating performance for the Random Task...'

final_performance_mean = []
final_performance_std = []
final_performance_5p = []
final_performance_95p = []
final_sequence_lengh = [6]

for experiment_folder in ['MySmallRandomTask']:


    partial_performance = np.zeros(number_of_files)

    for file_number in range(number_of_files):

        # read data files
        exper = 'result.h5'
        exper_path = experiment_folder
        h5 = tables.open_file(os.path.join(exper_path,exper),'r')
        data = h5.root

        # create training and test arrays
        train_steps = data.c.steps_readouttrain[0]
        print("Number of training steps for the RandomTask", train_steps)
        test_steps = data.c.steps_readouttest[0]
        print("Number of testing steps for the RandomTask", test_steps)

        # the letter to be presented at the time step
        y_train = data.countingletter[0][:train_steps]
        y_test = data.countingletter[0]\
                                    [train_steps:train_steps+test_steps]

        # # divide readout into n different units
        # y_read_train = np.zeros((10, len(y_train)))
        # y_read_test = np.zeros((10, len(y_test)))
        # for i, y in enumerate(y_train):
        #     y_read_train[y, i] = 1
        # for i, y in enumerate(y_test):
        #     y_read_test[y, i] = 1
        
        print("y_train dtype:", y_train.dtype)
        print("unique values in y_train:", np.unique(y_train))

        # Then modify the code to ensure integer indices:
        n_classes = len(np.unique(y_train))  # This will be 10 based on your output
        print("Number of classes in RandomTask: ", n_classes)
        y_read_train = np.zeros((n_classes, len(y_train)))
        y_read_test = np.zeros((n_classes, len(y_test)))

        # Convert to integers when indexing
        for i, y in enumerate(y_train):
            y_read_train[int(y), i] = 1
        for i, y in enumerate(y_test):
            y_read_test[int(y), i] = 1

        target = np.argmax(y_read_test, axis=0) # target for the data

        # internal state before letter presentation
        X_train = (data.countingactivity[0][:,:train_steps] >= 0) + 0.
        X_test = (data.countingactivity[0]\
                       [:,train_steps:train_steps+test_steps] >= 0) + 0.

        h5.close()

        # Readout training
        print("Training the readout model for the RandomTask...")
        X_train_pinv = np.linalg.pinv(X_train) # MP pseudo-inverse
        W_trained = np.dot(y_read_train, X_train_pinv) # least squares

        # Network prediction with trained weights
        print("Readout model trained, now doing inference (prediction)... ")
        y_predicted = np.dot(W_trained, X_test)

        # performance
        prediction = np.argmax(y_predicted, axis=0)
        perf_all = (prediction == target).sum()/float(len(y_test))
        partial_performance[file_number] = perf_all
        print("Performance on the RandomTask: ", perf_all)

    final_performance_mean.append(partial_performance.mean())
    final_performance_std.append(partial_performance.std())
    final_performance_5p.append(np.percentile(partial_performance, 5))
    final_performance_95p.append(np.percentile(partial_performance, 95))

subplot(414)
plot(final_sequence_lengh, final_performance_mean, '-b', \
                                               label=r'$\sigma = 0.05$')

low_err = np.array(final_performance_mean)-np.array(final_performance_5p)
hig_err = np.array(final_performance_95p)-np.array(final_performance_mean)
errorbar(final_sequence_lengh, final_performance_mean, \
                                     yerr=[low_err, hig_err], color='b')

ylabel('Performance', fontsize=letter_size)
xlabel('L', fontsize=letter_size)
legend(loc='best', prop={'size':letter_size}, frameon=False)
xlim([0, 110])
ylim([0.4, 1.1])
tick_params(axis='both', which='major', labelsize=letter_size)
xticks([0, 20, 40, 60, 80, 100], \
     ['$0$', '$20$', '$40$', '$60$', '$80$', '$100$'])
yticks([0.4, 0.6, 0.8, 1.0],\
            ['$0.4$', '$0.6$', '$0.8$', '$1.0$'])

title('Random Task')
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
