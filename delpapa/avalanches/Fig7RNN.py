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

class SimpleRNN(object):
    def __init__(self, input_size, hidden_size=64, output_size=6, sequence_length=10):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # Initialize weights with Xavier/Glorot initialization
        # Input to hidden weights
        self.Wxh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        # Hidden to hidden weights (recurrent connections)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size * 2))
        # Hidden to output weights
        self.Why = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        
        # Biases
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, inputs):
        """
        Forward pass for the RNN
        inputs: numpy array of shape (batch_size, sequence_length, input_size)
        """
        batch_size = inputs.shape[0]
        
        # Initialize hidden state with zeros
        h = np.zeros((batch_size, self.hidden_size))
        
        # Store all hidden states and outputs for backpropagation
        self.h_states = [h]
        self.x_inputs = []
        
        outputs = []
        for t in range(self.sequence_length):
            if t < inputs.shape[1]:
                x = inputs[:, t, :]
            else:
                # Pad with zeros if sequence is shorter than sequence_length
                x = np.zeros((batch_size, self.input_size))
                
            self.x_inputs.append(x)
            
            # RNN step: h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)
            h_prev = self.h_states[-1]
            h_raw = np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh
            h = self.relu(h_raw)
            
            # Store hidden state for backpropagation
            self.h_states.append(h)
            
            # Output layer: y_t = softmax(W_hy * h_t + b_y)
            y = np.dot(h, self.Why) + self.by
            y_prob = self.softmax(y)
            outputs.append(y_prob)
            
        # Stack the outputs for all time steps
        # We only care about the final output for classification
        return outputs[-1]
    
    def backward(self, inputs, targets, learning_rate=0.001):
        """
        Backpropagation through time (BPTT)
        inputs: numpy array of shape (batch_size, sequence_length, input_size)
        targets: numpy array of shape (batch_size, output_size)
        """
        batch_size = inputs.shape[0]
        
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Get the last hidden state
        h_last = self.h_states[-1]
        
        # Backprop from output layer
        # Compute error at output
        y_pred = np.dot(h_last, self.Why) + self.by
        y_prob = self.softmax(y_pred)
        dy = y_prob - targets
        
        # Gradient for Why and by
        dWhy = np.dot(h_last.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # Backpropagate through time
        dh_next = np.dot(dy, self.Why.T)
        
        for t in reversed(range(self.sequence_length)):
            # Get the current and previous hidden states
            h_current = self.h_states[t+1]
            h_prev = self.h_states[t]
            
            # Get the current input
            x_t = self.x_inputs[t]
            
            # Backprop through the hidden layer
            dh_raw = dh_next * self.relu_derivative(np.dot(x_t, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
            
            # Compute gradients
            dWxh += np.dot(x_t.T, dh_raw)
            dWhh += np.dot(h_prev.T, dh_raw)
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            # Propagate error to previous time step
            dh_next = np.dot(dh_raw, self.Whh.T)
        
        # Clip gradients to prevent exploding gradients
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

def train_rnn(X_train, y_read_train, X_test, y_read_test, 
              batch_size=32, epochs=100, learning_rate=0.001, sequence_length=10):
    """
    Train an RNN on the data.
    X_train: numpy array of shape (input_size, time_steps)
    y_read_train: numpy array of shape (output_size, time_steps)
    """
    input_size = X_train.shape[0]
    output_size = y_read_train.shape[0]
    
    # Transpose data for easier batch processing
    X_train = X_train.T  # Now shape (time_steps, input_size)
    y_train = y_read_train.T  # Now shape (time_steps, output_size)
    X_test = X_test.T
    
    # Create RNN model
    model = SimpleRNN(input_size=input_size, hidden_size=64, 
                      output_size=output_size, sequence_length=sequence_length)
    
    # Reshape data for sequence processing
    # We'll create sequences of length sequence_length by sliding a window
    train_sequences = []
    train_targets = []
    
    for i in range(len(X_train) - sequence_length):
        # Extract sequence
        seq = X_train[i:i+sequence_length]
        # Target is the last element in the sequence
        target = y_train[i+sequence_length-1]
        
        train_sequences.append(seq)
        train_targets.append(target)
    
    train_sequences = np.array(train_sequences)
    train_targets = np.array(train_targets)
    
    # If we have too few sequences, adjust sequence_length
    if len(train_sequences) < batch_size:
        sequence_length = max(1, len(X_train) // 2)
        print("Adjusting sequence length to", sequence_length)
        model.sequence_length = sequence_length
        
        # Recreate sequences
        train_sequences = []
        train_targets = []
        
        for i in range(len(X_train) - sequence_length):
            seq = X_train[i:i+sequence_length]
            target = y_train[i+sequence_length-1]
            
            train_sequences.append(seq)
            train_targets.append(target)
        
        train_sequences = np.array(train_sequences)
        train_targets = np.array(train_targets)
    
    # Create test sequences in a similar manner
    test_sequences = []
    test_targets = []
    
    for i in range(len(X_test) - sequence_length):
        seq = X_test[i:i+sequence_length]
        target = None  # We don't have targets for test, will be computed later
        
        test_sequences.append(seq)
        test_targets.append(target)
    
    test_sequences = np.array(test_sequences)
    
    # Number of batches
    n_batches = int(np.ceil(float(len(train_sequences)) / batch_size))
    
    # Training loop
    for epoch in xrange(epochs):
        # Shuffle data
        shuffle_idx = np.random.permutation(len(train_sequences))
        train_sequences = train_sequences[shuffle_idx]
        train_targets = train_targets[shuffle_idx]
        
        total_loss = 0
        for batch_idx in xrange(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_sequences))
            
            batch_X = train_sequences[start_idx:end_idx]
            batch_y = train_targets[start_idx:end_idx]
            
            # Forward pass
            output = model.forward(batch_X)
            
            # Calculate loss (cross-entropy)
            loss = -np.mean(np.sum(batch_y * np.log(np.clip(output, 1e-10, 1.0)), axis=1))
            total_loss += loss
            
            # Backward pass
            model.backward(batch_X, batch_y, learning_rate)
            
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, total_loss/n_batches))
    
    # Make predictions on test data
    predictions = []
    for i in range(0, len(test_sequences), batch_size):
        batch = test_sequences[i:i+batch_size]
        if len(batch) > 0:
            output = model.forward(batch)
            batch_predictions = np.argmax(output, axis=1)
            predictions.extend(batch_predictions)
    
    # If test sequences is empty but we still need predictions for all test points
    if len(test_sequences) == 0:
        # Handle the case by using single points as input
        # This is a fallback for very short test sets
        predictions = []
        for i in range(len(X_test)):
            # Create a sequence with repeated samples if necessary
            x = X_test[i]
            seq = np.tile(x, (1, sequence_length, 1))
            output = model.forward(seq)
            pred = np.argmax(output, axis=1)[0]
            predictions.append(pred)
    
    return model, np.array(predictions)


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
        exper_path =  experiment_folder
        h5 = tables.open_file(os.path.join(exper_path,exper),'r')
        data = h5.root

        # create training and test arrays
        train_steps = data.c.steps_readouttrain[0]
        print("Number of training steps: ", train_steps)
        test_steps = data.c.steps_readouttest[0]
        print("Number of testing steps: ", test_steps)

        # the letter to be presented at the time step
        y_train = data.countingletter[0][:train_steps]
        y_test = data.countingletter[0][train_steps:train_steps+test_steps]

        # # divide readout into n different units
        # y_read_train = np.zeros((6, len(y_train)))
        # y_read_test = np.zeros((6, len(y_test)))
        # for i, y in enumerate(y_train):
        #     y_read_train[y, i] = 1
        # for i, y in enumerate(y_test):
        #     y_read_test[y, i] = 1
        # First, let's see what we're working with
        print("y_train dtype:", y_train.dtype)
        print("unique values in y_train:", np.unique(y_train))

        # Then modify the code to ensure integer indices:
        n_classes = len(np.unique(y_train))  # This will be 10 based on your output
        print("Number of classes to distinguish: ", n_classes)
        y_read_train = np.zeros((n_classes, len(y_train)))
        y_read_test = np.zeros((n_classes, len(y_test)))

        # Convert to integers when indexing
        for i, y in enumerate(y_train):
            y_read_train[int(y), i] = 1
        for i, y in enumerate(y_test):
            y_read_test[int(y), i] = 1

        target = np.argmax(y_read_test, axis=0) # target for the training data

        # internal state before letter presentation
        X_train = (data.countingactivity[0][:,:train_steps] >= 0) + 0.
        X_test = (data.countingactivity[0][:,train_steps:train_steps+test_steps] >= 0) + 0.

        h5.close()

        # Readout training
        print("Training the readout model...")
        #X_train_pinv = np.linalg.pinv(X_train) # MP pseudo-inverse
        #W_trained = np.dot(y_read_train, X_train_pinv) # least squares weights

        # Network prediction with trained weights
        #y_predicted = np.dot(W_trained, X_test)
        #import ipdb; ipdb.set_trace()
        # Train recurrent neural network readout
        sequence_length = min(10, train_steps // 2)  # Adjust based on data size
        model, prediction = train_rnn(X_train, y_read_train, X_test, y_read_test, sequence_length=sequence_length)
    
        # Calculate performance
        # Adjust target to match prediction length if necessary
        if len(prediction) < len(target):
            target = target[-len(prediction):]
        elif len(prediction) > len(target):
            prediction = prediction[:len(target)]

        print("Readout model trained. Now running inference (prediction)... ")

        # performance
        # prediction = np.argmax(y_predicted, axis=0)
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
