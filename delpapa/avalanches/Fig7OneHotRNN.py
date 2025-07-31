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

def train_rnn(X_train, y_train, X_test, y_test, 
              batch_size=128, epochs=400, learning_rate=0.01, sequence_length=6):
    """
    Train an RNN on the data.
    X_train: numpy array of shape (input_size, time_steps)
    y_train: numpy array of shape (output_size, time_steps)
    """
    input_size = X_train.shape[0]
    output_size = y_train.shape[0]
    
    # Transpose to (time_steps, features)
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    
    # Create sequences using a sliding window with overlap
    train_sequences = []
    train_targets = []
    
    # Create overlapping sequences for training
    for i in range(len(X_train) - sequence_length):
        x_seq = X_train[i:i+sequence_length]
        y_target = y_train[i+sequence_length]
        
        train_sequences.append(x_seq)
        train_targets.append(y_target)
    
    train_sequences = np.array(train_sequences)
    train_targets = np.array(train_targets)
    
    # Create overlapping sequences for testing
    test_sequences = []
    for i in range(len(X_test) - sequence_length):
        test_sequences.append(X_test[i:i+sequence_length])
    test_sequences = np.array(test_sequences)
    
    # Create RNN model with larger hidden size
    model = SimpleRNN(input_size=input_size, 
                     hidden_size=256,  # Increased hidden size
                     output_size=output_size, 
                     sequence_length=sequence_length)
    
    # Training loop with learning rate decay
    n_batches = int(np.ceil(float(len(train_sequences)) / batch_size))
    best_loss = float('inf')
    patience = 20  # epochs to wait before reducing learning rate
    current_patience = patience
    current_lr = learning_rate
    
    for epoch in xrange(epochs):
        # Shuffle training data
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
            model.backward(batch_X, batch_y, current_lr)
        
        avg_loss = total_loss/n_batches
        
        # Learning rate decay
        if avg_loss < best_loss:
            best_loss = avg_loss
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience <= 0:
                current_lr *= 0.5
                current_patience = patience
                print('Reducing learning rate to', current_lr)
        
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f, LR: %.6f' % 
                  (epoch+1, epochs, avg_loss, current_lr))
            
    # Generate predictions
    predictions = []
    for i in range(0, len(test_sequences), batch_size):
        batch = test_sequences[i:i+batch_size]
        if len(batch) > 0:
            output = model.forward(batch)
            batch_predictions = np.argmax(output, axis=1)
            predictions.extend(batch_predictions)
    
    return model, np.array(predictions)


class ImprovedRNN(object):
    def __init__(self, input_size, hidden_size=256, output_size=6, sequence_length=12):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # Initialize weights with Xavier initialization
        # First layer
        self.Wxh1 = self._init_weights((input_size, hidden_size))
        self.Whh1 = self._init_weights((hidden_size, hidden_size))
        # Second layer
        self.Wxh2 = self._init_weights((input_size, hidden_size))  # Changed dimension
        self.Whh2 = self._init_weights((hidden_size, hidden_size))
        # Output layer
        self.Why = self._init_weights((hidden_size, output_size))
        
        # Biases
        self.bh1 = np.zeros((1, hidden_size))
        self.bh2 = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # Batch normalization parameters
        self.gamma1 = np.ones((hidden_size,))
        self.beta1 = np.zeros((hidden_size,))
        self.gamma2 = np.ones((hidden_size,))
        self.beta2 = np.zeros((hidden_size,))
        
    def _init_weights(self, shape):
        """Xavier initialization"""
        scale = np.sqrt(2.0 / sum(shape))
        return np.random.randn(*shape) * scale
    
    def batch_norm(self, x, gamma, beta, training=True):
        """Batch normalization"""
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True) + 1e-8
        else:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True) + 1e-8
            
        x_norm = (x - mu) / np.sqrt(var)
        return gamma * x_norm + beta
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # In the forward method of ImprovedRNN class
    def forward(self, inputs, training=True):
        batch_size = inputs.shape[0]
        
        # Initialize states
        h1 = np.zeros((batch_size, self.hidden_size))
        h2 = np.zeros((batch_size, self.hidden_size))
        
        # Store states for backprop
        self.layer1_states = [h1]
        self.layer2_states = [h2]
        self.inputs = []
        
        outputs = []
        for t in range(self.sequence_length):
            if t < inputs.shape[1]:
                x = inputs[:, t, :]
            else:
                x = np.zeros((batch_size, self.input_size))
            
            self.inputs.append(x)
            
            # First layer
            h1_prev = self.layer1_states[-1]
            h1_raw = np.dot(x, self.Wxh1) + np.dot(h1_prev, self.Whh1) + self.bh1
            h1 = self.relu(self.batch_norm(h1_raw, self.gamma1, self.beta1, training))
            
            if training:
                dropout_mask1 = (np.random.rand(*h1.shape) < 0.8) / 0.8
                h1 *= dropout_mask1
            
            self.layer1_states.append(h1)
            
            # Second layer with skip connection
            h2_prev = self.layer2_states[-1]
            x_proj = np.dot(x, self.Wxh2)  # Project x to hidden size
            h2_raw = np.dot(h2_prev, self.Whh2) + x_proj + h1 + self.bh2  # Add projected input
            h2 = self.relu(self.batch_norm(h2_raw, self.gamma2, self.beta2, training))
            
            if training:
                dropout_mask2 = (np.random.rand(*h2.shape) < 0.8) / 0.8
                h2 *= dropout_mask2
            
            self.layer2_states.append(h2)
            
            # Output layer
            y = np.dot(h2, self.Why) + self.by
            y_prob = self.softmax(y)
            outputs.append(y_prob)
        
        return outputs[-1]
    
    def backward(self, inputs, targets, learning_rate=0.001):
        """
        Backpropagation through time (BPTT)
        """
        batch_size = inputs.shape[0]
        
        # Initialize gradients
        dWxh1 = np.zeros_like(self.Wxh1)  # (input_size, hidden_size)
        dWhh1 = np.zeros_like(self.Whh1)  # (hidden_size, hidden_size)
        dWxh2 = np.zeros_like(self.Wxh2)  # (input_size, hidden_size)
        dWhh2 = np.zeros_like(self.Whh2)  # (hidden_size, hidden_size)
        dWhy = np.zeros_like(self.Why)    # (hidden_size, output_size)
        dbh1 = np.zeros_like(self.bh1)
        dbh2 = np.zeros_like(self.bh2)
        dby = np.zeros_like(self.by)
        
        # Get final states
        h2_last = self.layer2_states[-1]  # (batch_size, hidden_size)
        
        # Compute output gradient
        y_pred = self.forward(inputs)  # (batch_size, output_size)
        dy = y_pred - targets  # (batch_size, output_size)
        
        # Gradient for Why and by
        dWhy = np.dot(h2_last.T, dy)  # (hidden_size, output_size)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # Initialize hidden state gradients
        dh2 = np.dot(dy, self.Why.T)  # (batch_size, hidden_size)
        dh1 = np.zeros((batch_size, self.hidden_size))
        
        # Backpropagate through time
        for t in reversed(range(self.sequence_length)):
            if t >= len(self.layer2_states) - 1:
                continue
                
            # Get states for current timestep
            h1_t = self.layer1_states[t+1]    # (batch_size, hidden_size)
            h2_t = self.layer2_states[t+1]    # (batch_size, hidden_size)
            h1_prev = self.layer1_states[t]   # (batch_size, hidden_size)
            h2_prev = self.layer2_states[t]   # (batch_size, hidden_size)
            x_t = self.inputs[t]              # (batch_size, input_size)
            
            # Second layer gradients
            x_proj = np.dot(x_t, self.Wxh2)   # (batch_size, hidden_size)
            h2_input = np.dot(h2_prev, self.Whh2) + x_proj + h1_t + self.bh2
            dh2_raw = dh2 * self.relu_derivative(h2_input)
            
            # Update gradients for second layer
            dWxh2 += np.dot(x_t.T, dh2_raw)
            dWhh2 += np.dot(h2_prev.T, dh2_raw)
            dbh2 += np.sum(dh2_raw, axis=0, keepdims=True)
            
            # First layer gradients
            h1_input = np.dot(x_t, self.Wxh1) + np.dot(h1_prev, self.Whh1) + self.bh1
            dh1_raw = (dh2_raw + dh1) * self.relu_derivative(h1_input)  # Add skip connection gradient
            
            # Update gradients for first layer
            dWxh1 += np.dot(x_t.T, dh1_raw)
            dWhh1 += np.dot(h1_prev.T, dh1_raw)
            dbh1 += np.sum(dh1_raw, axis=0, keepdims=True)
            
            # Propagate gradients to previous timestep
            dh2 = np.dot(dh2_raw, self.Whh2.T)
            dh1 = np.dot(dh1_raw, self.Whh1.T)
        
        # Clip gradients
        clip_threshold = 5.0
        for grad in [dWxh1, dWhh1, dWxh2, dWhh2, dWhy, dbh1, dbh2, dby]:
            np.clip(grad, -clip_threshold, clip_threshold, out=grad)
        
        # Update weights
        self.Wxh1 -= learning_rate * dWxh1
        self.Whh1 -= learning_rate * dWhh1
        self.Wxh2 -= learning_rate * dWxh2
        self.Whh2 -= learning_rate * dWhh2
        self.Why -= learning_rate * dWhy
        self.bh1 -= learning_rate * dbh1
        self.bh2 -= learning_rate * dbh2
        self.by -= learning_rate * dby

def train_better_rnn(X_train, y_train, X_test, y_test, 
              batch_size=64, epochs=300, learning_rate=0.005, sequence_length=12):
    """
    Train the RNN on the data.
    """
    input_size = X_train.shape[0]
    output_size = y_train.shape[0]
    
    # Transpose data for sequence processing
    X_train = X_train.T  # (time_steps, input_size)
    y_train = y_train.T
    X_test = X_test.T
    
    # Create sequences
    train_sequences = []
    train_targets = []
    
    for i in range(len(X_train) - sequence_length):
        seq = X_train[i:i+sequence_length]
        target = y_train[i+sequence_length-1]
        train_sequences.append(seq)
        train_targets.append(target)
    
    train_sequences = np.array(train_sequences)
    train_targets = np.array(train_targets)
    
    # Create test sequences
    test_sequences = []
    for i in range(len(X_test) - sequence_length):
        seq = X_test[i:i+sequence_length]
        test_sequences.append(seq)
    test_sequences = np.array(test_sequences)
    
    # Initialize RNN
    model = ImprovedRNN(input_size=input_size, 
                       hidden_size=256,
                       output_size=output_size, 
                       sequence_length=sequence_length)
    
    # Training loop with early stopping
    n_batches = int(np.ceil(float(len(train_sequences)) / batch_size))
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    initial_lr = learning_rate
    
    for epoch in xrange(epochs):
        # Learning rate decay
        current_lr = initial_lr / (1 + epoch * 0.01)
        
        # Shuffle training data
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
            
            # Calculate loss
            loss = -np.mean(np.sum(batch_y * np.log(np.clip(output, 1e-10, 1.0)), axis=1))
            total_loss += loss
            
            # Backward pass
            model.backward(batch_X, batch_y, current_lr)
        
        avg_loss = total_loss / n_batches
        
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f, LR: %.6f' % 
                  (epoch+1, epochs, avg_loss, current_lr))
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping at epoch", epoch + 1)
            break
    
    # Generate predictions
    predictions = []
    for i in range(0, len(test_sequences), batch_size):
        batch = test_sequences[i:i+batch_size]
        if len(batch) > 0:
            output = model.forward(batch, training=False)
            batch_predictions = np.argmax(output, axis=1)
            predictions.extend(batch_predictions)
    
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

        # Readout training using RNN
        print("Training the readout model...")
        # Prepare next-letter prediction data
        X_train_current = X_train[:, :-1]  # all but last letter
        y_train_next = y_train[:, 1:]      # all but first letter

        # Network prediction with trained weights
        X_test_current = X_test[:, :-1]  # remove last letter as we can't verify its prediction
        y_test_next = y_test[:, 1:]      # Remove first timestep

        print("X_train_current shape:", X_train_current.shape)
        print("y_train_next shape:", y_train_next.shape)
        print("X_test_current shape:", X_test_current.shape)
        print("y_test_next shape:", y_test_next.shape)

        # Train RNN model with adjusted parameters
        print("Training RNN model...")
        model, prediction = train_better_rnn(
            X_train_current, y_train_next, 
            X_test_current, y_test_next,
            batch_size=64,
            epochs=300,
            learning_rate=0.005,
            sequence_length=12
        )
        # model, prediction = train_rnn(
        #     X_train_current, y_train_next, 
        #     X_test_current, y_test_next,
        #     batch_size=128,          
        #     epochs=400,              
        #     learning_rate=0.01,      
        #     sequence_length=6        
        # )
        print("RNN model trained.")

        # Convert one-hot test targets back to indices
        target = np.argmax(y_test_next, axis=0)
        prediction = np.array(prediction)

        # Debug prints
        print("Target shape:", target.shape)
        print("Prediction shape:", prediction.shape)
        print("Unique target values:", np.unique(target))
        print("Unique prediction values:", np.unique(prediction))
        print("First few targets:", target[:10])
        print("First few predictions:", prediction[:10])

        # Make sure lengths match
        min_len = min(len(target), len(prediction))
        target = target[:min_len]
        prediction = prediction[:min_len]

        perf_all = np.mean(prediction == target)
        print("Prediction done, performance: ", perf_all)

        # reduced performance
        except_first = np.where(np.logical_or(
                            np.logical_or(target == 1, target == 2),
                            np.logical_or(target == 4, target == 5)))[0]

        y_test_red = target[except_first]
        y_pred_red = prediction[except_first]

        # Debug prints for reduced performance
        print("Number of samples in reduced set:", len(y_test_red))
        print("Unique values in reduced test set:", np.unique(y_test_red))
        print("Unique values in reduced predictions:", np.unique(y_pred_red))

        perf_red = np.mean(y_test_red == y_pred_red)
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
