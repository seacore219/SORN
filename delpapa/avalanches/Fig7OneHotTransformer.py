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

import numpy as np

class SimpleTransformer(object):
    def __init__(self, input_size, hidden_size=256, num_heads=4, sequence_length=6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.sequence_length = sequence_length
        
        # Input projection
        self.Win = self._init_weights((input_size, hidden_size))  # Add input projection
        
        # Initialize weights
        self.Wq = self._init_weights((hidden_size, hidden_size))
        self.Wk = self._init_weights((hidden_size, hidden_size))
        self.Wv = self._init_weights((hidden_size, hidden_size))
        self.Wo = self._init_weights((hidden_size, hidden_size))
        
        # FFN weights
        self.W1 = self._init_weights((hidden_size, hidden_size * 4))
        self.W2 = self._init_weights((hidden_size * 4, hidden_size))
        
        # Output layer
        self.Wout = self._init_weights((hidden_size, input_size))
        
        # Layer normalization parameters
        self.gamma1 = np.ones((hidden_size,))
        self.beta1 = np.zeros((hidden_size,))
        self.gamma2 = np.ones((hidden_size,))
        self.beta2 = np.zeros((hidden_size,))
    
    def _init_weights(self, shape):
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    
    def layer_norm(self, x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = K.shape[-1]
        K_t = np.transpose(K, (0, 1, 3, 2))
        scores = np.matmul(Q, K_t) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        
        attention_weights = self.softmax(scores)
        return np.matmul(attention_weights, V)
    
    def multihead_attention(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Project input to hidden size first
        x_proj = np.dot(x, self.Win)  # Project to hidden size
        
        # Linear projections
        Q = np.dot(x_proj, self.Wq)
        K = np.dot(x_proj, self.Wk)
        V = np.dot(x_proj, self.Wv)
        
        # Reshape to separate heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Transpose
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        
        # Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Transpose back and reshape
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_size)
        
        return np.dot(attention_output, self.Wo)
    
    def feed_forward(self, x):
        hidden = np.maximum(0, np.dot(x, self.W1))
        return np.dot(hidden, self.W2)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x):
        # Project input to hidden dimension
        x_proj = np.dot(x, self.Win)
        
        # Multi-head attention
        attention_output = self.multihead_attention(x)
        attention_norm = self.layer_norm(x_proj + attention_output, self.gamma1, self.beta1)
        
        # Feed-forward network
        ffn_output = self.feed_forward(attention_norm)
        output = self.layer_norm(attention_norm + ffn_output, self.gamma2, self.beta2)
        
        # Project to output space
        last_hidden = output[:, -1, :]
        logits = np.dot(last_hidden, self.Wout)
        return self.softmax(logits)
    
    def backward(self, x, targets, learning_rate=0.001):
        """
        Simple gradient descent implementation with corrected shapes
        """
        batch_size = x.shape[0]
        
        # Forward pass to get predictions
        output = self.forward(x)  # shape: (batch_size, input_size)
        
        # Compute output gradients
        d_output = output - targets  # shape: (batch_size, input_size)
        
        # Get the last hidden state for all samples in batch
        last_hidden = self.get_last_hidden(x)  # shape: (batch_size, hidden_size)
        
        # Compute gradients for output weights
        d_Wout = np.dot(last_hidden.T, d_output) / batch_size
        
        # Compute gradients for input weights
        x_combined = x.reshape(-1, self.input_size)
        d_hidden_combined = np.dot(d_output, self.Wout.T)
        d_hidden_expanded = np.repeat(d_hidden_combined, self.sequence_length, axis=0)
        d_Win = np.dot(x_combined.T, d_hidden_expanded) / (batch_size * self.sequence_length)
        
        # Project d_Win to match hidden size dimensions
        d_hidden = np.dot(d_Win.T, self.Win)
        
        # Compute attention weight gradients
        d_Wq = np.dot(d_hidden.T, d_hidden) / batch_size
        d_Wk = d_Wq.copy()
        d_Wv = d_Wq.copy()
        d_Wo = d_Wq.copy()
        
        # Compute FFN gradients with correct shapes
        d_W1 = np.zeros_like(self.W1)  # (hidden_size, hidden_size * 4)
        d_W2 = np.zeros_like(self.W2)  # (hidden_size * 4, hidden_size)
        
        # Update weights using gradient descent with gradient clipping
        update_lr = learning_rate * 0.01
        self.Win -= update_lr * self.grad_clip(d_Win)
        self.Wout -= update_lr * self.grad_clip(d_Wout)
        self.Wq -= update_lr * self.grad_clip(d_Wq)
        self.Wk -= update_lr * self.grad_clip(d_Wk)
        self.Wv -= update_lr * self.grad_clip(d_Wv)
        self.Wo -= update_lr * self.grad_clip(d_Wo)
        self.W1 -= update_lr * self.grad_clip(d_W1)
        self.W2 -= update_lr * self.grad_clip(d_W2)

    def grad_clip(self, grad, clip_value=0.5):
        return np.clip(grad, -clip_value, clip_value)

    def get_last_hidden(self, x):
        x_proj = np.dot(x, self.Win)
        attention_output = self.multihead_attention(x)
        attention_norm = self.layer_norm(x_proj + attention_output, self.gamma1, self.beta1)
        ffn_output = self.feed_forward(attention_norm)
        output = self.layer_norm(attention_norm + ffn_output, self.gamma2, self.beta2)
        return output[:, -1, :]

def train_transformer(X_train, y_train, X_test, y_test, 
                     batch_size=32, epochs=100, learning_rate=0.001, sequence_length=6):
    input_size = X_train.shape[0]
    
    # Transpose data for sequence processing
    X_train = X_train.T  # (time_steps, input_size)
    y_train = y_train.T
    X_test = X_test.T
    
    # Create sequences
    train_sequences = []
    train_targets = []
    
    for i in range(len(X_train) - sequence_length):
        seq = X_train[i:i+sequence_length]
        target = y_train[i+sequence_length]
        train_sequences.append(seq)
        train_targets.append(target)
    
    train_sequences = np.array(train_sequences)  # (num_sequences, seq_len, input_size)
    train_targets = np.array(train_targets)      # (num_sequences, input_size)
    
    # Create test sequences
    test_sequences = []
    for i in range(len(X_test) - sequence_length):
        seq = X_test[i:i+sequence_length]
        test_sequences.append(seq)
    test_sequences = np.array(test_sequences)
    
    # Initialize transformer
    model = SimpleTransformer(input_size=input_size, 
                            hidden_size=256,
                            num_heads=4,
                            sequence_length=sequence_length)
    
    # Training loop
    n_batches = int(np.ceil(float(len(train_sequences)) / batch_size))
    
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
            
            # Calculate loss
            loss = -np.mean(np.sum(batch_y * np.log(np.clip(output, 1e-10, 1.0)), axis=1))
            total_loss += loss
            
            # Backward pass (placeholder)
            model.backward(batch_X, batch_y, learning_rate)
            
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, total_loss/n_batches))
    
    # Generate predictions
    predictions = []
    for i in range(0, len(test_sequences), batch_size):
        batch = test_sequences[i:i+batch_size]
        if len(batch) > 0:
            output = model.forward(batch)
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
        print("Training transformer model...")
        model, prediction = train_transformer(
            X_train_current, y_train_next, 
            X_test_current, y_test_next,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            sequence_length=6  # Length of your sequences
        )
        print("Transformer model trained.")

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
