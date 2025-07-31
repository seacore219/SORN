####
# Script for eigenvalue analysis of SORN network
####
from __future__ import division  # has to be imported in every file
from importlib import import_module
import imp  # add this for Python 2.7 file importing
import os
import sys
sys.path.insert(1, "../")
sys.path.append("D:/Users/seaco/SORN")

from pylab import *

import tables
from tempfile import TemporaryFile

# work around to run powerlaw package [Alstott et al. 2014]
import powerlaw as pl

import numpy as np

from common.sorn import Sorn

# Define experiment path
experiment_folder = '..\\backup\\test_single\\MySmallCountingTask\\'
sorn_file_base = "net.pickle"
sorn_path = experiment_folder
print(os.path.join(sorn_path,sorn_file_base))

# Load SORN network
sorn_obj = Sorn.quickload(os.path.join(sorn_path,sorn_file_base))
print(sorn_obj.W_ee.W.toarray())

# Extract W_ee as a dense array
W_ee = sorn_obj.W_ee.W.toarray()
num_rows = W_ee.shape[0]
print("Number of rows in W_ee:", num_rows)

# Construct X = W_ee^T * W_ee
W_ee_transpose = W_ee.transpose()
X_unnormalized = np.dot(W_ee_transpose, W_ee)
X = X_unnormalized / float(num_rows)
print("Normalized X = (W_ee^T * W_ee) / num_rows:")
print(X)

# Compute eigenvalues of normalized X
eigenvalues = np.linalg.eigvals(X)
print("Eigenvalues of normalized X:")
print(eigenvalues)

# 1. Compute histogram of eigenvalues
# Remove any very small eigenvalues that might be numerical artifacts
min_eigenvalue_threshold = 1e-10
valid_eigenvalues = eigenvalues.real[eigenvalues.real > min_eigenvalue_threshold]

# Create histogram data
hist_counts, hist_bins = np.histogram(valid_eigenvalues, bins=30)
bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2.0

# User-adjustable threshold for filtering low counts in log-log analysis
min_count_threshold = 2  # Change this value to filter out bins with few counts

# 2 & 3. Create a figure with 4 subplots
figure(figsize=(15, 12))

# Original complex plane plot
subplot(221)
scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6)
axhline(y=0, color='k', linestyle='-', alpha=0.3)
axvline(x=0, color='k', linestyle='-', alpha=0.3)
grid(alpha=0.3)
xlabel('Real part')
ylabel('Imaginary part')
title('Eigenvalues in complex plane')

# Original magnitude plot
subplot(222)
eig_magnitudes = np.abs(eigenvalues)
sorted_indices = np.argsort(eig_magnitudes)[::-1]
sorted_magnitudes = eig_magnitudes[sorted_indices]
stem(range(len(sorted_magnitudes)), sorted_magnitudes, 'b-', 'bo', 'k-')
grid(alpha=0.3)
xlabel('Index')
ylabel('Magnitude')
title('Eigenvalue magnitudes (sorted)')

# Histogram plot (linear scale)
subplot(223)
bar(bin_centers, hist_counts, width=hist_bins[1]-hist_bins[0], alpha=0.7)
grid(alpha=0.3)
xlabel('Eigenvalue')
ylabel('Frequency')
title('Histogram of eigenvalues')

# Log-log plot with power law fit
subplot(224)

# Filter zero values and low counts for log-log plot
mask = (hist_counts >= min_count_threshold) & (bin_centers > 0)
log_bins = bin_centers[mask]
log_counts = hist_counts[mask]

# 4. Fit power law using linear regression on log-log data
if len(log_bins) > 2:  # Need at least 3 points for meaningful fit
    log_x = np.log10(log_bins)
    log_y = np.log10(log_counts)
    
    # Linear fit on log-log scale
    coeffs = np.polyfit(log_x, log_y, 1)
    power_law_exponent = coeffs[0]
    
    # Generate fitted line points
    x_fit = np.logspace(np.min(log_x), np.max(log_x), 100)
    y_fit = 10**(coeffs[1]) * x_fit**power_law_exponent
    
    # Plot
    loglog(log_bins, log_counts, 'o', label='Data (counts >= {})'.format(min_count_threshold))
    loglog(x_fit, y_fit, 'r-', label='Power law fit, exponent = {:.3f}'.format(power_law_exponent))
    legend()
else:
    loglog(log_bins, log_counts, 'o', label='Data (counts >= {})'.format(min_count_threshold))
    text(0.5, 0.5, 'Not enough data points for power law fit', 
         horizontalalignment='center', verticalalignment='center',
         transform=gca().transAxes)
    
grid(alpha=0.3, which='both')
xlabel('Eigenvalue (log scale)')
ylabel('Frequency (log scale)')
title('Log-log plot of eigenvalue distribution')

# Alternative power law fitting using the powerlaw package
try:
    # More sophisticated fitting with goodness-of-fit tests
    fit = pl.Fit(valid_eigenvalues, discrete=False)
    alpha = fit.alpha
    xmin = fit.xmin
    D = fit.D
    power_law_info = '\nPowerlaw fit: alpha = {:.3f}, xmin = {:.6f}, D = {:.3f}'.format(alpha, xmin, D)
except Exception as e:
    print("Error in powerlaw fitting:", e)
    power_law_info = '\nCould not fit powerlaw using powerlaw package'

# Update the suptitle with max eigenvalue info and power law info
max_eigenvalue_idx = np.argmax(eig_magnitudes)
max_eigenvalue = eigenvalues[max_eigenvalue_idx]
max_eigenvalue_abs = np.abs(max_eigenvalue)
suptitle('Max eigenvalue: {0:.4f} (magnitude: {1:.4f}){2}\nCount threshold for log-log fit: {3}'.format(
    max_eigenvalue, max_eigenvalue_abs, power_law_info, min_count_threshold))

tight_layout()
savefig(os.path.join(sorn_path, 'eigenvalues_analysis_count{}.png'.format(min_count_threshold)), dpi=300)
show()

# Print additional information about the power law fit
if 'fit' in locals():
    print("\nPower law analysis results:")
    print("Alpha (power law exponent):", fit.alpha)
    print("xmin (lower bound of power law behavior):", fit.xmin)
    print("D (Kolmogorov-Smirnov statistic):", fit.D)
    
    # Compare with alternative distributions
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print("\nComparing power law with exponential distribution:")
    print("Log-likelihood ratio R:", R)
    print("p-value:", p)
    print("(R > 0 and small p-value supports power law over exponential)")
    
    R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    print("\nComparing power law with lognormal distribution:")
    print("Log-likelihood ratio R:", R)
    print("p-value:", p)
    print("(R > 0 and small p-value supports power law over lognormal)")