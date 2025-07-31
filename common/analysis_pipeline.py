print("[INFO] Starting import of libraries and function definitions...")
import os
import re
import tables
print("[INFO] tables imported successfully.")
import numpy as np
import pandas as pd
import math
import mrestimator as mre
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
import os.path as op
from decimal import Decimal, ROUND_HALF_UP
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import welch
from scipy.interpolate import interp1d
from copy import deepcopy as cdc
import time
# Remove these imports as we'll integrate the functions directly
# import criticality as cr
# from criticality import pvaluenew2 as pv
# from criticality import exclude as ex
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Remove this import as well
# import criticality_tumbleweed as crt
import psutil
import gc
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
gc.collect()
tables.file._open_files.close_all()
print("[INFO] all libraries imported successfully.")

##############################################################
# Integrated criticality functions
##############################################################

def get_avalanches(raster, perc=0.2, ncells=-1, const_threshold=None):
    """
    Detect avalanches in neural raster data.
    
    Parameters:
    -----------
    raster : array_like
        2D array of neural activity (neurons x time)
    perc : float
        Percentile threshold for activity detection
    ncells : int
        Number of cells to consider (-1 for all)
    const_threshold : float or None
        Constant threshold instead of percentile-based
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'S': avalanche sizes
        - 'T': avalanche durations
        - 'shapes': avalanche temporal profiles
    """
    if ncells == -1:
        ncells = raster.shape[0]
    
    # Calculate activity threshold
    if const_threshold is not None:
        threshold = const_threshold
    else:
        # Use percentile of non-zero activity
        activity = np.sum(raster[:ncells], axis=0)
        non_zero_activity = activity[activity > 0]
        if len(non_zero_activity) == 0:
            return {'S': np.array([]), 'T': np.array([]), 'shapes': []}
        threshold = np.percentile(non_zero_activity, perc * 100)
    
    # Detect avalanches
    activity = np.sum(raster[:ncells], axis=0)
    above_threshold = activity > threshold
    
    # Find avalanche boundaries
    avalanches = []
    shapes = []
    in_avalanche = False
    current_avalanche = []
    
    for i, above in enumerate(above_threshold):
        if above and not in_avalanche:
            # Start of avalanche
            in_avalanche = True
            current_avalanche = [activity[i]]
        elif above and in_avalanche:
            # Continue avalanche
            current_avalanche.append(activity[i])
        elif not above and in_avalanche:
            # End of avalanche
            in_avalanche = False
            if len(current_avalanche) > 0:
                avalanches.append(current_avalanche)
                shapes.append(np.array(current_avalanche))
            current_avalanche = []
    
    # Handle case where recording ends during avalanche
    if len(current_avalanche) > 0:
        avalanches.append(current_avalanche)
        shapes.append(np.array(current_avalanche))
    
    # Calculate sizes and durations
    S = np.array([np.sum(av) for av in avalanches])
    T = np.array([len(av) for av in avalanches])
    
    return {'S': S, 'T': T, 'shapes': shapes}


def pvaluenew(data, tau, xmin, nfactor=1, max_time=7200, verbose=True):
    """
    Calculate p-value for power-law distribution using KS test.
    
    Parameters:
    -----------
    data : array_like
        Data to test for power-law distribution
    tau : float
        Power-law exponent
    xmin : float
        Minimum value for power-law fit
    nfactor : float
        Factor for adjusting xmin search range
    max_time : float
        Maximum computation time in seconds
    verbose : bool
        Print progress information
    
    Returns:
    --------
    tuple : (p_value, ks_statistic, figure_handle, optimal_xmin)
    """
    import time
    start_time = time.time()
    
    # Clean data
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = data[data >= xmin]
    
    if len(data) == 0:
        return np.nan, np.nan, None, xmin
    
    n = len(data)
    
    # Generate synthetic power-law data
    n_synthetic = 1000
    ks_values = []
    
    for i in range(n_synthetic):
        if time.time() - start_time > max_time:
            if verbose:
                print(f"Reached time limit after {i} iterations")
            break
            
        # Generate synthetic data using inverse transform sampling
        u = np.random.uniform(0, 1, n)
        synthetic = xmin * (1 - u) ** (-1 / (tau - 1))
        
        # Calculate KS statistic
        ks_stat, _ = stats.ks_2samp(data, synthetic)
        ks_values.append(ks_stat)
    
    # Calculate p-value
    ks_observed, _ = stats.ks_2samp(data, 
                                    xmin * (1 - np.random.uniform(0, 1, n)) ** (-1 / (tau - 1)))
    p_value = np.sum(np.array(ks_values) >= ks_observed) / len(ks_values)
    
    # Create CDF plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Empirical CDF
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, ecdf, 'b-', label='Empirical CDF')
    
    # Theoretical CDF
    theoretical_cdf = 1 - (xmin / sorted_data) ** (tau - 1)
    ax.plot(sorted_data, theoretical_cdf, 'r--', label='Theoretical CDF')
    
    ax.set_xscale('log')
    ax.set_xlabel('Value')
    ax.set_ylabel('CDF')
    ax.set_title(f'KS Test: p-value = {p_value:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return p_value, ks_observed, fig, xmin


def EXCLUDE(data, bm, nfactor=1, verbose=True):
    """
    Find optimal xmin and xmax for power-law fitting using maximum likelihood.
    
    Parameters:
    -----------
    data : array_like
        Data to fit
    bm : int
        Minimum number of data points to include
    nfactor : float
        Factor for adjusting search range
    verbose : bool
        Print progress information
    
    Returns:
    --------
    tuple : (xmax, xmin, exponent)
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    data = np.sort(data)
    
    if len(data) < bm:
        if verbose:
            print(f"Not enough data points: {len(data)} < {bm}")
        return np.nan, np.nan, np.nan
    
    # Find unique values
    unique_vals = np.unique(data)
    if len(unique_vals) < 3:
        if verbose:
            print("Not enough unique values for fitting")
        return np.nan, np.nan, np.nan
    
    # Search for optimal xmin
    best_ks = np.inf
    best_xmin = unique_vals[0]
    best_alpha = 2.0
    
    for xmin_candidate in unique_vals[:-bm]:
        data_tail = data[data >= xmin_candidate]
        if len(data_tail) < bm:
            continue
            
        # Estimate alpha using MLE
        alpha = 1 + len(data_tail) / np.sum(np.log(data_tail / xmin_candidate))
        
        # Calculate KS statistic
        cdf_data = np.arange(1, len(data_tail) + 1) / len(data_tail)
        cdf_model = 1 - (xmin_candidate / np.sort(data_tail)) ** (alpha - 1)
        ks_stat = np.max(np.abs(cdf_data - cdf_model))
        
        if ks_stat < best_ks:
            best_ks = ks_stat
            best_xmin = xmin_candidate
            best_alpha = alpha
    
    # Find xmax based on where the fit breaks down
    data_tail = data[data >= best_xmin]
    xmax_candidates = np.unique(data_tail)
    
    best_xmax = xmax_candidates[-1]
    for i in range(len(xmax_candidates) - bm, 0, -1):
        xmax_candidate = xmax_candidates[i]
        data_range = data_tail[data_tail <= xmax_candidate]
        
        if len(data_range) < bm:
            continue
            
        # Check if power law still holds
        alpha_range = 1 + len(data_range) / np.sum(np.log(data_range / best_xmin))
        if abs(alpha_range - best_alpha) / best_alpha < 0.1:  # 10% tolerance
            best_xmax = xmax_candidate
        else:
            break
    
    if verbose:
        print(f"Found: xmin={best_xmin}, xmax={best_xmax}, alpha={best_alpha:.3f}")
    
    return best_xmax, best_xmin, best_alpha


def calc_d2_KLr(time_series, embedding_dim):
    """
    Calculate correlation dimension (d2) using Kantz-Schreiber algorithm.
    
    Parameters:
    -----------
    time_series : array_like
        1D time series data
    embedding_dim : int
        Embedding dimension for phase space reconstruction
    
    Returns:
    --------
    float : Correlation dimension d2
    """
    time_series = np.array(time_series).flatten()
    n = len(time_series)
    
    if n < 100 * embedding_dim:
        warnings.warn(f"Time series may be too short for reliable d2 estimation " +
                     f"(n={n}, embedding_dim={embedding_dim})")
    
    # Phase space reconstruction using time-delay embedding
    tau = 1  # Time delay (could be optimized using mutual information)
    embedded = np.zeros((n - (embedding_dim - 1) * tau, embedding_dim))
    
    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * tau:n - (embedding_dim - 1 - i) * tau]
    
    # Calculate correlation sum for different radii
    n_points = min(1000, len(embedded))  # Subsample for efficiency
    if len(embedded) > n_points:
        indices = np.random.choice(len(embedded), n_points, replace=False)
        embedded_sample = embedded[indices]
    else:
        embedded_sample = embedded
    
    # Use k-d tree for efficient neighbor searching
    nbrs = NearestNeighbors(algorithm='kd_tree', metric='euclidean')
    nbrs.fit(embedded_sample)
    
    # Calculate distances
    distances, _ = nbrs.kneighbors(embedded_sample)
    distances = distances[:, 1:]  # Exclude self-distances
    
    # Calculate correlation integral for different radii
    radii = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 20)
    correlation_integral = []
    
    for r in radii:
        count = np.sum(distances <= r)
        c_r = count / (n_points * (n_points - 1))
        if c_r > 0:
            correlation_integral.append(c_r)
        else:
            correlation_integral.append(np.nan)
    
    correlation_integral = np.array(correlation_integral)
    valid_mask = ~np.isnan(correlation_integral) & (correlation_integral > 0)
    
    if np.sum(valid_mask) < 3:
        return np.nan
    
    # Fit line in log-log space to estimate d2
    log_radii = np.log(radii[valid_mask])
    log_corr = np.log(correlation_integral[valid_mask])
    
    # Find linear region (typically in the middle of the range)
    n_valid = len(log_radii)
    start_idx = n_valid // 4
    end_idx = 3 * n_valid // 4
    
    if end_idx - start_idx < 3:
        return np.nan
    
    # Linear fit to estimate d2
    coeffs = np.polyfit(log_radii[start_idx:end_idx], 
                       log_corr[start_idx:end_idx], 1)
    d2 = coeffs[0]
    
    return d2

##############################################################
# Define the necessary functions here
##############################################################
print("[INFO] Searching for directories with simulation results...")
## Need to find all the directories where there are valid sims
# base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\GitHub\\SORN\\mu=0.02_sigma=0.05_500K+3.5M_plastic_raster\\test_single'
base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\SORN_PC\\backup\\noisevar\\finegrain\\batch_hip0.0800_fp0.00_cde0.00_cdi0.00_sig0.03'
# base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\CharlesSORNneo\\backup\\test_single\\batch_hip0.0075_fp0.01_cde0.01_cdi0.01
# base_dir = 'C:\\Users\\seaco\\Downloads\\sims\\randn_10%_e_hip0.2_uext0.2'
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+base_).*_raster"
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+)_(\d+)K.*_sigma_(\d+\.\d+)_.*raster"
#->pattern = r"SPmu=(0.08)_sigma=(0.05)_(\d+)K.*_sigma_(0.05)_.*raster"
# pattern = r"202"
print("-- base_dir dir: ", base_dir)
##############################################################

# Firing activity susceptibility (variance)
def susceptibility(raster):
    num_rows, num_columns = raster.shape
    term1 = np.mean(np.mean(raster,axis=0)**2)
    term2 = np.mean(np.mean(raster,axis=0))**2
    #susceptability = num_rows * (term1 - term2)
    susceptibility = (num_rows/(num_rows-1)) * (term1 - term2)
    return susceptibility #, suscept2

##############################################################

# Firing activity per bin size
def rho(raster):
    print("Computing rho")
    rho=np.mean(np.mean(raster, axis=0))
    return rho

##############################################################

# Firing activity coefficient of variation 
def cv(raster):
    var = susceptibility(raster)
    rho=np.mean(np.mean(raster, axis=0))
    cv=math.sqrt(var)/rho
    return cv
##############################################################

# Priesemann's BR method
def calc_BR(A_t, k_max,ava_binsz,pltname):

	dt = ava_binsz*1000
	kmax = k_max

	src = mre. input_handler(A_t)
	rks = mre. coefficients(src , steps =(1, kmax) , dt=dt, dtunit='ms', method='trialseparated')

	fit1 = mre.fit(rks , fitfunc='complex')
	fit2 = mre.fit(rks , fitfunc='exp_offset')

	fig,ax1 = plt.subplots()
	ax1.plot(rks.steps, rks.coefficients, '.k', alpha = 0.2, label=r'Data')

	ax1.plot(rks.steps, mre.f_complex(rks.steps*dt, *fit1.popt), label='complex m={:.5f}'.format(fit1.mre))
	ax1.plot(rks.steps, mre.f_exponential_offset(rks.steps*dt, *fit2.popt), label='exp + offset m={:.5f}'.format(fit2.mre))

	ax1.set_xlabel(r'Time lag $\delta t$')
	ax1.set_ylabel(r'Autocorrelation $r_{\delta t}$')
	ax1.legend()
	plt.savefig(pltname)
	plt.close()

	fit_acc1 = sc.stats.pearsonr(rks.coefficients, mre.f_complex(rks.steps*dt, *fit1.popt))[0]
	fit_acc2 = sc.stats.pearsonr(rks.coefficients, mre.f_exponential_offset(rks.steps*dt, *fit2.popt))[0]

	return fit1.mre #, fit2.mre, fit_acc1, fit_acc2


##############################################################

def myround(n):
    return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))

def branchparam(TIMERASTER, lverbose=0):
    # myround = np.vectorize(lambda x: round(x))
    # Number of rows and columns
    r, c = TIMERASTER.shape
    print("Raster shape: ", TIMERASTER.shape)
    if lverbose:
        print(f'Number of rows and columns are: {r} and {c}, respectively')    # Initialize arrays
    descendants = np.zeros(r + 1)
    if lverbose:
        print(f'descendants {descendants}')
    prob = np.zeros(r + 1)
    if lverbose:
        print(f'prob {prob}')    # Convert non-zero elements to 1
    TIMERASTER[TIMERASTER != 0] = 1
    # print(f'nonzero TIMERASTER {np.nonzero(TIMERASTER)}')
    # print(f'nonzero {np.where(TIMERASTER != 0)[0]}')    # Find frames with at least one active site and
    # no sites active in the previous frame
    sums = np.sum(TIMERASTER, axis=0)
    if lverbose:
        print(f'sums {sums}')
    if lverbose:
        print(f'sums: {sums.shape}')    
    actives = np.nonzero(sums)[0]
    print(f"Active frames: {len(actives)} out of {len(sums)} total")
    print(f"Activity range: min={np.min(sums[sums>0]) if len(actives)>0 else 0}, max={np.max(sums) if len(actives)>0 else 0}")
    if lverbose:
        print(f'actives {actives}')
        print(f'len actives {len(actives)}')
    max_num = 0 
    for i in range(1, len(actives) - 1):
        ancestors = 0
        if lverbose:
            print(f'i {i} actives[i] {actives[i]} '
                f'actives[i]-1 {actives[i]-1} '
                f'sums(actives[i]-1) {sums[actives[i]-1]}')
        if sums[actives[i] - 1] == 0:
            ancestors = sums[actives[i]]
            if lverbose:
                print(f'i {i} ancestors {ancestors}')
            num = sums[actives[i] + 1]
            if lverbose:
                print(f'i {i} num {num}')
            # num = round(num / ancestors)
            num = myround(num / ancestors)
            # num = int(np.ceil(num / ancestors))
            if lverbose:
                print(f'i {i} num {num}')
            # descendants[num + 1] += ancestors
            descendants[num] = descendants[num] + ancestors
                # descendants[num] += ancestors
            if lverbose:
                print(f'i {i} descendants {descendants[num ]}')
            # print(f'i {i} ancestors {ancestors} num {num} descendants(num+1)
            max_num = max(max_num, num)

    if lverbose:
        print(f'sum ancestors: {np.sum(ancestors)}')
        print(f'sum descendants: {np.sum(descendants)}')
        print(f'descendants: {descendants}')
        print(f'num: {num}')    
    # Calculate the probability of each number of descendants
    sumd = np.sum(descendants)

    print(f"CRITICAL ASSIGNMENT:")
    print(f"  descendants array: size={len(descendants)} (shape: {descendants.shape})")
    print(f"  prob array: size={len(prob)} (shape: {prob.shape})")
    print(f"  Assignment: trying to assign {len(descendants)} elements to {len(prob)} elements")
    prob = descendants / sumd if sumd != 0 else np.zeros(r + 1)  
    if lverbose:
        print("Array assignment completed")   
    # Calculate the expected value
    # sig = np.sum((np.arange(r + 1) - 1) * prob)
    sig = 0.0

    print(f"CRITICAL LOOP:")
    print(f"  Loop range: i from 0 to {r} (inclusive) = {r+1} iterations")
    print(f"  prob array: size={len(prob)}, max_valid_index={len(prob)-1}")
    print(f"  Problem: loop will try prob[{r}] but max valid is prob[{len(prob)-1}]")
    if lverbose:
        print(f"CRITICAL: Final loop range(0,{r+1}) vs prob array size {len(prob)}")
    # for i in range(r + 1):
    # CHARLES -> for i in range(r):
    for i in range(r+1):
        sig = sig + ((i)*prob[i])
        if lverbose:
            print(f'i{i} prob(i){prob[i]} '
                  f'(i)*probi{(i)*prob[i]} sig{sig}')
    if lverbose:
        print(f'sig: {sig}')    
    return sig

##############################################################
# Function to find avalanches from raster data

def find_avalanches(array):
    activity_array = np.sum(array, axis=0)
    avalanches = []
    current_avalanche = []
    for activity in activity_array:
        if activity > 0:
            current_avalanche.append(activity)
        elif current_avalanche:
            avalanches.append(current_avalanche)
            current_avalanche = []
    if current_avalanche:  # Add the last avalanche if it exists
        avalanches.append(current_avalanche)
    return avalanches
#############################################################


#####################################################################

def rebinner(raster, bin_size):
    channels, timesteps = raster.shape

    # Calculating the number of new bins based on bin_size
    new_timesteps = int(np.ceil(timesteps / bin_size))

    # Preallocate the new raster matrix
    new_raster = csr_matrix((channels, new_timesteps), dtype=int)

    # Loop through each channel
    for i_channel in range(channels):
        # Extract the spike times for this channel
        spike_times = np.where(raster[i_channel, :])[0]

        # Rebin spike times
        new_spike_times = np.unique(np.ceil((spike_times + 1) / bin_size).astype(int) - 1)

        # Update the new raster matrix
        new_raster[i_channel, new_spike_times] = 1

    return new_raster


from copy import deepcopy as cdc
import time


##############################################################

def avgshapes(shapes, durations, method=None, args=()):
    # Determine sampling method
    target_indices = np.ones(len(durations), dtype=bool)
    
    if method is not None:
        if method == 'limits':
            lower_lim, upper_lim = args
            target_indices = (durations >= lower_lim) & (durations <= upper_lim)
        elif method == 'order':
            magnitude = args[0]
            if np.isscalar(magnitude):
                lower_lim = 10 ** magnitude
                upper_lim = 10 ** (magnitude + 1)
            else:
                lower_lim = 10 ** np.min(magnitude)
                upper_lim = 10 ** np.max(magnitude)
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (freqs >= lower_lim) & (freqs < upper_lim)
        elif method == 'linspace':
            lower_lim, upper_lim, n = args
            target_durations = np.round(np.linspace(lower_lim, upper_lim, n))
            target_indices = np.isin(durations, target_durations)
        elif method == 'logspace':
            x, lower_lim, upper_lim = args
            target_durations = x ** np.arange(lower_lim, upper_lim + 1)
            target_indices = np.isin(durations, target_durations)
        elif method == 'durations':
            target_durations = args[0]
            target_indices = np.isin(durations, target_durations)
        elif method == 'cutoffs':
            lower_lim, threshold = args
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (np.unique(durations) >= lower_lim) & (freqs >= threshold)

    # Compute average shapes
    sampled_shapes = [shape for shape, index in zip(shapes, target_indices) if index]
    sampled_durations = durations[target_indices]

    unique_durations = np.unique(sampled_durations)
    avg_profiles = []
    for dur in unique_durations:
        these_shapes = [shape for shape, d in zip(sampled_shapes, sampled_durations) if d == dur]
        avg_profiles.append(np.mean(these_shapes, axis=0))

    return avg_profiles

##############################################################

##############################################################
# Avalanche shape collapse

def avshapecollapse(shapes, durations, method=None, args=(),plot_flag=True, save_flag=False, filename=base_dir):
    
    if not shapes:
        print("[WARNING] No shapes provided to avshapecollapse")
        return None
    
    # Determine sampling method
    target_indices = np.ones(len(durations), dtype=bool)
    
    if method is not None:
        #uses avalanche shapes whose durations are inclusively bound by specified limits (scalar doubles)
        if method == 'limits': 
            lower_lim, upper_lim = args
            target_indices = (durations >= lower_lim) & (durations <= upper_lim)
        # uses avalanche shapes whose durations occur with frequency on the same order of magnitude
        # if magnitude is scalar or within the bounds of decades 10^(min(magnitude)) and 10^(max(magnitude)) if magnitude is a vector.
        elif method == 'order': 
            magnitude = args[0]
            if np.isscalar(magnitude):
                lower_lim = 10 ** magnitude
                upper_lim = 10 ** (magnitude + 1)
            else:
                lower_lim = 10 ** np.min(magnitude)
                upper_lim = 10 ** np.max(magnitude)
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (freqs >= lower_lim) & (freqs < upper_lim)
        # uses avalanche shapes of n different durations, linearly spaced between specified limits (scalar double)
        elif method == 'linspace':
            lower_lim, upper_lim, n = args
            target_durations = np.round(np.linspace(lower_lim, upper_lim, n))
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes whose durations are logarithmically spaced between x^(lowerLim) and x^(upperLim) (scalar doubles)
        elif method == 'logspace':
            x, lower_lim, upper_lim = args
            target_durations = x ** np.arange(lower_lim, upper_lim + 1)
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes of specific durations, durs (vector double)
        elif method == 'durations':
            target_durations = args[0]
            target_indices = np.isin(durations, target_durations)
        # uses avalanche shapes bounded below by both an absolute minimum duration (>= minDur) and a 
        # threshold for the frequency of occurrence (>= threshold) (scalar  doubles)
        elif method == 'cutoffs':
            lower_lim, threshold = args
            freqs = [np.sum(durations == dur) for dur in np.unique(durations)]
            target_indices = (np.unique(durations) >= lower_lim) & (freqs >= threshold)

    # Compute average shapes
    sampled_shapes = [shape for shape, index in zip(shapes, target_indices) if index]
    sampled_durations = durations[target_indices]

    if not sampled_shapes:
        print("[WARNING] No shapes remain after filtering in avshapecollapse")
        return {
            'exponent': None,
            'secondDer': None,
            'range': [],
            'errors': [],
            'coefficients': None
        }
    
    unique_durations = np.unique(sampled_durations)
    avg_shapes = []
    for dur in unique_durations:
        these_shapes = [shape for shape, d in zip(sampled_shapes, sampled_durations) if d == dur]
        avg_shapes.append(np.mean(these_shapes, axis=0))
    
    if not avg_shapes:
        print("[WARNING] No average shapes could be computed in avshapecollapse")
        return {
            'exponent': None,
            'secondDer': None,
            'range': [],
            'errors': [],
            'coefficients': None
        }
    
    ##############################
    precision=1e-3
    n_interp_points=1000
    bounds=(0, 3)
    n_avs = len(avg_shapes)
    max_duration = max([len(shape) for shape in avg_shapes])
    
    # Scale durations by duration length (t/T)
    scaled_durs = [(np.arange(1, len(shape) + 1) / len(shape)) for shape in avg_shapes]

    # Continually refine exponent value range to find optimal 1/(sigma nu z)
    n_iterations = int(-np.log10(precision))
    errors = []
    ranges = []
    
    for i in range(n_iterations):
        exponent_range = np.arange(bounds[0], bounds[1], 10 ** (-i - 1))
        errors_iteration = []
        for exponent in exponent_range:
            # Scale shapes by T^{1 - 1/(sigma nu z)}
            scaled_shapes = [shape * (len(shape) ** (1 - exponent)) for shape in avg_shapes]
            # Interpolate shapes to match maximum duration length
            interp_shapes = [np.interp(np.linspace(0, 1, n_interp_points), scaled_dur, scaled_shape) for scaled_shape, scaled_dur in zip(scaled_shapes, scaled_durs)]
            # Compute error of all shape collapses
            error = np.mean(np.var(interp_shapes, axis=0)) / ((np.max(interp_shapes) - np.min(interp_shapes)) ** 2)
            errors_iteration.append(error)
        errors.append(errors_iteration)
        ranges.append(exponent_range)
        # Find exponent value that minimizes error
        best_index = np.argmin(errors_iteration)
        sigma_nu_z_inv = exponent_range[best_index]
        # Generate new range of exponents to finer precision
        if i < n_iterations - 1:
            bounds = (sigma_nu_z_inv - 10 ** (-i - 1), sigma_nu_z_inv + 10 ** (-i - 1))
    
    # Fit to 2nd degree polynomial and find second derivative
    best_shapes = [shape * (len(shape) ** (1 - sigma_nu_z_inv)) for shape in avg_shapes]
    interp_best_shapes = [np.interp(np.linspace(0, 1, n_interp_points), scaled_dur, scaled_shape) for scaled_shape, scaled_dur in zip(best_shapes, scaled_durs)]
    avg_scaled_shape = np.mean(interp_best_shapes, axis=0)
    coeffs = np.polyfit(np.linspace(0, 1, n_interp_points), avg_scaled_shape, 2)
    second_drv = 2 * coeffs[0]
    
    # Plot and save
    if plot_flag or save_flag:
        plt.figure(figsize=(6, 6))
        # Plot all shapes
        for shape in interp_best_shapes:
            plt.plot(np.linspace(0, 1, n_interp_points), shape, alpha=0.5)
        # Overlay polynomial fit
        exponent_label = f'Exponent fit = {sigma_nu_z_inv:.2f}' if sigma_nu_z_inv is not None else 'Exponent not computed'
        plt.plot(np.linspace(0, 1, n_interp_points), np.polyval(coeffs, np.linspace(0, 1, n_interp_points)), '-r', linewidth=3, label=exponent_label)
    
        # Label axes
        plt.xlabel('Scaled Avalanche Duration (t/T)', fontsize=14)
        plt.ylabel('Scaled Avalanche Shapes', fontsize=14)
        # Set title and legend
        plt.title('Avalanche Shape Collapse', fontsize=14)
        #plt.legend(['Scaled shape', 'Polynomial fit'])
        plt.legend()
        # Save figure if required
        if save_flag:
            fname = filename + '_shapes'
            fig_name = f'{fname}.pdf'
            plt.savefig(fig_name)
            if not plot_flag:
                plt.close()
        else:
            fig_name = None
        plt.show()
    else:
        fig_name = None
        
    print('exponent 1/(sigma nu z):', sigma_nu_z_inv)
    
    Result = {
        'exponent': sigma_nu_z_inv,
        'secondDer': second_drv,
        'range': ranges,
        'errors': errors,
        'coefficients': coeffs
    }
    
    return Result

##############################################################

##############################################################
# Function to compute branching ratios, the naive way

def branching_ratios(shapes):
    branching_ratios = []

    for shape in shapes:
        # Skip avalanches with a duration of 1
        if len(shape) <= 1:
            continue
        # Calculate the branching ratio for each step in the avalanche,
        # skipping steps where the ancestor count is zero to avoid division by zero
        ratios = [shape[i] / shape[i - 1] for i in range(1, len(shape)) if shape[i - 1] != 0]
        # Calculate the average branching ratio for the avalanche
        if ratios:
            avg_ratio = np.mean(ratios)
            branching_ratios.append(avg_ratio)

    # Check if we have any valid branching ratios
    if len(branching_ratios) == 0:
        print("[WARNING] No valid branching ratios calculated - no avalanches with duration > 1")
        total_avg_branching_ratios = np.nan
        std_error = np.nan
    else:
        # Calculate the total average branching ratio and its standard error
        total_avg_branching_ratios = np.mean(branching_ratios)
        std_error = np.std(branching_ratios) / np.sqrt(len(branching_ratios))
    
    Result = {
    'BR': branching_ratios,
    'avgBR': total_avg_branching_ratios,
    'BRstd': std_error
    }
    return  Result