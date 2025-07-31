import os
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat
import scipy.optimize
import scipy.stats as statistics
import seaborn as sns
import pickle
import os
import os.path as op
import sys; sys.path.append('.')
from tempfile import TemporaryFile
import tables
from copy import deepcopy as cdc
import time
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse

#import torch

def variations_copy(raster):
    """
    Calculate average activity (Actavg), LTF (LTFavg), and susceptibility from a raster matrix.
    
    Parameters:
        raster (ndarray): Binary raster matrix of shape (N_neurons, T_timesteps)
        
    Returns:
        Actavg (float): Mean activity
        LTFavg (float): Long-Term Fluctuation (Coefficient of Variation)
        susceptibility (float): Susceptibility
    """
    N, T = raster.shape

    # Calculate susceptibility
    susceptibility = N * (
        (1 / (T * N**2)) * np.sum(np.sum(raster, axis=0)**2) -
        (np.count_nonzero(raster) / (N * T))**2
    )

    # Mean activity
    Actavg = np.count_nonzero(raster) / (N * T)

    # LTFavg as coefficient of variation
    LTFavg = (1 / Actavg) * np.sqrt(susceptibility / N)

    return Actavg, LTFavg, susceptibility



def max_eigenvalue_power_iteration(A, max_iter=1000, tol=1e-6):
    # Ensure the matrix is on the same device as the operations (CPU or GPU)
    device = A.device
    n = A.shape[0]
    
    # Start with a random vector
    b = torch.randn(n, device=device)
    b = b / torch.norm(b)

    eigenvalue = 0
    for _ in range(max_iter):
        # Matrix-vector product
        Ab = torch.mv(A, b)
        new_eigenvalue = torch.norm(Ab)
        b_new = Ab / new_eigenvalue

        # Convergence check
        if torch.abs(new_eigenvalue - eigenvalue) < tol:
            break
        b = b_new
        eigenvalue = new_eigenvalue

    return eigenvalue.item()

def fast_correlation_matrix(raster, T_adjusted, lag):
    N, T = raster.shape
    series_i = raster[:, :T_adjusted]  # Extract the first series for all i
    series_j = raster[:, lag:(lag + T_adjusted)]  # Extract the lagged series for all j

    # Compute mask for valid rows
    mask_i = np.any(series_i, axis=1)
    mask_j = np.any(series_j, axis=1)
    valid_mask = np.outer(mask_i, mask_j)

    # Normalize series
    series_i = (series_i - np.mean(series_i, axis=1, keepdims=True)) / np.std(series_i, axis=1, keepdims=True)
    series_j = (series_j - np.mean(series_j, axis=1, keepdims=True)) / np.std(series_j, axis=1, keepdims=True)

    # Compute the dot product for correlation
    corr_matrix = (series_i @ series_j.T) / T_adjusted
    corr_matrix[~valid_mask] = 0  # Set invalid entries to 0
    return corr_matrix

def compute_correlations_copy(raster, lag=1):
    """
    Compute correlation matrix, max eigenvalue, and exponential fit parameters for a raster matrix.
    
    Parameters:
        raster (ndarray): Binary raster matrix of shape (N_neurons, T_timesteps)
        lag (int): Time lag for correlation computation (default=1)
        
    Returns:
        corr_matrix (ndarray): NxN correlation matrix
        max_eigenvalue (float): Maximum eigenvalue of the correlation matrix
        exp_params (list): Parameters of exponential fit [a, b, c]
    """
    print("Constructing matrix...")
    N, T = raster.shape
    T_adjusted = T - lag
    corr_matrix = np.zeros((N, N))
    corr_matrix = fast_correlation_matrix(raster, T_adjusted, lag)
    
    # for i in range(N):
    #      for j in range(N):
    #          series_i = raster[i, :T_adjusted]
    #          series_j = raster[j, lag:(lag + T_adjusted)]
    #          if np.any(series_i) and np.any(series_j):
    #              corr_coef = np.corrcoef(series_i, series_j)[0, 1]
    #              corr_matrix[i, j] = corr_coef
    #          else:
    #              corr_matrix[i, j] = 0
    print("Constructed... now computing eigenvalue")
    # Compute eigenvalues and maximum eigenvalue
    eigenvalues = np.linalg.eigvals(corr_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    # 1-> max_eigenvalue = eigsh(corr_matrix, k=1, which='LM', return_eigenvectors=False)[0]
    # 3-> max_eigenvalue = max_eigenvalue_power_iteration(corr_matrix)

    # Exponential fit
    sorted_weights = np.sort(np.abs(corr_matrix), axis=1)[:, ::-1]
    avg_sorted_weights = np.mean(sorted_weights, axis=0)
    
    x_data = np.arange(len(avg_sorted_weights))
    
    def exp_fit(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the exponential curve
    popt, _ = curve_fit(exp_fit, x_data, avg_sorted_weights, p0=[1, 0.1, 0])

    exp_params = popt.tolist()  # [a, b, c]
    
    return corr_matrix, max_eigenvalue, exp_params

def branchparam(TIMERASTER):
    """
    Computes the branching parameter, sigma, from a raster matrix.

    Parameters:
        TIMERASTER (ndarray): A binary 2D NumPy array (raster matrix) where
                              rows represent neurons and columns represent time.

    Returns:
        sig (float): The branching parameter.
    """
    # Ensure binary raster (0 or 1)
    TIMERASTER = np.where(TIMERASTER != 0, 1, 0)

    # Initialize variables
    r, c = TIMERASTER.shape
    print("raster shape", TIMERASTER.shape)
    descendants = np.zeros(r + 1)
    prob = np.zeros(r)

    # Sum across rows to find active frames
    sums = np.sum(TIMERASTER, axis=0)
    actives = np.where(sums != 0)[0]

    # Loop through active frames to calculate descendants
    for i in range(1, len(actives) - 1):
        ancestors = 0
        if sums[actives[i] - 1] == 0:  # Check if previous frame is inactive
            print("Found inactive frame:", i)
            ancestors = sums[actives[i]]
            num = sums[actives[i] + 1]  # Count descendants in the next frame
            if ancestors > 0:
                num = round(num / ancestors)
            descendants[num] += ancestors  # Increment descendants counter

    # Calculate probabilities
    sum_descendants = np.sum(descendants)
    if sum_descendants > 0:
        prob = descendants / sum_descendants
        print("prob: ", prob)

    # Calculate the expected value (branching parameter)
    sig = np.sum((np.arange(len(prob))) * prob)

    return sig

def branchparam_improved(TIMERASTER):
    """
    Computes the branching parameter, sigma, from a raster matrix.

    Parameters:
        TIMERASTER (ndarray): A binary 2D NumPy array (raster matrix) where
                              rows represent neurons and columns represent time.

    Returns:
        sig (float): The branching parameter.
    """
    # Ensure binary raster (0 or 1)
    TIMERASTER = np.where(TIMERASTER != 0, 1, 0)

    # Initialize variables
    r, c = TIMERASTER.shape
    
    # Sum across rows to find number of active neurons per frame
    sums = np.sum(TIMERASTER, axis=0)
    
    # Calculate ratio of successive active frames
    ratios = []
    for t in range(len(sums)-1):
        if sums[t] > 0:  # If there are any active neurons
            ratio = sums[t+1] / sums[t]  # Calculate descendant/ancestor ratio
            ratios.append(ratio)
    
    # Branching parameter is the average ratio
    sig = np.mean(ratios) if ratios else 0
    
    return sig

def branchparam_copy(TIMERASTER):
    """
    Computes branching parameter accounting for super-critical dynamics.
    """
    TIMERASTER = np.where(TIMERASTER != 0, 1, 0)
    r, c = TIMERASTER.shape
    sums = np.sum(TIMERASTER, axis=0)
    
    ratios = []
    for t in range(len(sums)-1):
        if sums[t] > 0 and sums[t] < r*0.9:  # Only use unsaturated states
            ratio = sums[t+1] / sums[t]
            ratios.append(ratio)
    
    if not ratios:
        return float('inf')  # System is likely extremely super-critical
        
    sig = np.mean(ratios)
    return sig


def avprops4ras3(raster_struct, *args):
    """
    Computes avalanche properties from raster data.
    
    Parameters:
        raster_struct: dict or ndarray
            A structure containing a 'raster' field with the raster data or
            the raster data itself (sparse or dense ndarray).
        *args: optional arguments
            'ratio' to compute branching ratio.
            'fingerprint' to compute avalanche fingerprint.

    Returns:
        Avalanche: dict
            A dictionary containing avalanche properties:
            - 'duration': durations of avalanches.
            - 'size': sizes of avalanches.
            - 'shape': shapes of avalanches.
            - 'fingerPrint': (optional) avalanche fingerprints.
            - 'branchingRatio': (optional) branching ratios.
    """
    # Initialize flags
    Flag = {'branchingRatio': False, 'fingerPrint': False}

    # Parse optional arguments
    for arg in args:
        if isinstance(arg, str):
            if arg == 'ratio':
                Flag['branchingRatio'] = True
            elif arg == 'fingerprint':
                Flag['fingerPrint'] = True
            else:
                print(f"(AVPROPS) Ignoring invalid argument: {arg}")
    
    # Extract raster
    if isinstance(raster_struct, dict) and 'raster' in raster_struct:
        raster = raster_struct['raster']
    else:
        raster = raster_struct

    # Get all event times and sites
    if isinstance(raster, np.ndarray) and not issparse(raster):
        allTimes, allSites = np.nonzero(raster)
    else:
        allSites, allTimes = raster.nonzero()
    
    allTimes = np.asarray(allTimes).flatten()
    allSites = np.asarray(allSites).flatten()

    if len(allTimes) == 0:
        raise ValueError("Empty raster; no avalanches detected.")

    # Sort events chronologically
    sorted_indices = np.argsort(allTimes)
    allTimes = allTimes[sorted_indices]
    allSites = allSites[sorted_indices]

    # Calculate time differences and avalanche boundaries
    diffTimes = np.diff(allTimes, prepend=allTimes[0])
    diffTimes[diffTimes == 1] = 0  # Consecutive timesteps = same avalanche
    avBoundaries = np.where(diffTimes > 0)[0]
    avBoundaries = np.append(avBoundaries, len(allTimes))  # Final boundary

    # Initialize avalanche properties
    nAvs = len(avBoundaries)
    Avalanche = {
        'duration': np.zeros(nAvs, dtype=int),
        'size': np.zeros(nAvs, dtype=int),
        'shape': [None] * nAvs
    }

    if Flag['fingerPrint']:
        Avalanche['fingerPrint'] = {}

    if Flag['branchingRatio']:
        Avalanche['branchingRatio'] = np.zeros(nAvs)

    # Process each avalanche
    avStart = 0
    for iAv in range(nAvs):
        avEnd = avBoundaries[iAv]
        thisAvTimes = allTimes[avStart:avEnd]
        
        # Compute avalanche duration and size
        Avalanche['duration'][iAv] = len(np.unique(thisAvTimes))
        Avalanche['size'][iAv] = len(thisAvTimes)
        
        # Compute shape
        unique_times = np.unique(thisAvTimes)
        Avalanche['shape'][iAv] = np.histogram(thisAvTimes, bins=np.append(unique_times, unique_times[-1] + 1))[0]
        
        # Fingerprint computation
        if Flag['fingerPrint']:
            size_key = Avalanche['size'][iAv]
            if size_key not in Avalanche['fingerPrint']:
                Avalanche['fingerPrint'][size_key] = []
            Avalanche['fingerPrint'][size_key].append(np.vstack([allTimes[avStart:avEnd], allSites[avStart:avEnd]]))
        
        # Branching ratio computation
        if Flag['branchingRatio']:
            shape = Avalanche['shape'][iAv]
            Avalanche['branchingRatio'][iAv] = np.sum(shape[1:] / shape[:-1]) / Avalanche['duration'][iAv]

        # Move to next avalanche
        avStart = avEnd

    return Avalanche

def process_raster(data_raster, start_time=None, end_time=None):
    # Convert from sparse to dense matrix if necessary
    if hasattr(data_raster, 'toarray'):
        tmp_p_raster = data_raster.toarray()
    else:
        tmp_p_raster = data_raster
    
    # Print shape (equivalent to size in MATLAB)
    print(tmp_p_raster.shape)
    
    tmp_raster = tmp_p_raster
    
    # Calculate average activity
    avgActivity = np.mean(np.sum(tmp_raster > 0, axis=0))
    print(f"Average activity: {avgActivity}")
    
    # Create binary raster
    binaryRaster = tmp_raster > 0
    
    # Calculate activity per time point
    activityPerTimePoint = np.sum(binaryRaster, axis=0)
    
    # Create mask for above average activity
    aboveAverageMask = activityPerTimePoint >= avgActivity/2
    
    # Create full mask
    fullMask = np.broadcast_to(aboveAverageMask, tmp_raster.shape)
    
    # Apply mask to raster
    raster = tmp_raster * fullMask
    #raster = tmp_raster
    
    # Handle time range subsetting
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = raster.shape[1]
        
    # Ensure indices are within bounds
    start_time = max(0, start_time)
    end_time = min(raster.shape[1], end_time)
    
    # Return the time-ranged subset of the raster
    return raster[:, start_time:end_time]




# 1. read files
try:
    experiment_tag = sys.argv[1]
except:
    raise ValueError('Please specify a valid filename.')

print('Loading file:', experiment_tag, '...')

# Additional visualization of the population activity
starting_time_point = 2500000  # Adjust as needed
h5 = tables.open_file(sys.argv[1],'r')
data = h5.root
try:
    # Debug prints to understand the structure
    print("File structure:")
    h5.list_nodes('/')

    # Try different methods to read the data
    # Method 1: Using iter_nodes
    for node in h5.iter_nodes(h5.root.c, 'Array'):
        print(node.name)
        if node.name == 'logfilepath':
            pickle_dir = str(node[0])
            break
    # Method 2: Print children using _v_children
    print("\nAll attributes:")
    print(dir(data))

    if data.__contains__('Spikes'):
        print("Looking at raster in file: ", sys.argv[1])
        # raster plot (last_n_spikes)
        last_spikes = data.c.stats.only_last_spikes[0]
        tmp_p_raster = data.Spikes[0, :, -last_spikes:]
        raster = process_raster(tmp_p_raster, starting_time_point)

finally:
    h5.close()

print(f"Data loaded\nNumber of neurons: {raster.shape[0]}\nNumber of time points: {raster.shape[1]}")

# Initialize arrays
all_sizes, all_durations = [], []
all_firing_rates = []
avgAct, susc, cv, eval1, exp1, br = [], [], [], [], [], []

    
# Average firing rate
num_neurons, num_timesteps = raster.shape
neuron_firing_rates = np.sum(raster, axis=1) / num_timesteps
avg_firing_rate = np.mean(neuron_firing_rates)
all_firing_rates.append(avg_firing_rate)
    
# Additional metrics
print("variations_copy")
rho, ltf, chi = variations_copy(raster)
print("compute_correlations_copy")
_, evals, corr_params = compute_correlations_copy(raster, 1)
exps = corr_params[0]
print("branchparam_copy")
sigma = branchparam_copy(raster)
print("Now plotting...")
    
avgAct.append(rho)
print("rho")
print(rho)
susc.append(chi)
print("chi")
print(chi)
cv.append(ltf)
print("cv")
print(ltf)
eval1.append(evals)
print("pearson")
print(evals)
exp1.append(exps)
print("exponennts")
print(exps)
br.append(sigma)
print("branching")
print(sigma)
    
# Avalanche properties
Avalanche = avprops4ras3(raster)
all_sizes.extend(Avalanche.get('size'))
all_durations.extend(Avalanche.get('duration'))
    
# Standard error calculations
sem = lambda x: np.std(x) / np.sqrt(len(x))
sem_avgAct, sem_susc, sem_cv = sem(avgAct), sem(susc), sem(cv)
sem_eval1, sem_exp1, sem_br = sem(eval1), sem(exp1), sem(br)

# # Visualization
# plt.figure(figsize=(12, 12))

# # Error bar plots for new metrics
# metrics = [(avgAct, sem_avgAct, r'$\rho$'), (susc, sem_susc, r'$\chi$'),
#            (cv, sem_cv, 'CV'), (eval1, sem_eval1, r'Pearson $\kappa$'),
#            (exp1, sem_exp1, r'$\beta$'), (br, sem_br, 'Branching Ratio')]
# titles = ['Average Activity', 'Susceptibility', 'Coefficient of Variation',
#           'Pearson Correlation', 'Exponent', 'Branching Ratio']

# for idx, (metric, sem_metric, ylabel) in enumerate(metrics, start=1):
#     plt.subplot(2, 3, idx)
#     plt.errorbar(kappas, np.mean(metric), yerr=sem_metric, fmt='ro-')
#     plt.xlabel(r'$\kappa$')
#     plt.ylabel(ylabel)
#     plt.title(titles[idx - 1])

# # Saving the plots
# plt.tight_layout()
# plt.savefig('combined_metrics_plots.png')
# plt.show()

# Save data
# np.savez('processed_data_complete.npz', all_sizes=all_sizes, all_durations=all_durations,
#          all_N=all_N, all_kin=all_kin, all_ps=all_ps, all_k=all_k, all_ref=all_ref,
#          all_B=all_B, all_run=all_run, all_firing_rates=all_firing_rates,
#          avgAct=avgAct, susc=susc, cv=cv, eval1=eval1, exp1=exp1, br=br,
#          sem_avgAct=sem_avgAct, sem_susc=sem_susc, sem_cv=sem_cv,
#          sem_eval1=sem_eval1, sem_exp1=sem_exp1, sem_br=sem_br)
