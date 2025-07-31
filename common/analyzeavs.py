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

import os
import tables
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

def test_hdf5_file(filepath):
    """Test if HDF5 file is readable"""
    try:
        h5 = tables.open_file(filepath, 'r')
        h5.close()
        return True
    except Exception as e:
        print(f"Error while opening file {filepath}: {e}")
        return False

def get_backup_folders():
    """Get list of timestamped backup folders with valid HDF5 files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    backup_root = os.path.join(root_dir, 'SORN', 'sigma0.05h_ip0.4BASELINE2_5M', 'test_single')
    folders = sorted([d for d in os.listdir(backup_root) if d.startswith('202')])
    
    # Test and filter for valid folders
    valid_folders = []
    print("Testing backup folders...")
    for folder in folders:
        result_path = os.path.join(backup_root, folder, 'common', 'result.h5')
        if os.path.exists(result_path) and test_hdf5_file(result_path):
            print(f"Valid backup found: {folder}")
            valid_folders.append(folder)
    
    print(f"Found {len(valid_folders)} valid backup folders")
    return valid_folders

def get_result_path(base_num):
    """Get path to result.h5 from a specific backup folder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    backup_folders = get_backup_folders()
    if base_num > len(backup_folders):
        raise ValueError("Not enough backup folders available")
    folder = backup_folders[-base_num]  # Use newest folders first
    path = os.path.join(root_dir, 'SORN', 'sigma0.05h_ip0.4BASELINE2_5M', 'test_single', folder, 'common', 'result.h5')
    print(f"Using backup from: {folder}")
    return path

### Files to run
number_of_files = 10

### figure parameters
width  =  8
height = width / 1.718  # Fix this to the golden ratio
fig = plt.figure(1, figsize=(width, height))
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
stable_steps = int(3e6)  # Convert to integer # steps to use: after transient (2e6)
THETA = 'half'

# count files
print('Loading experiment files...')
exper_path = ''

# load the data in 'data_all'
data_all = np.zeros((number_of_files, stable_steps))
for result_file in range(number_of_files):
    result_path = get_result_path(result_file + 1)
    print(f"Loading: {result_path}")
    h5 = tables.open_file(result_path, 'r')
    data = h5.root
    data_all[result_file] = np.around(data.activity[0][-stable_steps:] * data.c.N_e)
    N_e_val = data.c.N_e[...]
    h5.close()

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
    # Susceptibility calculation
    N, T = raster.shape
    susceptibility = N * (
        (1 / (T * N**2)) * np.sum(np.sum(raster, axis=0)**2) - 
        (np.count_nonzero(raster) / (N * T))**2
    )

    # Mean activity calculation
    Actavg = np.count_nonzero(raster) / (N * T)

    # LTFavg as coefficient of variation
    LTFavg = (1 / Actavg) * np.sqrt(susceptibility / N)

    return Actavg, LTFavg, susceptibility

# Now compute the variations for `data_all`
avgAct_all, susc_all, cv_all = [], [], []

for result_file in range(number_of_files):
    raster = data_all[result_file]  # Get the raster for each result file
      # Debugging: Check shape of raster before calling variations_copy
    print(f"Processing variations for file {result_file + 1}...")
    print(f"Shape of raster: {raster.shape}")
    
    if len(raster.shape) == 1:  # If raster is 1D, reshape to 2D
        raster = raster.reshape(1, -1)  # Reshape to a 2D array with 1 row

    # Calculate the variations for each raster (i.e., each file's data)
    print(f"Processing variations for file {result_file + 1}...")
    rho, ltf, chi = variations_copy(raster)
    
    avgAct_all.append(rho)
    susc_all.append(chi)
    cv_all.append(ltf)

# Standard error calculations
sem = lambda x: np.std(x) / np.sqrt(len(x))
sem_avgAct, sem_susc, sem_cv = sem(avgAct_all), sem(susc_all), sem(cv_all)

# Additional print/debugging to check the values
print("Average Activities:", avgAct_all)
print("Susceptibility:", susc_all)
print("CV:", cv_all)

#compulte_correlations_copy

#updated method of creating matrix
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

#rest is the same

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
    
    #old compute_correlation matrix building
    #
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

    # Exponential fit
    sorted_weights = np.sort(np.abs(corr_matrix), axis=1)[:, ::-1]
    avg_sorted_weights = np.mean(sorted_weights, axis=0)
    
    x_data = np.arange(len(avg_sorted_weights))

    def exp_fit(x, a, b, c):
        return a * np.exp(-b * x) + c  # Return an array, not a single value

    # Fit the exponential curve
    popt, _ = curve_fit(exp_fit, x_data, avg_sorted_weights, p0=[1, 0.1, 0])

    exp_params = popt.tolist()  # [a, b, c]
    
    return corr_matrix, max_eigenvalue, exp_params

#branchparam_copy

def branchparam_copy(TIMERASTER):
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
            ancestors = sums[actives[i]]
            num = sums[actives[i] + 1]  # Count descendants in the next frame
            if ancestors > 0:
                num = round(num / ancestors)
            descendants[num] += ancestors  # Increment descendants counter

    # Calculate probabilities
    sum_descendants = np.sum(descendants)
    if sum_descendants > 0:
        prob = descendants / sum_descendants

    # Calculate the expected value (branching parameter)
    sig = np.sum((np.arange(len(prob))) * prob)

    return sig



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




# # 1. read files
# try:
#     experiment_tag = sys.argv[1]
# except:
#     raise ValueError('Please specify a valid filename.')

# print('Loading file:', experiment_tag, '...')

# # Additional visualization of the population activity
# starting_time_point = 2500000  # Adjust as needed
# h5 = tables.open_file(sys.argv[1],'r')
# data = h5.root
# try:
#     # Debug prints to understand the structure
#     print("File structure:")
#     h5.list_nodes('/')

#     # Try different methods to read the data
#     # Method 1: Using iter_nodes
#     for node in h5.iter_nodes(h5.root.c, 'Array'):
#         if node.name == 'logfilepath':
#             pickle_dir = str(node[0])
#             break

#     if data.__contains__('Spikes'):
#         print("Looking at raster in file: ", sys.argv[1])
#         # raster plot (last_n_spikes)
#         last_spikes = data.c.stats.only_last_spikes[0]
#         tmp_p_raster = data.Spikes[0, :, -last_spikes:]
#         raster = process_raster(tmp_p_raster, starting_time_point)

# finally:
#     h5.close()

# print(f"Data loaded\nNumber of neurons: {raster.shape[0]}\nNumber of time points: {raster.shape[1]}")

# Initialize arrays
avgAct, susc, cv, eval1, exp1, br = [], [], [], [], [], []
    
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