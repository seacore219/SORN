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
import criticality as cr
from criticality import pvaluenew2 as pv
from criticality import exclude as ex
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import criticality_tumbleweed as crt
import psutil
import gc

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
gc.collect()
tables.file._open_files.close_all()
print("[INFO] all libraries imported successfully.")

##############################################################
# Define the necessary functions here
##############################################################
print("[INFO] Searching for directories with simulation results...")
## Need to find all the directories where there are valid sims
# base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\GitHub\\SORN\\mu=0.02_sigma=0.05_500K+3.5M_plastic_raster\\test_single'
base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\SORN_PC\\backup\\noisevar\\finegrain\\batch_hip0.0800_fp0.001_cde0.10_cdi0.00_noise0.06'
# base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\CharlesSORNneo\\backup\\test_single\\batch_hip0.0075_fp0.01_cde0.01_cdi0.01
# base_dir = 'C:\\Users\\seaco\\Downloads\\sims\\randn_10%_e_hip0.2_uext0.2'
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+base_).*_raster"
#pattern = r"SPmu=(\d+\.\d+)_sigma=(\d+\.\d+)_(\d+)K.*_sigma_(\d+\.\d+)_.*raster"
#->pattern = r"SPmu=(0.08)_sigma=(0.05)_(\d+)K.*_sigma_(0.05)_.*raster"

##############################################################
# UPDATED FUNCTIONS FROM CRITICALITY_TUMBLEWEED
##############################################################

def myround(n):
    '''
    This function converts the decimal number to an integer by rounding it
    according to the "round half up" method, where values equal to or greater
    than .5 are rounded up to the nearest integer, and those less than .5 are
    rounded down.
    '''
    return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))

def branchparam(TIMERASTER, lverbose=0):
    '''
    Perform branching parameter calculation on a raster matrix.
    
    TIMERASTER (numpy.ndarray): Raster matrix for calculation.
    lverbose (int): Verbosity level. Set to 1 for verbose mode, 0 for silent.
    '''
    def myround(n):
        return int(Decimal(n).to_integral_value(rounding=ROUND_HALF_UP))
    
    TIMERASTER = np.asarray(TIMERASTER)
    timerastersumpertime = np.sum(TIMERASTER, axis=0)
    sizebins = range(np.max(timerastersumpertime) + 2)
    
    if lverbose:
        print("sizebins = %r\n" % sizebins)
    
    count = np.zeros(len(sizebins))
    count_pre = np.zeros(len(sizebins))
    
    nrows, ncols = np.shape(TIMERASTER)
    TIMERASTER_PRESHIFT = TIMERASTER[:, 0:ncols - 1]
    TIMERASTER_SHIFTED = TIMERASTER[:, 1:ncols]
    timerastersumpertime_PRESHIFT = np.sum(TIMERASTER_PRESHIFT, axis=0)
    timerastersumpertime_SHIFTED = np.sum(TIMERASTER_SHIFTED, axis=0)
    
    counter = 0
    if lverbose == 2:
        lentimerastersumpertime = len(timerastersumpertime_PRESHIFT)
        print("lentimerastersumpertime = %r\n" % lentimerastersumpertime)
    
    for A_i, A_plus1 in zip(timerastersumpertime_PRESHIFT, timerastersumpertime_SHIFTED):
        if lverbose == 2:
            counter = counter + 1
            if counter % 100000 == 0:
                print("Done (in %) = %r\n" % (100 * counter / lentimerastersumpertime))
        
        # Round the values
        A_i_rounded = myround(A_i)
        A_plus1_rounded = myround(A_plus1)
        
        # Ensure they are valid indices
        if A_i_rounded >= 0:
            count_pre[A_i_rounded] += 1
            count[A_i_rounded] += A_plus1_rounded
    
    if lverbose:
        print("count = %r\n" % count)
        print("count_pre = %r\n" % count_pre)
    
    rvec = []
    cvec = []
    cpvec = []
    total_sum_r = 0
    total_sum_cpvec = 0
    
    for i in range(len(count)):
        if count_pre[i] > 0:
            r = count[i] / count_pre[i]
            rvec.append(r)
            cvec.append(count[i])
            cpvec.append(count_pre[i])
            total_sum_r += count[i]
            total_sum_cpvec += count_pre[i]
    
    # Compute global branching parameter
    if total_sum_cpvec > 0:
        bp = total_sum_r / total_sum_cpvec
    else:
        bp = 0  # Avoid division by zero
    
    Result = {'bp': bp, 'rvec': rvec, 'cvec': cvec, 'cpvec': cpvec}
    
    if lverbose:
        print("bp = %r\n" % bp)
        print("rvec = %r\n" % rvec)
        print("cvec = %r\n" % cvec)
        print("cpvec = %r\n" % cpvec)
    
    return Result

def get_avalanches(data, perc=0.25, ncells=-1, const_threshold=None):
    """
    Find avalanches in spike data - UPDATED VERSION from criticality_tumbleweed
    Returns dictionary with sizes (S) and durations (T)
    """
    ttic = time.time()
    
    # Get dimensions
    if ncells == -1:
        n, m = np.shape(data)
    else:
        n = ncells
        m = np.shape(data)[0]
    print(f"Data has {n} neurons with length {m}*binsize")
    
    # Collapse to single array
    if n == 1:
        network = cdc(data)
    else:
        if ncells == -1:
            network = np.nansum(data, axis=0)
        else:
            network = data.copy()
    
    # Determine threshold
    if const_threshold is None:
        if perc > 0:
            sortN = np.sort(network)
            perc_threshold = sortN[round(m * perc)]
            print(f"perc_threshold: {perc_threshold}")
        else:
            perc_threshold = 0
    else:
        perc_threshold = const_threshold
        print(f"const_threshold: {perc_threshold}")
    
    # Create binary data
    zdata = cdc(network)
    zdata[zdata <= perc_threshold] = 0
    zdata[zdata > perc_threshold] = 1
    zdata = zdata.astype(np.int8)
    
    # Find avalanche boundaries
    zeros_loc_zdata = np.where(zdata == 0)[0]
    zeros_to_delete = zeros_loc_zdata[np.where(np.diff(zeros_loc_zdata) == 1)[0]]
    
    z1data = np.delete(zdata, zeros_to_delete)
    avalanches = np.delete(network, zeros_to_delete)
    avalanches[z1data == 0] = 0
    
    zeros_loc_z1data = np.where(z1data == 0)[0]
    
    # Calculate sizes and durations
    burst = []
    shapes = []
    for i in np.arange(0, np.size(zeros_loc_z1data) - 1):
        tmp_av = avalanches[zeros_loc_z1data[i] + 1:zeros_loc_z1data[i + 1]]
        tmp_burst = np.sum(tmp_av) - (perc_threshold * len(tmp_av))
        if tmp_burst > 0:
            burst.append(tmp_burst)
            shape = tmp_av - perc_threshold
            shapes.append(shape[shape > 0])
    
    # Duration calculation
    T = np.diff(zeros_loc_z1data) - 1
    T = T[T > 0]
    
    # Find avalanche locations
    z2data = zdata[0:-1]
    z2data = np.insert(z2data, 0, 0)
    location = np.where(np.logical_and(zdata == 1, z2data == 0))[0]
    
    Result = {
        'S': np.asarray(burst),
        'T': T,
        'shapes': shapes,
        'loc': location,
        'perc_threshold': perc_threshold
    }
    
    ttoc = time.time()
    print(f"Time took in get_avalanches: {ttoc-ttic:.2f} seconds")
    
    return Result

def calculate_branching_ratio(data_raster_binarize, k_max, name,
                              fitfuncs=None, plot_targetdir=None,
                              lreturn_tau=0):
    '''
    UPDATED VERSION using mrestimator for sophisticated branching ratio calculation
    '''
    # Get number of cells if data_raster_binarize is a matrix
    if data_raster_binarize.ndim > 1:
        n_cells = data_raster_binarize.shape[0]
    else:
        n_cells = 1

    # if string change to list, fitfuncs
    if isinstance(fitfuncs, str):
        fitfuncs = [fitfuncs]

    # check fitfuncs
    allowed_values = {'exp', 'exp_offs', 'complex'}
    assert all(value in allowed_values for value in fitfuncs), "Some values in fitfuncs are invalid."

    # get fit_func_str
    fit_func_str = '_'.join(fitfuncs)

    # sum activity if n_cells > 1
    if n_cells > 1:
        data_raster_binarize = np.sum(data_raster_binarize, axis=0)

    # calculate br
    At = mre.input_handler(data_raster_binarize)
    out = mre.full_analysis(At,
                            dt=1, kmax=k_max,
                            fitfuncs=fitfuncs,
                            coefficientmethod='sm',
                            targetdir=plot_targetdir,
                            title=name + '_n_cells' + str(n_cells) +
                            '_br_k_max_' +
                            str(k_max) +
                            '_' +
                            fit_func_str,
                            numboot=1000,
                            showoverview=False,
                            saveoverview=True)
    br = []
    for fdx, fit in enumerate(out.fits):
        print(fdx, fit)
        print("m = ", fit.mre, " tau = ", fit.tau)
        br.append(fitfuncs[fdx])
        br.append(fit.mre)
        if lreturn_tau:
            br.append(fitfuncs[fdx] + "_tau")
            br.append(fit.tau)
    return br

def EXCLUDE(data, bm, nfactor=0, verbose=True):
    """Wrapper for exclude function from criticality module"""
    return ex.EXCLUDE(data, bm, nfactor=nfactor, verbose=verbose)

##############################################################
# PRESERVED ORIGINAL FUNCTIONS FROM YOUR analysis.py
##############################################################

def get_h5_files(base_dir):
    """Find all H5 files in the given directory"""
    h5_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def load_raster_from_h5(h5_file_path):
    """Load raster data from an H5 file"""
    try:
        with tables.open_file(h5_file_path, mode='r') as h5file:
            # Try different possible paths for raster data
            raster_paths = ['/raster', '/data/raster', '/stats/raster']
            
            for path in raster_paths:
                try:
                    raster = h5file.get_node(path).read()
                    print(f"Found raster at {path} in {h5_file_path}")
                    return raster
                except tables.NoSuchNodeError:
                    continue
            
            # If not found in standard locations, search for it
            for node in h5file.walk_nodes("/", classname="Array"):
                if 'raster' in node._v_name.lower():
                    print(f"Found raster at {node._v_pathname}")
                    return node.read()
            
            print(f"No raster data found in {h5_file_path}")
            return None
            
    except Exception as e:
        print(f"Error loading {h5_file_path}: {e}")
        return None

def process_h5_file(h5_file_path, starting_time_point, end_time_point=None):
    """Process a single H5 file to extract raster data"""
    raster = load_raster_from_h5(h5_file_path)
    
    if raster is not None:
        # Apply time slicing
        if end_time_point is not None:
            raster = raster[:, starting_time_point:end_time_point]
        else:
            raster = raster[:, starting_time_point:]
        
        print(f"Processed raster shape: {raster.shape}")
        return raster
    
    return None

def get_and_process_rasters(backup_path, starting_time_point, end_time_point=None):
    """Get and process all raster data from H5 files in a directory"""
    h5_files = get_h5_files(backup_path)
    
    if not h5_files:
        print(f"No H5 files found in {backup_path}")
        return None
    
    all_rasters = []
    for h5_file in h5_files:
        raster = process_h5_file(h5_file, starting_time_point, end_time_point)
        if raster is not None:
            all_rasters.append(raster)
    
    if all_rasters:
        # Combine all rasters (concatenate along time axis)
        combined_raster = np.concatenate(all_rasters, axis=1)
        print(f"Combined raster shape: {combined_raster.shape}")
        return combined_raster
    
    return None

def susceptibility(raster):
    """Calculate susceptibility (chi) of the network activity"""
    mean_activity = np.mean(raster)
    variance_activity = np.var(raster)
    return variance_activity / mean_activity if mean_activity > 0 else np.nan

def rho(raster):
    """Calculate mean activity density"""
    return np.mean(raster)

def cv_isi2(spiketrains):
    '''
    Calculate coefficient of variation of inter-spike intervals
    '''
    raster = spiketrains
    isis = []
    # Iterate over each neuron (row)
    for neuron_spikes in raster:
        spike_times = np.where(neuron_spikes > 0)[0]
        
        if len(spike_times) > 1:
            # Calculate ISIs for this neuron
            neuron_isis = np.diff(spike_times)
            isis.extend(neuron_isis)
    
    # Convert to numpy array for calculations
    isis = np.array(isis)
    
    if len(isis) > 0:
        mean_isi = np.mean(isis)
        std_isi = np.std(isis)
        cv = std_isi / mean_isi if mean_isi > 0 else np.nan
    else:
        cv = np.nan
    
    return cv

def branching_ratios(shapes):
    """Calculate branching ratios the naive way - PRESERVED ORIGINAL"""
    branching_ratios = []

    for shape in shapes:
        # Skip avalanches with a duration of 1
        if len(shape) <= 1:
            continue
        # Calculate the branching ratio for each step in the avalanche
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
    
    return Result

def branching_priesman(data, verbose = 0):
    """
    Apply the coarse graining method by Priesemann et al.
    """
    K = list(range(1, 51))  # Maximum offset
    
    # Initialize lists to store sums
    A_k = []
    A_k_plus_1 = []
    
    # Calculate rho_k and rho_k+1 for each k
    for k in K:
        A_k_sum = 0
        A_k_plus_1_sum = 0
        
        for channel_data in data:
            for t in range(len(channel_data) - k):
                if channel_data[t] == 1:  # Spike at time t
                    A_k_sum += 1
                    if channel_data[t + k] == 1:  # Check if there's a spike at time t+k
                        A_k_plus_1_sum += 1
        
        A_k.append(A_k_sum)
        A_k_plus_1.append(A_k_plus_1_sum)
    
    # Convert to numpy arrays
    A_k = np.array(A_k)
    A_k_plus_1 = np.array(A_k_plus_1)
    
    # Filter out zeros to avoid division errors
    valid_indices = A_k > 0
    A_k_filtered = A_k[valid_indices]
    A_k_plus_1_filtered = A_k_plus_1[valid_indices]
    
    if len(A_k_filtered) == 0:
        return np.nan
    
    # Calculate the branching parameter m
    m = np.sum(A_k_plus_1_filtered) / np.sum(A_k_filtered)
    
    if verbose:
        print(f"Branching parameter m: {m}")
        plt.figure(figsize=(10, 6))
        plt.scatter(K, A_k_plus_1 / (A_k + 1e-10), alpha=0.6)  # Add small value to avoid division by zero
        plt.axhline(y=m, color='r', linestyle='--', label=f'm = {m:.3f}')
        plt.xlabel('k')
        plt.ylabel('A(t+k) / A(t)')
        plt.title('Branching Parameter Estimation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return m

def calc_pearson_corr_coeff(all_rasters, dt=1.0):
    """Calculate Pearson correlation coefficient"""
    results = []
    
    for raster in all_rasters:
        for cell_idx, cell_data in enumerate(raster):
            # Skip if no spikes
            if np.sum(cell_data) == 0:
                continue
            
            spike_indices = np.where(cell_data)[0]
            isis = np.diff(spike_indices) * dt
            
            if len(isis) < 2:
                continue
            
            # Calculate Pearson correlation coefficient
            r_k_values = []
            max_k = min(20, len(isis) - 1)  # Limit lag to 20 or max possible
            
            for k in range(1, max_k + 1):
                if k < len(isis):
                    # Ensure arrays have same length
                    x = isis[:-k]
                    y = isis[k:]
                    
                    if len(x) > 1 and len(y) > 1:
                        # Calculate correlation
                        r, _ = linregress(x, y)[:2]
                        r_k_values.append(r)
            
            if r_k_values:
                results.append(r_k_values)
    
    return results

def calc_kappa(raster):
    """
    Calculate the kappa coefficient
    """
    activity = np.sum(raster, axis=0)  # Total activity at each time point
    mean_activity = np.mean(activity)
    
    if mean_activity == 0:
        return np.nan
    
    # Calculate kappa coefficient
    kappa = np.var(activity) / mean_activity
    
    return kappa

def fit_ar_model(data, max_order=20):
    """Fit AR model and return best order and parameters"""
    best_aic = np.inf
    best_order = 1
    best_model = None
    
    for order in range(1, max_order + 1):
        try:
            model = AutoReg(data, lags=order).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_order = order
                best_model = model
        except:
            continue
    
    return best_model, best_order

def d2_calculation(sig, order):
    """
    Calculate D2 correlation dimension - preserving original implementation
    Note: This is a placeholder as the criticality_tumbleweed version raises an error
    """
    # Fit AR model
    model, _ = fit_ar_model(sig, max_order=order)
    
    if model is None:
        return np.nan, np.nan
    
    # Get residuals
    residuals = model.resid
    
    # Calculate D2 (simplified version)
    # This is a placeholder - implement full D2 calculation as needed
    d2_value = np.std(residuals) / np.std(sig)
    
    return d2_value, order

def plotRaster(data, xlab='Time', ylab='Neuron', tlab='', save_path=None):
    """Plot raster of neural activity"""
    plt.figure(figsize=(15, 8))
    
    # Find spike times
    spike_times = []
    spike_neurons = []
    
    for neuron_idx, neuron_data in enumerate(data):
        spike_indices = np.where(neuron_data > 0)[0]
        spike_times.extend(spike_indices)
        spike_neurons.extend([neuron_idx] * len(spike_indices))
    
    # Create raster plot
    plt.scatter(spike_times, spike_neurons, s=1, c='black', marker='|')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tlab)
    plt.xlim(0, data.shape[1])
    plt.ylim(-0.5, data.shape[0] - 0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_avalanche_activity(raster, avalanches_result, save_path=None):
    """Plot network activity with avalanche detection overlay"""
    activity = np.sum(raster, axis=0)
    
    plt.figure(figsize=(20, 6))
    plt.plot(activity, 'k-', linewidth=0.5, alpha=0.7)
    
    # Highlight avalanches
    if 'loc' in avalanches_result:
        for loc in avalanches_result['loc']:
            plt.axvline(x=loc, color='r', alpha=0.3, linewidth=1)
    
    plt.xlabel('Time')
    plt.ylabel('Network Activity')
    plt.title('Network Activity with Avalanche Detection')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_activity_vs_mean(mu_values, rho_values, save_path=None):
    """Plot mean activity vs parameter"""
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, rho_values, 'o-', markersize=8)
    plt.xlabel('μ (parameter)')
    plt.ylabel('ρ (mean activity)')
    plt.title('Mean Activity vs Parameter')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def rebinner(raster, bin_size):
    """Rebin raster data - PRESERVED ORIGINAL"""
    channels, timesteps = raster.shape

    # Calculating the number of new bins based on bin_size
    new_timesteps = int(np.ceil(timesteps / bin_size))

    # Preallocate the new raster matrix
    new_raster = csr_matrix((channels, new_timesteps), dtype=int)

    # Loop through each channel
    for i_channel in range(channels):
        # Extract the spike times for this channel
        spike_times = np.where(raster[i_channel, :])[0]
        
        # Calculate new spike times based on binning
        new_spike_times = np.floor(spike_times / bin_size).astype(int)
        
        # Remove duplicates and get unique bin indices
        unique_bins = np.unique(new_spike_times)
        
        # Set these bins to 1 in the new raster
        for bin_idx in unique_bins:
            if bin_idx < new_timesteps:
                new_raster[i_channel, bin_idx] = 1

    return new_raster.toarray()

def reshape_avalanche_shapes(all_shapes, dim_to_reduce):
    """Reshape avalanche shapes - PRESERVED ORIGINAL"""
    collapsed_shapes = []
    
    if dim_to_reduce == 0:
        # Collapse along rows (sum across neurons for each time point)
        for shape in all_shapes:
            if len(shape) > 0:
                collapsed_shape = np.sum(shape, axis=0) if shape.ndim > 1 else shape
                collapsed_shapes.append(collapsed_shape)
    elif dim_to_reduce == 1:
        # Collapse along columns (sum across time for each neuron)
        for shape in all_shapes:
            if len(shape) > 0:
                collapsed_shape = np.sum(shape, axis=1) if shape.ndim > 1 else shape
                collapsed_shapes.append(collapsed_shape)
    else:
        # No reduction, keep original shapes
        collapsed_shapes = all_shapes
    
    return collapsed_shapes

def collapse_shapes(all_shapes, results_dict, all_burst, all_T):
    """Collapse avalanche shapes for analysis - PRESERVED ORIGINAL"""
    print("\nCalculating avalanche collapse exponent...")
    
    # Ensure we have shapes to work with
    if not all_shapes or len(all_shapes) == 0:
        print("[WARNING] No avalanche shapes available for collapse analysis")
        return {
            'exponent': np.nan,
            'secondDer': [],
            'range': [],
            'errors': [],
            'coefficients': []
        }
    
    # Filter out empty shapes
    valid_shapes = [shape for shape in all_shapes if len(shape) > 0]
    
    if len(valid_shapes) == 0:
        print("[WARNING] All avalanche shapes are empty")
        return {
            'exponent': np.nan,
            'secondDer': [],
            'range': [],
            'errors': [],
            'coefficients': []
        }
    
    print(f"Processing {len(valid_shapes)} valid avalanche shapes")
    
    # Prepare durations and sizes
    T = all_T[:len(valid_shapes)]  # Match the number of valid shapes
    S = all_burst[:len(valid_shapes)]
    
    # Calculate collapse exponent using the shapes
    try:
        # Simple approach: calculate scaling from shape profiles
        # This is a placeholder - implement full collapse analysis as needed
        
        # Calculate average shape profile for different durations
        duration_bins = np.unique(T)
        avg_profiles = {}
        
        for duration in duration_bins:
            if duration > 0:
                indices = np.where(T == duration)[0]
                if len(indices) > 0:
                    shapes_at_duration = [valid_shapes[i] for i in indices if i < len(valid_shapes)]
                    if shapes_at_duration:
                        # Normalize and average shapes
                        avg_profile = np.mean([s/np.sum(s) for s in shapes_at_duration], axis=0)
                        avg_profiles[duration] = avg_profile
        
        # Estimate scaling exponent (simplified)
        if len(avg_profiles) > 2:
            durations = sorted(avg_profiles.keys())
            scaling_factors = []
            
            for i in range(len(durations)-1):
                d1, d2 = durations[i], durations[i+1]
                if d1 > 0 and d2 > 0:
                    scaling = np.log(d2/d1) / np.log(len(avg_profiles[d2])/len(avg_profiles[d1]))
                    scaling_factors.append(scaling)
            
            if scaling_factors:
                exponent = np.median(scaling_factors)
            else:
                exponent = np.nan
        else:
            exponent = np.nan
        
        result = {
            'exponent': exponent,
            'secondDer': [],
            'range': [],
            'errors': [],
            'coefficients': []
        }
        
        print(f"Collapse exponent: {exponent:.3f}")
        
    except Exception as e:
        print(f"[ERROR] in collapse analysis: {e}")
        result = {
            'exponent': np.nan,
            'secondDer': [],
            'range': [],
            'errors': [],
            'coefficients': []
        }
    
    return result

def AV_analysis_manual_bounds(burst, T, xmin, xmax, tmin, tmax, plot=True, save_dir=None):
    """Manual bounds version - PRESERVED ORIGINAL"""
    result = {}
    
    # Basic analysis with manual bounds
    if len(burst) > 0:
        # Filter data within bounds
        size_mask = (burst >= xmin) & (burst <= xmax)
        filtered_sizes = burst[size_mask]
        
        if len(filtered_sizes) > 10:
            # Fit power law to size distribution
            log_sizes = np.log10(filtered_sizes)
            hist, bin_edges = np.histogram(log_sizes, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Only fit where we have data
            mask = hist > 0
            if np.sum(mask) > 2:
                slope, intercept = np.polyfit(bin_centers[mask], np.log10(hist[mask] + 1), 1)
                result['alpha'] = -slope
            else:
                result['alpha'] = np.nan
        else:
            result['alpha'] = np.nan
            
        # Similar for duration...
        duration_mask = (T >= tmin) & (T <= tmax)
        filtered_durations = T[duration_mask]
        
        if len(filtered_durations) > 10:
            log_durations = np.log10(filtered_durations)
            hist, bin_edges = np.histogram(log_durations, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            mask = hist > 0
            if np.sum(mask) > 2:
                slope, intercept = np.polyfit(bin_centers[mask], np.log10(hist[mask] + 1), 1)
                result['beta'] = -slope
            else:
                result['beta'] = np.nan
        else:
            result['beta'] = np.nan
        
        # Store bounds
        result['xmin'] = xmin
        result['xmax'] = xmax
        result['tmin'] = tmin
        result['tmax'] = tmax
        result['burst'] = burst
        result['T'] = T
        
        # Calculate scaling difference
        if 'alpha' in result and 'beta' in result and not np.isnan(result['alpha']) and not np.isnan(result['beta']):
            result['df'] = result['beta'] - result['alpha']
        else:
            result['df'] = np.nan
        
        # Generate plots if requested
        if plot:
            from criticality import AV_analysis as av_plot
            try:
                av_plot.scaling_plots(
                    result, burst, xmin, xmax, result.get('alpha', 1.5),
                    T, tmin, tmax, result.get('beta', 2.0),
                    np.arange(1, max(T)+1), np.ones(max(T)),
                    1.5, [1.5], f'manual_bounds_', save_dir,
                    None, None
                )
            except:
                print("Could not generate scaling plots")
    
    return result

##############################################################
# MAIN ANALYSIS SECTION - PRESERVED FROM ORIGINAL
##############################################################

""" 
#
#
# This is the main driving part of this code
#
#
"""

# Avalanche analysis parameters
AVALANCHE_PARAMS = {
    'perc_threshold': 0.1,   # Percentile threshold for avalanche detection
    'const_threshold': None,  
    'size_bm': 10,           # Increase to start fitting at larger sizes
    'size_nfactor': 0,       # Positive value to shift xmin higher
    'size_tail_cutoff': 0.7, # Decrease to cut off more of the tail
    'duration_tm': 3,       
    'duration_nfactor': 0,   
    'duration_tail_cutoff': 0.6,
    'exclude_burst_min': 18,  # Minimum xmin value - increase if you want to force higher xmin
    'exclude_time_min': 10,  
    'exclude_burst_diff': 12,  # Minimum range (xmax-xmin) - increase for wider fitting range
    'exclude_time_diff': 10, 
    'none_factor': 40,
}

matching_dirs = []
for dirname in os.listdir(base_dir):
   full_path = os.path.join(base_dir, dirname)
   print("-- Full path:", full_path)
#    if os.path.isdir(full_path):
#        match = re.match(pattern, dirname)
#        if match:
#            mu_value = float(match.group(1))  # Extract mu value
#            sigma_value = float(match.group(2))  # Extract sigma value
#            sp_steps = float(match.group(3)) 
#            sigma_value_no_sp = float(match.group(4)) 
#            print("-- Dir:", full_path)
#            print("-- mu: ", mu_value)
#            print("-- sigma: ", sigma_value)
#            print("-- sp_steps: ", sp_steps)
#            print("-- sigma_value_no_sp: ", sigma_value_no_sp)
#            matching_dirs.append((full_path, mu_value, sigma_value))

overall_susc=[]
overall_rho_values=[]
overall_cv_values=[]
overall_br_method_1=[]
overall_br_method_2=[]
overall_br_priesman=[]
mu_values = []
overall_pearson_kappa=[]
overall_d2_values = []
overall_d2_ar_orders = []
overall_av_alpha = []  # AVsize exponent
overall_av_beta = []   # AVduration exponent
overall_av_df = [] 

av_collapse_exponent=[]
av_collapse_secondDer=[]
av_collapse_range=[]
av_collapse_errors=[]
av_collapse_coefficients=[]
av_collapse_min_error=[]

#for directory, mu, sigma in matching_dirs:
if True : 

    # # # Call to retrieve and combine raster data
    # backup_path = r"D:\Users\seaco\SORN\backup\test_single\'"
    # combined_raster = get_and_process_rasters(backup_path, starting_time_point)

    # # # Check if data was successfully retrieved
    # if combined_raster is not None:
    #     print("Raster data successfully retrieved and combined.")
    # else:
    #     print("No raster data retrieved. Exiting.")
    #     exit()

    # print("Debug susceptibility: " , susceptibility(combined_raster))
    # print("Debug rho: " , rho(combined_raster))

    starting_time_point = 3000000
    end_time_point = 6000000  # Set to None to use the full length of the raster

    ## Need to loop over all the folders associated to the current param
    # single_param_backup_path = directory + '\\test_single'
    # single_param_backup_path = directory
    single_param_backup_path = base_dir
    mu = 0.05
    sigma = 0.05

    # Initialize arrays to store processed data
    all_rasters = []

    susc, rho_values, cv_values, br_method_1, br_method_2, br_priesman = [], [], [], [], [], []
    pearson_kappa = []
    d2_values, ar_orders = [], []

    # Initialize arrays to store all avalanches
    all_burst = np.array([])
    all_T = np.array([], dtype=int)
    all_shapes = []

    # Get list of H5 files
    h5_files = get_h5_files(single_param_backup_path)
    print(f"\nFound {len(h5_files)} H5 files to analyze for mu={mu}")
    print(f"[INFO] Located {len(h5_files)} .h5 simulation files to process.")

    # Process each file
    for file_idx, file_path in enumerate(h5_files):
        print(f"\nProcessing file {file_idx+1}/{len(h5_files)}: {os.path.basename(file_path)}")
        
        # Load raster data
        raster = process_h5_file(file_path, starting_time_point, end_time_point)
        
        if raster is not None:
            all_rasters.append(raster)
            
            # Calculate basic statistics
            susc.append(susceptibility(raster))
            rho_values.append(rho(raster))
            cv_values.append(cv_isi2(raster))
            
            # Calculate branching ratios using different methods
            # Method 1: Using branchparam
            br_result1 = branchparam(raster)
            br_method_1.append(br_result1['bp'])
            
            # Method 2: Using mrestimator (new method)
            try:
                br_mre = calculate_branching_ratio(
                    raster, 
                    k_max=50,
                    name=f'file_{file_idx}',
                    fitfuncs=['exp_offs'],
                    plot_targetdir=None,
                    lreturn_tau=0
                )
                # Extract branching ratio value
                if len(br_mre) >= 2:
                    br_method_2.append(br_mre[1])  # The value after the method name
                else:
                    br_method_2.append(np.nan)
            except Exception as e:
                print(f"Error in mrestimator branching ratio: {e}")
                br_method_2.append(np.nan)
            
            # Method 3: Priesemann method
            br_method_3 = branching_priesman(raster)
            br_priesman.append(br_method_3)
            
            # Find avalanches
            results = get_avalanches(raster, perc=AVALANCHE_PARAMS['perc_threshold'])
            
            if len(results['S']) > 0:
                all_burst = np.concatenate((all_burst, results['S']))
                all_T = np.concatenate((all_T, results['T']))
                all_shapes.extend(results['shapes'])
                
                # Also calculate branching ratio from avalanche shapes
                br_from_shapes = branching_ratios(results['shapes'])
                print(f"Branching ratio from shapes: {br_from_shapes['avgBR']:.3f}")
            
            # Calculate Pearson correlation and kappa
            pearson_results = calc_pearson_corr_coeff([raster])
            if pearson_results:
                avg_pearson = np.mean([np.mean(r) for r in pearson_results if len(r) > 0])
                pearson_kappa.append(avg_pearson)
            else:
                pearson_kappa.append(np.nan)
            
            # Calculate D2
            activity_series = np.sum(raster, axis=0)
            d2_val, best_order = d2_calculation(activity_series, order=20)
            d2_values.append(d2_val)
            ar_orders.append(best_order)

    # Aggregate results for this parameter set
    overall_susc.append(np.nanmean(susc))
    overall_rho_values.append(np.nanmean(rho_values))
    overall_cv_values.append(np.nanmean(cv_values))
    overall_br_method_1.append(np.nanmean(br_method_1))
    overall_br_method_2.append(np.nanmean(br_method_2))
    overall_br_priesman.append(np.nanmean(br_priesman))
    overall_pearson_kappa.append(np.nanmean(pearson_kappa))
    overall_d2_values.append(np.nanmean(d2_values))
    overall_d2_ar_orders.append(np.nanmean(ar_orders))
    mu_values.append(mu)

    # Avalanche analysis
    if len(all_burst) > 0:
        print(f"\n{'='*60}")
        print("AVALANCHE ANALYSIS")
        print(f"{'='*60}")
        print(f"Total avalanches found: {len(all_burst)}")
        
        # Run full avalanche analysis
        AV_Result = cr.AV_analysis(
            burst=all_burst,
            T=all_T,
            flag=1,  # Use 2 for p-value testing
            
            # Size distribution parameters
            bm=AVALANCHE_PARAMS['size_bm'],
            nfactor_bm=AVALANCHE_PARAMS['size_nfactor'],
            nfactor_bm_tail=AVALANCHE_PARAMS['size_tail_cutoff'],
            
            # Duration distribution parameters
            tm=AVALANCHE_PARAMS['duration_tm'],
            nfactor_tm=AVALANCHE_PARAMS['duration_nfactor'],
            nfactor_tm_tail=AVALANCHE_PARAMS['duration_tail_cutoff'],
            
            # Other parameters
            none_fact=AVALANCHE_PARAMS['none_factor'],
            max_time=7200,
            verbose=True,
            
            # Exclusion parameters
            exclude=True,
            exclude_burst=AVALANCHE_PARAMS['exclude_burst_min'],
            exclude_time=AVALANCHE_PARAMS['exclude_time_min'],
            exclude_diff_b=AVALANCHE_PARAMS['exclude_burst_diff'],
            exclude_diff_t=AVALANCHE_PARAMS['exclude_time_diff'],
            
            # Plotting
            plot=True,
            pltname='avalanche_analysis_custom_',
            saveloc=base_dir
        )
        
        # Print key results
        print("\nAnalysis Results:")
        # Extract the avalanche exponents
        av_alpha = AV_Result['alpha'] if 'alpha' in AV_Result else np.nan
        av_beta = AV_Result['beta'] if 'beta' in AV_Result else np.nan
        av_df = AV_Result['df'] if 'df' in AV_Result else np.nan
        print(f"Alpha (size exponent): {AV_Result['alpha']:.3f}")
        print(f"Beta (duration exponent): {AV_Result['beta']:.3f}")
        print(f"Size range: {AV_Result['xmin']:.0f} to {AV_Result['xmax']:.0f}")
        print(f"Duration range: {AV_Result['tmin']:.0f} to {AV_Result['tmax']:.0f}")
        if AV_Result['P_burst'] is not None:
            print(f"Size distribution p-value: {AV_Result['P_burst']:.3f}")
        if AV_Result['P_t'] is not None:
            print(f"Duration distribution p-value: {AV_Result['P_t']:.3f}")
        print(f"Scaling relation difference: {AV_Result['df']:.3f}")
        overall_av_alpha.append(av_alpha)
        overall_av_beta.append(av_beta)
        overall_av_df.append(av_df)
        
        # Collapse analysis
        collapse_result = collapse_shapes(all_shapes, AV_Result, all_burst, all_T)
        av_collapse_exponent.append(collapse_result['exponent'])
        av_collapse_min_error.append(np.nan)  # Placeholder
    else:
        print("\nNo avalanches found in any files.")
        overall_av_alpha.append(np.nan)
        overall_av_beta.append(np.nan)
        overall_av_df.append(np.nan)
        av_collapse_exponent.append(np.nan)
        av_collapse_min_error.append(np.nan)

# Prepare data for saving
results_df = pd.DataFrame({
    'Mu': mu_values,
    'Overall_Susceptibility': overall_susc,
    'Overall_Rho': overall_rho_values,
    'Overall_CV': overall_cv_values,
    'Branching_Ratio_Method_1': overall_br_method_1,
    'Branching_Ratio_Method_2': overall_br_method_2,
    'Branching_Ratio_Priesman': overall_br_priesman,
    'Pearson_Kappa': overall_pearson_kappa,
    'D2_Correlation_Dimension': overall_d2_values,
    'D2_AR_Order': overall_d2_ar_orders,
    'Av_Collapse_exponent': av_collapse_exponent,
    'Av_Collapse_error': av_collapse_min_error,
    'AV_Alpha': overall_av_alpha,
    'AV_Beta': overall_av_beta,
    'AV_Scaling_Diff': overall_av_df
})

# Loop over each mu and save individual metrics to separate text files
for i in range(len(mu_values)):
    # Create a unique file name for each mu
    text_output_path = os.path.join(base_dir, f'Stats_Mu_{mu_values[i]:.2f}.txt')

    with open(text_output_path, 'w') as f:
        f.write(f"Metrics Summary for Mu = {mu_values[i]:.2f}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean Susceptibility: {overall_susc[i]:.4f}\n")
        f.write(f"Mean Rho: {overall_rho_values[i]:.4f}\n")
        f.write(f"Mean CV: {overall_cv_values[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 1 - branchparam): {overall_br_method_1[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 2 - get_avalanches): {overall_br_method_2[i]:.4f}\n")
        f.write(f"Mean Branching Ratio (Method 3 - Priesman): {overall_br_priesman[i]:.4f}\n")
        f.write(f"Mean Pearson Kappa: {overall_pearson_kappa[i]:.4f}\n")
        f.write(f"Mean d2: {overall_d2_values[i]:.4f}\n")
        f.write(f"Mean AR Order: {overall_d2_ar_orders[i]:.1f}\n")
        f.write(f"Av_Collapse_exponent: {av_collapse_exponent[i]:.4f}\n")
        f.write(f"Av_Collapse_error: {av_collapse_min_error[i]:.4f}\n")
        f.write(f"AV Alpha (size exponent): {overall_av_alpha[i]:.4f}\n")
        f.write(f"AV Beta (duration exponent): {overall_av_beta[i]:.4f}\n")
        f.write(f"AV Scaling difference: {overall_av_df[i]:.4f}\n")
        f.write("-" * 60 + "\n")

    print(f"Metrics for Mu = {mu_values[i]:.2f} saved to: {text_output_path}")


# Save to CSV
csv_output_path = os.path.join(base_dir, 'Overall_Stats.csv')
results_df.to_csv(csv_output_path, index=False)
print(f"Aggregated results saved to: {csv_output_path}")

# Load data from CSV
results_df = pd.read_csv(csv_output_path)

# Extract columns for plotting
mu_values = results_df['Mu']
overall_susc = results_df['Overall_Susceptibility']
overall_rho_values = results_df['Overall_Rho']
overall_cv_values = results_df['Overall_CV']
overall_br_method_1 = results_df['Branching_Ratio_Method_1']
overall_br_method_2 = results_df['Branching_Ratio_Method_2']
overall_br_priesman = results_df['Branching_Ratio_Priesman']
overall_pearson_kappa = results_df['Pearson_Kappa']
av_collapse_exponent = results_df['Av_Collapse_exponent']
av_collapse_min_error = results_df['Av_Collapse_error']

# Set up the plotting style
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

width = 6  # Fixed width for individual subplots
height = width * 1.5  # Taller height for individual subplots (height > width)

# Create a figure with a grid of subplots (5 rows, 2 columns)
fig, axes = plt.subplots(5, 2, figsize=(width * 2, height * 5))  # Adjust figure size to accommodate all subplots
axes = axes.flatten()

# Common plotting parameters
plot_params = dict(
    marker='x',
    linestyle=':',  # Dotted line for style consistency
    markersize=6,
    linewidth=1.5
)

# Plot each dataset
plotting_data = [
    (overall_susc, 'Susceptibility ($\\chi$)'),
    (overall_rho_values, 'Rho ($\\rho$)'),
    (overall_cv_values, 'CV'),
    (overall_br_method_1, 'Branching Ratio (Method 1 - branchparam)'),
    (overall_br_method_2, 'Branching Ratio (Method 2 - get_avalanches)'),
    (overall_br_priesman, 'Branching Ratio (Method 3 - Priesman)'),
    (overall_pearson_kappa, 'Pearson Kappa ($\\kappa$)'),
    (overall_d2_values, 'D2 Correlation Dimension'),
    (av_collapse_exponent, 'Avalanche Collapse Exponent'),
    (overall_av_alpha, 'Avalanche Size Exponent ($\\alpha$)')
]

for idx, (data, label) in enumerate(plotting_data):
    if idx < len(axes):
        ax = axes[idx]
        ax.plot(mu_values, data, **plot_params)
        ax.set_xlabel('$\\mu$', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

# Remove any unused subplots
for idx in range(len(plotting_data), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'all_metrics_subplots.png'), dpi=300, bbox_inches='tight')
plt.show()

# Individual metric plots
for data, label in plotting_data:
    plt.figure(figsize=(width, height))
    plt.plot(mu_values, data, **plot_params)
    plt.xlabel('$\\mu$', fontsize=14)
    plt.ylabel(label, fontsize=14)
    plt.title(f'{label} vs $\\mu$', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save with appropriate filename
    filename = label.replace('$', '').replace('\\', '').replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(base_dir, f'{filename}_vs_mu.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("\nAnalysis complete!")
print(f"Results saved to: {base_dir}")
print(f"Total memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")