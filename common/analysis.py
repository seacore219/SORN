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
    # print(f’nonzero TIMERASTER {np.nonzero(TIMERASTER)}‘)
    # print(f’nonzero {np.where(TIMERASTER != 0)[0]}‘)    # Find frames with at least one active site and
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
            # print(f’i {i} ancestors {ancestors} num {num} descendants(num+1)
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
from copy import deepcopy as cdc
import time


def get_avalanches(data, perc=0.10, ncells=-1, const_threshold=None):

    '''
    Function that goes through an array of binned spikes and determines
        the avalanch boundaries and properties

    parameters:
        data - array of spike times. one row for each neuron

        perc - threshold for defining an avalanche,
            if network is dominated by silent periods you can use 0
            if const_threshold is used perc will be ignored

        ncells - default (-1), number of cells calculated from data shape
                 if ncells given expect data to be 1 dimensional
                 np.nansum(data, axis=0)

        const_threshold - Define avalanche threshold directly instead of perc

                         default (None), use perc to find threshold

                         if const_threshold is not None,
                            use this value (integer >= 0)

    returns:
    Result - a dictionary with 2 inputs.
        'S' is the size of each avalanche (number of spikes above threshold)
        'T' is the duration (number of time bins avalanche spanned)
    '''

    # To time the function
    ttic = time.time()

    # num cells, num bins
    if ncells == -1:
        n, m = np.shape(data)
    else:
        n = ncells
        m = np.shape(data)[0]
    print("Data has {} neurons with length {}*binsize".format(n, m))

    if n == 1:
        network = cdc(data)
    else:
        # collapse into single array. sum the amount of activity in each bin.
        if ncells == -1:
            network = np.nansum(data, axis=0)
        else:
            network = data.copy()

    data = None
    del data

    if const_threshold is None:
        if perc > 0:
            sortN = np.sort(network)
            # determine the treshold. if perc is .25,
            # then its 25% of network activity essentially
            perc_threshold = sortN[round(m * perc)]
            print("perc_threshold ", perc_threshold)
            sortN = None
            del sortN
        else:
            perc_threshold = 0
            print("perc_threshold ", perc_threshold)
    elif const_threshold is not None:
        if const_threshold < 0:
            raise ValueError(f'const_threshold < 0, {const_threshold}')
        print("const_threshold ", const_threshold)
        perc_threshold = const_threshold
        print("perc_threshold ", perc_threshold)

    zdata = cdc(network)

    # intervals
    zdata[zdata <= perc_threshold] = 0

    # avalanches
    zdata[zdata > perc_threshold] = 1
    zdata = zdata.astype(np.int8)

    # location of intervals
    zeros_loc_zdata = np.where(zdata == 0)[0]
    zeros_to_delete = \
        zeros_loc_zdata[np.where(np.diff(zeros_loc_zdata) == 1)[0]]
    zeros_loc_zdata = None
    del zeros_loc_zdata

    # cuts out irrelevant 0s result=>, series of 1s and a single 0 separation
    # 1 0 0 1 :  12  5  6  9 (if perc_threshold = 7)  =>  1 0  1 : 12  6  9
    # in short in a series of zeros last zero index is not deleted
    z1data = np.delete(zdata, zeros_to_delete)
    avalanches = np.delete(network, zeros_to_delete)
    # use a single 0 to separate network activities in each avalanche
    avalanches[z1data == 0] = 0

    # location of the intervals
    zeros_loc_z1data = np.where(z1data == 0)[0]

    # Now calculate S and T based on avalanches and z1data
    burst = []
    shapes = []
    for i in np.arange(0, np.size(zeros_loc_z1data) - 1):
        tmp_av = avalanches[zeros_loc_z1data[i] + 1:zeros_loc_z1data[i + 1]]
        tmp_burst = np.sum(tmp_av) - (perc_threshold * len(tmp_av))
        if tmp_burst <= 0:
            raise RuntimeError('Burst value {}, zero/negative'
                               .format(tmp_burst))
        burst.append(tmp_burst)
        shape = tmp_av - perc_threshold
        shapes.append(shape[shape > 0])
    tmp_av = None
    del tmp_av

    # AVduration
    T = np.diff(zeros_loc_z1data) - 1
    # Duration should be positive
    T = T[T > 0]

    z2data = zdata[0:-1]
    z2data = np.insert(z2data, 0, 0)
    location = np.where(np.logical_and(zdata == 1, z2data == 0))[0]
    #
    # # data = np.array([[1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 3, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0]])
    # r = get_avalanches(data, perc=0.5, ncells=-1, const_threshold=None)
    # r['S'] array([1, 5])
    # r['loc'] gave array([]) instead of array([ 6, 15])
    # so commented line below
    # location = location[1:-1]

    Result = {
        'S': np.asarray(burst),
        'T': T,
        'shapes': shapes,
        'loc': location,
        'perc_threshold': perc_threshold
    }

    ttoc = time.time()
    print("Time took in get_avalanches is {:.2f} seconds".format(ttoc-ttic),
          flush=True)

    return Result

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
    
##############################################################

##############################################################
# Function for logarithmic binning avalanche data

def bin_data(data, logb, xmin, xmax):
    dist = np.sort(data)
    Nav = len(dist)
    jmax = int(np.round(np.log(max(dist)) / np.log(logb)))

    bincell = []

    for i in range(jmax):
        bincell.append(dist[(dist < logb ** (i + 1)) & (dist >= logb ** i)])

    Ps = []
    dist1 = []

    for j in range(len(bincell)):
        if len(bincell[j]) > 0:
            Ps.append(len(bincell[j]) / (Nav * (max(bincell[j]) - min(bincell[j]) + 1)))
            dist1.append(np.sqrt(max(bincell[j]) * min(bincell[j])))
        else:
            Ps.append(0)
            dist1.append(0)

    Ps = np.array(Ps)[Ps != 0]
    dist1 = np.array(dist1)[dist1 != 0]

    while xmax > (len(Ps) - 3):
        xmax = xmax - 1

    x = np.log10(dist1[xmin:xmax])
    y = np.log10(Ps[xmin:xmax])

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    m = abs(slope)
    merr = std_err

    x1 = np.linspace(0, 4.5)
    y1 = -m * x1 + intercept
    
    Result = {
    'Ps': Ps,
    'dist': dist1,
    'exponent': m,
    'expstd': merr,
    'fitdistx': x,
    'fitdisty': y,
    'fitx': x1,
    'fity': y1
    }

    return Result

##############################################################

##############################################################
# Function to compute Pearson correlations with lags
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def correlations(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, saveloc=base_dir
, pltname='analysis', state = ''):
    """
    Compute the Pearson correlation matrix for neuron activities at a given lag, optionally plot eigenvalue spectrum, 
    plot the distribution of weights, and fit an exponential decay.

    Parameters:
        raster (np.array): The raster data matrix with shape (n_neurons, t_length).
        lag (int): The lag for computing correlations.
        plot_eigenvalues (bool): If True, plot the eigenvalue spectrum in the complex plane.
        plot_weights (bool): If True, plot the distribution of weights and fit an exponential decay.
        save_plot (bool): If True, save the plots instead of displaying them.
        saveloc (str): The directory to save the plot.
        pltname (str): The name of the saved plot file.
        
    Returns:
        correlation_matrix (np.array): The computed correlation matrix.
        max_eigenvalue (float): The maximum absolute eigenvalue of the correlation matrix.
        exponential_params (tuple): Parameters of the fitted exponential decay function.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag  # adjust length for lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(n_neurons):
            series_i = raster[i, :(t_length)]
            series_j = raster[j, lag:(lag + t_length)]
            if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
            else:
                correlation_matrix[i, j] = 0

    # Compute eigenvalues for the original correlation matrix
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    # Plot distribution of weights and fit an exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Get the absolute values of the correlation matrix and sort them for each neuron
        sorted_weights = np.sort(np.abs(correlation_matrix), axis=1)[:, ::-1]  # Sort in descending order
        
        # Aggregate the sorted weights across all neurons
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        # Plot the distribution of weights
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        # Fit an exponential decay to the distribution
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        
        # Initial parameter guess and bounds
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, p0=p0, bounds=bounds, maxfev=10000)
        exponential_params = popt
        
        # Plot the fitted exponential decay
        plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        else:
            plt.show()
        plt.close()

    #return correlation_matrix, max_eigenvalue, exponential_params
    return max_eigenvalue, exponential_params

##############################################################


# ---- with shuffling
##############################################################
def correlations_with_shuffling(raster, lag=1, plot_eigenvalues=False, save_plot=True, saveloc=base_dir, pltname='eigenvalue_spectrum', n_shuffles=100):
    """
    Compute the Pearson correlation matrix for neuron activities at a given lag and optionally plot eigenvalue spectrum.
    Includes shuffling for noise estimation.

    Parameters:
        raster (np.array): The raster data matrix with shape (n_neurons, t_length).
        lag (int): The lag for computing correlations.
        plot_eigenvalues (bool): If True, plot the eigenvalue spectrum in the complex plane.
        save_plot (bool): If True, save the plot instead of displaying it.
        save_location (str): The directory to save the plot.
        plot_name (str): The name of the saved plot file.
        n_shuffles (int): Number of shuffles to generate surrogate data for noise estimation.
        
    Returns:
        correlation_matrix (np.array): The computed correlation matrix.
        max_eigenvalue_original (float): The maximum eigenvalue from the original data.
        max_eigenvalue_shuffled_mean (float): The mean of the maximum eigenvalues from shuffled data.
        max_eigenvalue_shuffled_std (float): The standard deviation of the maximum eigenvalues from shuffled data.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag  # adjust length for lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(n_neurons):
            series_i = raster[i, :(t_length)]
            series_j = raster[j, lag:(lag + t_length)]
            if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
            else:
                correlation_matrix[i, j] = 0

    # Compute eigenvalues for the original correlation matrix
    eigenvalues_original = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue_original = np.max(np.abs(eigenvalues_original))

    # Generate surrogate data by shuffling and compute eigenvalues
    max_eigenvalues_shuffled = []
    for _ in range(n_shuffles):
        shuffled_raster = np.copy(raster)
        for i in range(n_neurons):
            np.random.shuffle(shuffled_raster[i, :])  # shuffle spike times within each neuron
        shuffled_correlation_matrix = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                series_i = shuffled_raster[i, :(t_length)]
                series_j = shuffled_raster[j, lag:(lag + t_length)]
                if np.any(series_i) and np.any(series_j):  # check if there's data in the slice
                    shuffled_correlation_matrix[i, j] = np.corrcoef(series_i, series_j)[0, 1]
                else:
                    shuffled_correlation_matrix[i, j] = 0
        eigenvalues_shuffled = np.linalg.eigvals(shuffled_correlation_matrix)
        max_eigenvalues_shuffled.append(np.max(np.abs(eigenvalues_shuffled)))

    max_eigenvalue_shuffled_mean = np.mean(max_eigenvalues_shuffled)
    max_eigenvalue_shuffled_std = np.std(max_eigenvalues_shuffled)

    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues_original.real, eigenvalues_original.imag, color='red', label='Original Data')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Spectrum in the Complex Plane')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    #return correlation_matrix, max_eigenvalue_original, max_eigenvalue_shuffled_mean, max_eigenvalue_shuffled_std
    return max_eigenvalue_original, max_eigenvalue_shuffled_mean, max_eigenvalue_shuffled_std

##############################################################

##############################################################
# Function to compute power spectrum
def welch_psd(raster_data, fs=1000, nperseg=10000, x1=0, x2=1, plot=False, save_plot=True, filename=os.path.join(base_dir, 'psd_plot.pdf')):
    """
    Compute the Power Spectral Density (PSD) using Welch's method.
    
    :param raster_data: 2D numpy array where each row is a time series for a given neuron.
    :param fs: Sampling frequency in Hz.
    :param nperseg: Length of each segment for Welch's method (buiilt in function, maybe we can try others later).
    :param x1, x2: Lower and upper bounds of the frequency region to fit.
    :param plot: Boolean, whether to display the plot.
    :param save_plot: Boolean, whether to save the plot as a PDF.
    :param filename: String, name of the file to save the plot.
    :return: Frequencies, PSD, and slope of the fit in the specified region.
    """
    # Collapse the raster data into one time series
    collapsed_series = np.sum(raster_data, axis=0)
    
    # Compute PSD using Welch's method (you can change the overlap window, but by default it does the nperseg/2 which is fine)
    f, psd = welch(collapsed_series, fs, nperseg=nperseg, noverlap=None)

    # Convert data to logarithmic scale
    log_f = np.log10(f[f > 0])  # avoid log(0) by ensuring frequencies are > 0
    log_psd = np.log10(psd[f > 0])

    # Define the region for fitting
    index1 = (log_f > x1) & (log_f < x2)  # Example region 1

    # Fit linear model to the region
    slope, intercept = np.polyfit(log_f[index1], log_psd[index1], 1)

    # Plotting stuff
    if plot or save_plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(f, psd, label='Original PSD')
        plt.loglog(10**log_f[index1], 10**(log_f[index1]*slope + intercept), 'r-', label=f'Fit: slope={slope:.2f}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('Power Spectral Density')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        if save_plot:
            plt.savefig(filename)
        if plot:
            plt.show()
        plt.close()

    return f, psd, slope

#################################
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.stats.stattools import durbin_watson

# Define the triple exponential model
def triple_exponential(x, A1, A2, A3, lambda1, lambda2, lambda3):
    return A1 * np.exp(-lambda1 * x) + A2 * np.exp(-lambda2 * x) + A3 * np.exp(-lambda3 * x)

# Define the double exponential model
def double_exponential(x, A1, lambda1, lambda2):
    return A1 * np.exp(-lambda1 * x) + (1 - A1) * np.exp(-lambda2 * x)

##############################################################
# Function to compute autocorrelation and fit the model
def compute_autocorrelation(time_series, name, min_lag=0, max_lag=None, model_type='triple'):
    """
    Compute the autocorrelation of a given time series and fit an exponential decay model (double or triple).
    
    :param time_series: 1D numpy array of time series data
    :param name: Name for saving the plot
    :param min_lag: Minimum lag to consider for the fit
    :param max_lag: Maximum lag to consider for the fit
    :param model_type: Choose between 'double' or 'triple' exponential fit
    :return: Fitted parameters based on the chosen model type
    """
    # Preprocess the time series
    time_series = time_series - np.mean(time_series)
    variance = np.var(time_series)

    # Compute the autocorrelation
    n = len(time_series)
    result = np.correlate(time_series, time_series, mode='full')
    autocorr = result[result.size // 2:] / (variance * np.arange(n, 0, -1))

    if max_lag is None:
        max_lag = len(autocorr)

    lags = np.arange(len(autocorr))
    fit_lags = lags[min_lag:max_lag]
    fit_autocorr = autocorr[min_lag:max_lag]

    # Fit the chosen model type
    try:
        if model_type == 'triple':
            # Initial parameters for triple exponential: A1, A2, A3, lambda1, lambda2, lambda3
            initial_params = (0.3, 0.3, 0.4, 0.01, 0.001, 0.0001)
            popt, _ = curve_fit(triple_exponential, fit_lags, fit_autocorr, p0=initial_params, maxfev=5000)
            
            # Extract the parameters A1, A2, A3, lambda1, lambda2, lambda3
            A1, A2, A3, lambda1, lambda2, lambda3 = popt
            # Sort parameters based on coefficients (A1, A2, A3) in descending order
            coeffs_and_lambdas = sorted(zip([A1, A2, A3], [lambda1, lambda2, lambda3]), reverse=True)
            (A1, lambda1), (A2, lambda2), (A3, lambda3) = coeffs_and_lambdas

            fitted_values = triple_exponential(fit_lags, A1, A2, A3, lambda1, lambda2, lambda3)
            fitted_params = (A1, A2, A3, lambda1, lambda2, lambda3)

            # Format the parameters for the legend
            legend_text = f"A1={A1:.2f}, A2={A2:.2f}, A3={A3:.2f},\n" \
                          f"tau1={1/lambda1:.4f}, tau2={1/lambda2:.4f}, tau3={1/lambda3:.4f}"

        elif model_type == 'double':
            # Initial parameters for double exponential: A1, lambda1, lambda2
            initial_params = (0.5, 0.01, 0.001)
            popt, _ = curve_fit(double_exponential, fit_lags, fit_autocorr, p0=initial_params, maxfev=5000)

            # Extract the parameters A1, lambda1, lambda2
            A1, lambda1, lambda2 = popt

            fitted_values = double_exponential(fit_lags, A1, lambda1, lambda2)
            fitted_params = (A1, lambda1, lambda2)

            # Format the parameters for the legend
            legend_text = f"A1={A1:.2f}, tau1={1/lambda1:.4f}, tau2={1/lambda2:.4f}"

        else:
            raise ValueError("Invalid model type. Choose 'double' or 'triple'.")

        # Compute residuals
        residuals = fit_autocorr - fitted_values
        dw_stat = durbin_watson(residuals)  # Apply Durbin-Watson test for autocorrelation in residuals
        print(f'Durbin-Watson statistic for {name}: {dw_stat}')

    except RuntimeError:
        print(f"Failed to fit the {model_type} exponential model for {name}.")
        if model_type == 'triple':
            fitted_params = [np.nan] * 6
            legend_text = "Fit failed"
        elif model_type == 'double':
            fitted_params = [np.nan] * 3
            legend_text = "Fit failed"
        fitted_values = np.zeros_like(fit_autocorr)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Linear scale plot
    plt.subplot(1, 2, 1)
    plt.plot(fit_lags, fit_autocorr, label='Autocorrelation')
    plt.plot(fit_lags, fitted_values, 'r-', label=f'{model_type.capitalize()} Exponential Fit')
    plt.legend(loc='upper right', title=legend_text)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation with {model_type.capitalize()} Exponential Fit (Linear Scale)')
    plt.grid(False)

    # Logarithmic scale plot
    plt.subplot(1, 2, 2)
    plt.plot(fit_lags, fit_autocorr, label='Autocorrelation')
    plt.plot(fit_lags, fitted_values, 'r-', label=f'{model_type.capitalize()} Exponential Fit')
    plt.yscale('log')
    plt.legend(loc='upper right', title=legend_text)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation (log scale)')
    plt.title(f'Autocorrelation with {model_type.capitalize()} Exponential Fit (Log Scale)')
    plt.grid(False)

    # Save the figure as a PDF
    plt.savefig(f'ACF_{name}_{model_type}_exp_fit.pdf')
    plt.close()

    return fitted_params

# Example usage:
# compute_autocorrelation(time_series_data, 'example', model_type='double')  # For double exponential fit
# compute_autocorrelation(time_series_data, 'example', model_type='triple')  # For triple exponential fit


'''
def double_exponential(x, A1, lambda1, A2, lambda2):
    return A1 * np.exp(-lambda1 * x) + A2 * np.exp(-lambda2 * x)

def compute_autocorrelation(time_series,name):
    """
    Compute the autocorrelation of a given time series and fit a double exponential decay.

    :param time_series: 1D numpy array of time series data
    :return: autocorrelation values
    """
    n = len(time_series)
    time_series = time_series - np.mean(time_series)
    result = np.correlate(time_series, time_series, mode='full')
    autocorr = result[result.size // 2:] / max(result)
    
    # Fit the double exponential decay
    lags = np.arange(len(autocorr))
    try:
        popt, _ = curve_fit(double_exponential, lags, autocorr, p0=(1, 0.01, 1, 0.001), maxfev=5000, bounds=(0, np.inf))
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan, np.nan]  # if fitting fails, return NaNs


    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Linear y-axis plot
    plt.subplot(1, 2, 1)
    plt.plot(lags, autocorr, label='Autocorrelation')
    plt.plot(lags, double_exponential(lags, *popt), 'r-', label=f'Fit: A1={popt[0]:.2f}, lambda1={popt[1]:.4f}, A2={popt[2]:.2f}, lambda2={popt[3]:.4f}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation with Double Exponential Fit (Linear Scale)')
    plt.legend()
    plt.grid(False)
    
    # Logarithmic y-axis plot
    plt.subplot(1, 2, 2)
    plt.plot(lags, autocorr, label='Autocorrelation')
    plt.plot(lags, double_exponential(lags, *popt), 'r-', label=f'Fit: A1={popt[0]:.2f}, lambda1={popt[1]:.4f}, A2={popt[2]:.2f}, lambda2={popt[3]:.4f}')
    plt.yscale('log')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation (log scale)')
    plt.title('Autocorrelation with Double Exponential Fit (Log Scale)')
    plt.legend()
    plt.grid(False)
    
    # Save the figure as a PDF
    #plt.savefig(f'autocorrelation_{name}.pdf')
    plt.show
    #plt.close()

    return autocorr, popt
'''

# Function to compute eigenvalues and identify state-dense windows
def compute_measures_and_state_dense(data, sleep_states, window_size_bins, overlap_bins, binsz, condition):
    n_windows = (data.shape[1] - window_size_bins) // overlap_bins + 1
    print("# windows:", n_windows)
    results = []

    for i in range(n_windows):
        print("----------- New window ---------------")
        print("--------------------------------------")
        start_bin = i * overlap_bins
        end_bin = start_bin + window_size_bins
        temp = data[:, start_bin:end_bin] # this is the piece of data
        print("start bin", start_bin)
        print("end bin", end_bin)
        print("----------- sleep states -------------")

        # Compute correlation matrix
        correlation_matrix = compute_correlations(temp, lag=0)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue0 = np.max(np.abs(eigenvalues))
        correlation_matrix = compute_correlations(temp, lag=1)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue1 = np.max(np.abs(eigenvalues))
        correlation_matrix = compute_correlations(temp, lag=2)
        eigenvalues, _ = np.linalg.eig(correlation_matrix)
        max_evalue2 = np.max(np.abs(eigenvalues))

        # compute autocorrelations
        
        _, fits = compute_autocorrelation(np.sum(temp, axis=0), condition + '_' + str(i))
        # Determine if the window is sleep-dense or wake-dense
                
        # convert time bins into 4 second bins for the sleep_states
        startss = int(start_bin*binsz/4)
        endss = int(end_bin*binsz/4)
        print("startss", startss)
        print("endss", endss)
        
        sleep_duration = np.sum(sleep_states[startss:endss] == 2) + np.sum(sleep_states[startss:endss] == 3)
        print("sleep duration:", sleep_duration)
        wake_duration = int(window_size_bins*binsz/4) - sleep_duration
        print("wake duration:", wake_duration)
        if sleep_duration >= 0.6 * int(window_size_bins*binsz/4):
            state = 'sleep'
            print("sleep dense")
        elif wake_duration >= 0.6 * int(window_size_bins*binsz/4):
            state = f'wake/{condition}'
            print("wake dense")
        else:
            state = 'none'

        results.append({
            'window': i+1, 
            'behavior_condition': state, 
            'max_evalue lag 0': max_evalue0,
            'max_evalue lag 1': max_evalue1,
            'max_evalue lag 2': max_evalue2, 
            'A1': fits[0],'lambda1': 1/fits[1],
            'A2': fits[2],'lambda2': 1/fits[3],})
        

    return results

##############################################################
##############################################################

# Function to extract and plot excitatory and inhibitory rasters
def extract_and_plot_rasters(h5_path, time_window=None, save_plot=True, saveloc=base_dir, max_timesteps=5000000):
    """
    Extract and plot both excitatory and inhibitory neuron rasters
    
    Args:
        h5_path: Path to result.h5 file
        time_window: Optional tuple (start, end) to plot specific time range
        save_plot: Whether to save the plot
        saveloc: Directory to save the plot
        max_timesteps: Maximum timesteps to load (default 500k)
    """
    h5 = tables.open_file(h5_path, 'r')
    
    # Get network parameters
    N_e = h5.root.c.N_e[0]  # Number of excitatory neurons
    N_i = h5.root.c.N_i[0]  # Number of inhibitory neurons
    
    print(f"Network: {N_e} excitatory, {N_i} inhibitory neurons")
    
    # Check what spike data exists and get total length
    excitatory_raster = None
    inhibitory_raster = None
    total_timesteps = 0
    
    # Check for main Spikes data
    if hasattr(h5.root, 'Spikes'):
        spike_data_shape = h5.root.Spikes.shape
        total_timesteps = spike_data_shape[2]
        print(f"Found Spikes data: shape {spike_data_shape}")
        
        # Calculate which timesteps to load (last max_timesteps)
        if time_window is None:
            start_idx = max(0, total_timesteps - max_timesteps)
            end_idx = total_timesteps
            print(f"Loading LAST {end_idx - start_idx} timesteps (from {start_idx} to {end_idx})")
        else:
            start_idx, end_idx = time_window
            print(f"Loading specified window: {start_idx} to {end_idx}")
        
        # Load only the required slice directly from HDF5
        spike_data = h5.root.Spikes[0, :, start_idx:end_idx]
        
        if spike_data.shape[0] == N_e:
            # Only excitatory neurons saved
            excitatory_raster = spike_data
            print("Spikes contains only excitatory neurons")
        elif spike_data.shape[0] == N_e + N_i:
            # Both types saved together
            excitatory_raster = spike_data[:N_e, :]
            inhibitory_raster = spike_data[N_e:, :]
            print("Spikes contains both neuron types")
    
    # Check for separate inhibitory spikes
    if hasattr(h5.root, 'SpikesInh'):
        inh_shape = h5.root.SpikesInh.shape
        if time_window is None:
            start_idx = max(0, inh_shape[2] - max_timesteps)
            end_idx = inh_shape[2]
        else:
            start_idx, end_idx = time_window
        inhibitory_raster = h5.root.SpikesInh[0, :, start_idx:end_idx]
        print(f"Found SpikesInh data: shape {inhibitory_raster.shape}")
    
    h5.close()
    
    # Get dimensions
    if excitatory_raster is not None:
        time_steps = excitatory_raster.shape[1]
    elif inhibitory_raster is not None:
        time_steps = inhibitory_raster.shape[1]
    else:
        time_steps = 1000  # fallback
    
    print(f"Loaded data shape: {time_steps} timesteps")
    
    # Calculate average interspike intervals
    def calc_avg_isi(raster_data, neuron_type):
        if raster_data is None or raster_data.size == 0:
            return None
        
        spike_times, neuron_ids = np.where(raster_data)
        if len(spike_times) == 0:
            return None
        
        # Calculate ISIs for each neuron
        isis = []
        for neuron in range(raster_data.shape[0]):
            neuron_spikes = spike_times[neuron_ids == neuron]
            if len(neuron_spikes) > 1:
                neuron_isis = np.diff(neuron_spikes)  # Keep in timesteps
                isis.extend(neuron_isis)
        
        if len(isis) > 0:
            avg_isi = np.mean(isis)
            print(f"{neuron_type} - Avg ISI: {avg_isi:.2f} timesteps")
            return avg_isi
        else:
            return None
    
    # Calculate ISIs for both populations
    exc_isi = calc_avg_isi(excitatory_raster, "Excitatory")
    inh_isi = calc_avg_isi(inhibitory_raster, "Inhibitory")
    
    # Create figure with appropriate size
    fig_width = 16  # Wide figure for time series
    fig_height = 8  # Reasonable height
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    
    # Time axis (showing actual timestep indices)
    time_axis = np.arange(start_idx, start_idx + time_steps)
    
    # Plot excitatory neurons with grid-based approach
    if excitatory_raster is not None:
        spike_times_e, neuron_ids_e = np.where(excitatory_raster)
        # Convert to actual timesteps
        actual_times = spike_times_e + start_idx
        
        # Use imshow for grid-like visualization with no gaps
        axes[0].imshow(excitatory_raster, aspect='auto', origin='lower', 
                      extent=[start_idx, start_idx + time_steps, 0, N_e],
                      cmap='Greys', interpolation='nearest')
        
        # Add ISI information to title
        title = f'Excitatory Neurons (N={N_e})'
        if exc_isi is not None:
            title += f' - Avg ISI: {exc_isi:.1f} timesteps'
        axes[0].set_title(title)
        axes[0].set_ylabel('Excitatory Neuron ID')
        axes[0].set_ylim(0, N_e)
    else:
        axes[0].text(0.5, 0.5, 'No excitatory spike data found', 
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Excitatory Neurons - NO DATA')
        axes[0].set_ylabel('Excitatory Neuron ID')
    
    # Plot inhibitory neurons with grid-based approach
    if inhibitory_raster is not None:
        spike_times_i, neuron_ids_i = np.where(inhibitory_raster)
        # Convert to actual timesteps
        actual_times = spike_times_i + start_idx
        
        # Use imshow for grid-like visualization with no gaps
        axes[1].imshow(inhibitory_raster, aspect='auto', origin='lower',
                      extent=[start_idx, start_idx + time_steps, 0, N_i],
                      cmap='Greys', interpolation='nearest')
        
        # Add ISI information to title
        title = f'Inhibitory Neurons (N={N_i})'
        if inh_isi is not None:
            title += f' - Avg ISI: {inh_isi:.1f} timesteps'
        axes[1].set_title(title)
        axes[1].set_ylabel('Inhibitory Neuron ID')
        axes[1].set_ylim(0, N_i)
    else:
        axes[1].text(0.5, 0.5, 'No inhibitory spike data found', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Inhibitory Neurons - NO DATA')
        axes[1].set_ylabel('Inhibitory Neuron ID')
    
    # Add time axis label
    xlabel = f'Time (timesteps: {start_idx} to {start_idx + time_steps})'
    axes[1].set_xlabel(xlabel)
    
    plt.tight_layout()
    
    if save_plot:
        # Extract filename from path for naming
        filename = os.path.basename(os.path.dirname(os.path.dirname(h5_path)))
        plt.savefig(os.path.join(saveloc, f'raster_plot_{filename}.pdf'))
        print(f"Raster plot saved to: {os.path.join(saveloc, f'raster_plot_{filename}.pdf')}")
    
    # plt.show()  # Comment out to match original behavior
    plt.close()
    
    return excitatory_raster, inhibitory_raster

##############################################################
##############################################################

def get_h5_files(backup_path):
    """Get paths to all result.h5 files from a specified backup directory."""
    # Get all timestamped folders
    all_folders = [f for f in os.listdir(backup_path) if os.path.isdir(os.path.join(backup_path, f))]
    date_folders = [f for f in all_folders if f.startswith('202') or f.startswith('sim')]
    date_folders.sort()  # Sort chronologically


    h5_files = []
    for folder in date_folders:
        h5_path = os.path.join(backup_path, folder, 'common', 'result.h5')
        if os.path.exists(h5_path):
            h5_files.append(h5_path)
            print(f"Found H5 file in: {folder}")

    return h5_files

def process_h5_file(file_path, starting_time_point, end_time_point):
    """
    Process a single .h5 file to extract raster data starting from a given time point.
    """
    try:
        h5 = tables.open_file(file_path, 'r')
        data = h5.root

        if data.__contains__('Spikes'):
            print(f"Looking at raster in file: {file_path}")
            last_spikes = data.c.stats.only_last_spikes[0]
            tmp_p_raster = data.Spikes[0, :, -last_spikes:]
            raster = tmp_p_raster[:, starting_time_point:end_time_point] if starting_time_point is not None else tmp_p_raster
            print(f"Raster shape for {file_path}: {raster.shape}")
            return raster
        else:
            print(f"No 'Spikes' data found in {file_path}.")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
    finally:
        try:
            h5.close()
        except:
            pass

def get_and_process_rasters(backup_path, starting_time_point):
    """
    Retrieve and process raster data from all .h5 files in a given backup path.
    """
    # Get list of .h5 files
    h5_files = get_h5_files(backup_path)
    print(f"\nFound {len(h5_files)} H5 files to analyze")

    all_rasters = []

    # Process each file
    for file_path in h5_files:
        print(f"\nProcessing: {file_path}")
        raster = process_h5_file(file_path, starting_time_point)
        if raster is not None:
            all_rasters.append(raster)

    # Combine all rasters into a single structure if needed
    if all_rasters:
        combined_raster = np.hstack(all_rasters)
        print(f"\nCombined raster shape: {combined_raster.shape}")
        return combined_raster
    else:
        print("No valid rasters processed.")
        return None
    
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

def correlationsOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, 
                saveloc=base_dir,
                pltname='analysis', state=''):
    """
    Fixed version of the optimized correlations function that matches original behavior.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag
    correlation_matrix = np.zeros((n_neurons, n_neurons))

    # Pre-compute the time series for all neurons
    series_early = raster[:, :t_length]  # All neurons, early timepoints
    series_late = raster[:, lag:lag + t_length]  # All neurons, later timepoints
    
    # Check for non-zero series
    any_early = np.any(series_early, axis=1)
    any_late = np.any(series_late, axis=1)
    valid_pairs = np.outer(any_early, any_late)
    
    # Where both series have data, compute correlation using np.corrcoef
    for i in range(n_neurons):
        for j in range(n_neurons):
            if valid_pairs[i, j]:
                correlation_matrix[i, j] = np.corrcoef(
                    series_early[i], 
                    series_late[j]
                )[0, 1]

    # Compute eigenvalues using faster method for symmetric matrices
    # eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        else:
            plt.show()
        plt.close()

    # Plot distribution of weights and fit exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Vectorized computation of sorted weights
        sorted_weights = -np.sort(-np.abs(correlation_matrix), axis=1)
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, 
                              p0=p0, bounds=bounds, maxfev=10000)
            exponential_params = popt
            
            plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                    label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        except:
            print("Warning: Could not fit exponential decay")
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        else:
            plt.show()
        plt.close()

    return max_eigenvalue, exponential_params
##############################################################
# D2 Analysis Functions
##############################################################

def determine_optimal_ar_order(time_series, max_lag=10, plot_diagnostics=True, save_plot=True, 
                              saveloc=base_dir, pltname='ar_order_selection'):
    """
    Determine optimal AR order for d2 calculation using ACF, PACF, AIC, and BIC.
    """
    time_series_centered = time_series - np.mean(time_series)
    
    if plot_diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(time_series_centered, lags=min(30, len(time_series_centered)//4), ax=axes[0])
        axes[0].set_title("Autocorrelation Function (ACF)")
        plot_pacf(time_series_centered, lags=min(20, len(time_series_centered)//5), ax=axes[1], method="ywm")
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_acf_pacf.pdf')
        plt.show()
    
    max_lag = min(max_lag, len(time_series_centered)//10)
    aic_values = []
    bic_values = []
    lags = list(range(1, max_lag + 1))
    
    for p in lags:
        try:
            model = AutoReg(time_series_centered, lags=p).fit()
            aic_values.append(model.aic)
            bic_values.append(model.bic)
        except:
            aic_values.append(np.inf)
            bic_values.append(np.inf)
    
    if plot_diagnostics:
        plt.figure(figsize=(8, 5))
        plt.plot(lags, aic_values, label='AIC', marker='o')
        plt.plot(lags, bic_values, label='BIC', marker='s')
        plt.xlabel('AR Order (p)')
        plt.ylabel('Criterion Value')
        plt.title('AIC and BIC for AR Model Order Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_model_selection.pdf')
        plt.show()
    
    optimal_p_aic = lags[np.argmin(aic_values)]
    optimal_p_bic = lags[np.argmin(bic_values)]
    
    print(f"Optimal AR order based on AIC: {optimal_p_aic}")
    print(f"Optimal AR order based on BIC: {optimal_p_bic}")
    
    recommended_order = optimal_p_bic
    
    return {
        'optimal_p_aic': optimal_p_aic,
        'optimal_p_bic': optimal_p_bic,
        'recommended_order': recommended_order,
        'aic_values': aic_values,
        'bic_values': bic_values,
        'lags': lags
    }

def compute_d2_analysis(raster, auto_select_order=True, manual_order=5, 
                       plot_diagnostics=True, save_plot=True, 
                       saveloc=base_dir, pltname='d2_analysis'):
    """
    Compute d2 correlation dimension with automatic or manual AR order selection.
    """
    activity = np.sum(raster, axis=0)
    print(f"Computing d2 for activity time series of length {len(activity)}")
    
    if auto_select_order:
        print("Automatically determining optimal AR order...")
        ar_results = determine_optimal_ar_order(activity, 
                                              plot_diagnostics=plot_diagnostics,
                                              save_plot=save_plot,
                                              saveloc=saveloc,
                                              pltname=f'{pltname}_ar_selection')
        p_order = ar_results['recommended_order']
        print(f"Using AR order: {p_order}")
    else:
        p_order = manual_order
        ar_results = None
        print(f"Using manual AR order: {p_order}")
    
    try:
        d2_value = crt.calc_d2_KLr(activity, p_order)
        print(f"d2 correlation dimension: {d2_value:.4f}")
        
        if d2_value < 0.8:
            interpretation = "Subcritical regime (d2 < 0.8)"
        elif 0.8 <= d2_value <= 1.2:
            interpretation = "Near-critical regime (0.8 ≤ d2 ≤ 1.2)"
        else:
            interpretation = "Supercritical/chaotic regime (d2 > 1.2)"
        
        print(f"Interpretation: {interpretation}")
        
    except Exception as e:
        print(f"Error computing d2: {e}")
        d2_value = np.nan
        interpretation = "Computation failed"
    
    return {
        'd2_value': d2_value,
        'ar_order': p_order,
        'interpretation': interpretation,
        'ar_selection_results': ar_results,
        'activity_length': len(activity)
    }

##############################################################

def correlationsMoreOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=True, save_plot=True, 
                saveloc=base_dir,
                pltname='analysis', state=''):
    """
    Vectorized version of the correlations function using numpy operations.
    """
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag

    # Pre-compute the time series
    series_early = raster[:, :t_length]
    series_late = raster[:, lag:lag + t_length]
    
    # Compute means and standard deviations for all neurons at once
    means_early = np.mean(series_early, axis=1, keepdims=True)
    means_late = np.mean(series_late, axis=1, keepdims=True)
    
    stds_early = np.std(series_early, axis=1, keepdims=True)
    stds_late = np.std(series_late, axis=1, keepdims=True)
    
    # Normalize the data in a vectorized way
    # Handle zero standard deviations to avoid division by zero
    valid_early = stds_early > 0
    valid_late = stds_late > 0
    
    # Initialize normalized arrays
    norm_early = np.zeros_like(series_early)
    norm_late = np.zeros_like(series_late)
    
    # Normalize only where std > 0
    norm_early[valid_early.ravel()] = ((series_early - means_early) / stds_early)[valid_early.ravel()]
    norm_late[valid_late.ravel()] = ((series_late - means_late) / stds_late)[valid_late.ravel()]
    
    # Compute correlation matrix using matrix multiplication
    correlation_matrix = np.zeros((n_neurons, n_neurons))
    valid_pairs = np.outer(valid_early.ravel(), valid_late.ravel())
    
    # Only compute correlations for valid pairs
    if np.any(valid_pairs):
        # Use matrix multiplication for vectorized correlation computation
        correlation_matrix[valid_pairs] = (
            np.dot(norm_early, norm_late.T)[valid_pairs] / t_length
        )

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    # Plot eigenvalue spectrum if requested
    if plot_eigenvalues:
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum in the Complex Plane for {state}')
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_evalues.pdf')
        plt.close()

    # Plot distribution of weights and fit exponential decay if requested
    exponential_params = None
    if plot_weights:
        # Vectorized computation of sorted weights
        sorted_weights = -np.sort(-np.abs(correlation_matrix), axis=1)
        average_sorted_weights = np.mean(sorted_weights, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(average_sorted_weights, 'o-', label='Average sorted weights')
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        x_data = np.arange(len(average_sorted_weights))
        p0 = [max(average_sorted_weights), 0.1, min(average_sorted_weights)]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(exp_decay, x_data, average_sorted_weights, 
                              p0=p0, bounds=bounds, maxfev=10000)
            exponential_params = popt
            
            plt.plot(x_data, exp_decay(x_data, *popt), 'r--', 
                    label=f'Exponential fit: a*exp(-b*x) + c\n(a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f})')
        except:
            print("Warning: Could not fit exponential decay")
        
        plt.xlabel('Index Node')
        plt.ylabel('Weight')
        plt.title(f'Distribution of {state} Weights and Exponential Fit')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            plt.savefig(f'{saveloc}/{pltname}_weights.pdf')
        plt.close()

    return max_eigenvalue, exponential_params

def scaling_plots(Result, burst, burstMin, burstMax, alpha, T, tMin, tMax,
                  beta, TT, Sm, sigma, fit_sigma, pltname, saveloc, p_val_b, p_val_t):
    # burst PDF
    burstMax = int(burstMax)
    burstMin = int(burstMin)
    fig1, ax1 = plt.subplots(nrows = 1, ncols = 3, figsize = [10, 6])
    pdf = np.histogram(burst, bins = np.arange(1, np.max(burst) + 2))[0]
    p = pdf / np.sum(pdf)
    ax1[0].plot(np.arange(1, np.max(burst) + 1), p, marker = 'o',
                markersize = 5, fillstyle = 'none', mew = .5,
                linestyle = 'none', color = 'darkorchid', alpha = 0.75)
    ax1[0].plot(np.arange(1, np.max(burst) + 1)[burstMin:burstMax],
                p[burstMin:burstMax], marker = 'o', markersize = 5,
                fillstyle = 'full', linestyle = 'none', color = 'darkorchid',
                alpha = 0.75)
    ax1[0].set_yscale('log')
    ax1[0].set_xscale('log')

    x = np.arange(burstMin, burstMax + 1)
    y = (np.size(np.where(burst == burstMin + 6)[0]) / np.power(burstMin + 6, -alpha)) *\
        np.power(x, -alpha)
    y = y / np.sum(pdf)

    ax1[0].plot(x, y, color = '#c5c9c7')
    ax1[0].set_xlabel('AVsize')
    ax1[0].set_ylabel('PDF(S)')
    ax1[0].set_title('AVsize PDF, ' + str(np.round(alpha, 3)))
    if p_val_b is not None:
        ax1[0].text(10, .1, f'p_val = {p_val_b}')

    # time pdf
    tdf = np.histogram(T, bins = np.arange(1, np.max(T) + 2))[0]
    t = tdf / np.sum(tdf)
    ax1[1].plot(np.arange(1, np.max(T) + 1), t, marker = 'o',
                markersize = 5, fillstyle = 'none', mew = .5,
                linestyle = 'None', color = 'mediumseagreen', alpha = 0.75)
    ax1[1].plot(np.arange(1, np.max(T) + 1)[tMin:tMax], t[tMin:tMax],
                marker = 'o', markersize = 5, fillstyle = 'full',
                linestyle = 'none', color = 'mediumseagreen', alpha = 0.75)
    ax1[1].set_yscale('log')
    ax1[1].set_xscale('log')
    sns.despine()

    x = np.arange(tMin, tMax + 1)
    y = np.size(np.where(T == tMin)) / (np.power(tMin, -beta)) *\
        np.power(x, -beta)
    y = y / np.sum(tdf)
    ax1[1].plot(x, y, color = '#c5c9c7')
    ax1[1].set_xlabel('AVduration')
    ax1[1].set_ylabel('PDF(D)')
    ax1[1].set_title('AVdura PDF, ' + str(np.round(beta, 3)))
    if p_val_t is not None:
        ax1[1].text(10, .1, f'p_val = {p_val_t}')


    # figure out how to plot shuffled data

    # scaling relation
    i = np.where(TT == tMax)  # just getting the last value we use, getting rid of the hard codes
    ax1[2].plot(TT, ((np.power(TT, sigma) / np.power(TT[7], sigma)) * Sm[7]),
                label = 'pre', color = '#4b006e')
    ax1[2].plot(TT, (np.power(TT, fit_sigma[0]) / np.power(TT[7], fit_sigma[0]) * Sm[7]),
                'b', label = 'fit', linestyle = '--', color = '#826d8c')
    ax1[2].plot(TT, Sm, 'o', color = '#fb7d07', markersize = 5, mew = .5,
                fillstyle = 'none', alpha = 0.75)

    locs = np.where(np.logical_and(TT < tMax, TT > tMin))[0]

    ax1[2].plot(TT[locs], Sm[locs], 'o', markersize = 5, mew = .5,
                color = '#fb7d07', fillstyle = 'full', alpha = 1)
    ax1[2].set_xscale('log')
    ax1[2].set_yscale('log')
    ax1[2].set_ylabel('<S>')
    ax1[2].set_xlabel('Duration')
    ax1[2].set_title('Difference = ' + str(np.round(Result['df'], 3)))

    plt.tight_layout()
    plt.legend()
    # plt.savefig(saveloc + "/" + pltname + 'scaling_relations' + '.svg', format='svg')
    savefigpath = op.join(saveloc , pltname + 'scaling_relations' + '.svg')
    print("savefigpath ", savefigpath)
    plt.savefig(savefigpath, format='svg')

    return fig1


def AV_analysis(burst, T, flag = 1, bm = 20, tm = 10, nfactor_bm=0, nfactor_tm=0,
                nfactor_bm_tail=0.8, nfactor_tm_tail=1.0, none_fact=40,
                max_time=7200, verbose = True, exclude = False, exclude_burst = 50, 
                exclude_time = 20, exclude_diff_b=20, exclude_diff_t=10, plot=True, pltname='', saveloc=''):
    #print('VERBOSE: ', verbose)

    if bm is None:
        if verbose:
            print('none_fact ', none_fact)
        bm = int(np.max(burst)/none_fact)

    Result = {}
    burstMax, burstMin, alpha = \
        ex.EXCLUDE(burst[burst < np.power(np.max(burst), nfactor_bm_tail)], bm,
                   nfactor=nfactor_bm, verbose = verbose)
    idx_burst = \
        np.where(np.logical_and(burst <= burstMax, burst >= burstMin))[0]
    # print("burst[idx_burst] ", burst[idx_burst])
    # print("idx_burst ", idx_burst)
    if verbose:
        print("alpha ", alpha)
        print("burst min: ", burstMin)
        print("burst max:", burstMax, flush=True)

    Result['burst'] = burst
    Result['alpha'] = alpha
    Result['xmin'] = burstMin
    Result['xmax'] = burstMax

    Result['P_burst'] = None
    Result['EX_b'] = False
    if exclude:
        if burstMin > exclude_burst or (burstMax-burstMin)<exclude_diff_b:
            print(f'This block excluded for burst: xmin {burstMin} diff: {burstMax-burstMin}')
            Result['EX_b'] = True
    
    if flag == 2 and not Result['EX_b']:
        if verbose:
            print("About to do the p val test for burst")
        # pvalue test
        Result['P_burst'], ks, hax_burst, ptest_bmin = \
            pv.pvaluenew(burst[idx_burst], alpha, burstMin, nfactor=nfactor_bm,
                         max_time=max_time, verbose = verbose)

    if tm is None:
        if verbose:
            print('none_fact ', none_fact)
        tm = int(np.max(T)/none_fact)

    # print("tMax, tMin, beta = ex.EXCLUDE(T, tm)")
    # ckbn tMax, tMin, beta = ex.EXCLUDE(T, tm, nfactor=nfactor_tm)
    tMax, tMin, beta = \
        ex.EXCLUDE(T[T < np.power(np.max(T), nfactor_tm_tail)], tm,
                   nfactor=nfactor_tm, verbose = verbose)
    idx_time = np.where(np.logical_and(T >= tMin, T <= tMax))[0]

    if verbose:
        print("beta ", beta)
        print(f'time min: {tMin}')
        print(f'time max: {tMax}', flush=True)

    Result['T'] = T
    Result['beta'] = beta
    Result['tmin'] = tMin
    Result['tmax'] = tMax

    Result['P_t'] = None
    Result['EX_t'] = False
    if exclude:
        if tMin > exclude_time or (tMax-tMin)<exclude_diff_t:
            print(f'This block excluded for time: tmin {tMin} diff: {tMax-tMin}')
            Result['EX_t'] = True

    if flag == 2 and not Result['EX_t'] and not Result['EX_b']:
        if verbose:
            print("About to do the p val test for time")
        # pvalue for time
        Result['P_t'], ks, hax_time, ptest_tmin = \
            pv.pvaluenew(T[idx_time], beta, tMin, nfactor=nfactor_tm,
                         max_time=max_time, verbose=verbose)

    TT = np.arange(1, np.max(T) + 1)
    Sm = []
    for i in np.arange(0, np.size(TT)):
        Sm.append(np.mean(burst[np.where(T == TT[i])[0]]))
    Sm = np.asarray(Sm)
    Loc = np.where(Sm > 0)[0]
    TT = TT[Loc]
    Sm = Sm[Loc]

    # ckbndnt
    fit_sigma = \
        np.polyfit(np.log(TT[np.where(np.logical_and(TT > tMin,
                                                     TT < tMax))[0]]),
                   np.log(Sm[np.where(np.logical_and(TT > tMin,
                                                     TT < tMax))[0]]), 1)
    sigma = (beta - 1) / (alpha - 1)
    if verbose:
        print("fit_sigma ", fit_sigma)
        print("sigma ", sigma, flush=True)


    Result['pre'] = sigma
    Result['fit'] = fit_sigma
    Result['df'] = np.abs(sigma - fit_sigma[0])
    Result['TT'] = TT
    Result['Sm'] = Sm
    Result['burst_cdf'] = None
    Result['time_cdf'] = None
    Result['scaling_relation_plot'] = None 

    if plot:
        fig1 = scaling_plots(Result, burst, burstMin, burstMax, alpha, T,
                             tMin, tMax, beta, TT, Sm, sigma, fit_sigma,
                             pltname, saveloc, Result['P_burst'], Result['P_t'])
        if flag == 2 and not Result['EX_t'] and not Result['EX_b']:
            hax_burst.axes[0].set_xlabel('Size (S)', fontsize=16)
            hax_burst.axes[0].set_ylabel('Prob(size < S)', fontsize=16)
            # hax_burst.savefig(saveloc + "/" + pltname + 'pvalue_burst' + '.svg', format='svg')
            savefigpathb = op.join(saveloc,  pltname + 'pvalue_burst' + '.svg')
            print("savefigpathb ", savefigpathb)
            hax_burst.savefig(savefigpathb, format='svg')

            hax_time.axes[0].set_xlabel('Duration (D)', fontsize=16)
            hax_time.axes[0].set_ylabel('Prob(size < D)', fontsize=16)
            # hax_time.savefig(saveloc + "/" + pltname + 'pvalue_time' +  '.svg', format='svg')
            savefigpatht = op.join(saveloc, pltname + 'pvalue_time' + '.svg')
            print("savefigpatht ", savefigpatht)
            hax_burst.savefig(savefigpatht, format='svg')
            Result['burst_cdf'] = hax_burst
            Result['time_cdf'] = hax_time
        Result['scaling_relation_plot'] = fig1
        plt.close('all')

    return Result

# ============================================
# CUSTOM AVALANCHE ANALYSIS PARAMETERS
# ============================================

# Custom bounds for avalanche size distribution fitting
AVALANCHE_PARAMS = {
    # Size distribution parameters
    'size_xmin': None,        # Set to value like 10, or None for auto
    'size_xmax': None,        # Set to value like 1000, or None for auto
    'size_bm': 20,           # Minimum bin for size fitting
    'size_nfactor': 0,       # Adjustment factor for xmin selection
    'size_tail_cutoff': 0.8, # Tail cutoff factor (0.8 = use up to S^0.8)
    
    # Duration distribution parameters  
    'duration_xmin': None,    # Set to value like 5, or None for auto
    'duration_xmax': None,    # Set to value like 500, or None for auto
    'duration_tm': 10,       # Minimum bin for duration fitting
    'duration_nfactor': 0,   # Adjustment factor for xmin selection
    'duration_tail_cutoff': 1.0, # Tail cutoff factor
    
    # Exclusion criteria (set these to exclude bad fits)
    'exclude_burst_min': 50,  # Exclude if xmin > this
    'exclude_time_min': 20,   # Exclude if xmin > this
    'exclude_burst_diff': 20, # Exclude if xmax-xmin < this
    'exclude_time_diff': 10,  # Exclude if xmax-xmin < this
    
    # Other parameters
    'perc_threshold': 0.25,   # Percentile threshold for get_avalanches
    'const_threshold': None,  # Or set specific value like 5
    'none_factor': 40,       # Factor for auto bm/tm calculation
}

# Function to override automatic xmin/xmax selection
def apply_custom_bounds(burst, T, params):
    """Apply custom xmin/xmax if specified in params"""
    # Make copies to avoid modifying original data
    burst_filtered = burst.copy()
    T_filtered = T.copy()
    
    # Apply size bounds if specified
    if params['size_xmin'] is not None or params['size_xmax'] is not None:
        mask = np.ones(len(burst), dtype=bool)
        if params['size_xmin'] is not None:
            mask &= burst >= params['size_xmin']
        if params['size_xmax'] is not None:
            mask &= burst <= params['size_xmax']
        burst_filtered = burst[mask]
        T_filtered = T[mask]
        print(f"Applied custom size bounds: {params['size_xmin']} to {params['size_xmax']}")
        print(f"Avalanches after filtering: {len(burst_filtered)}/{len(burst)}")
    
    return burst_filtered, T_filtered

def analyze_avalanches_custom_bounds(burst, T, manual_bounds=None, plot_results=True, save_dir=base_dir):
    """
    Analyze avalanches with option for manual xmin/xmax override
    
    Parameters:
    -----------
    manual_bounds : dict, optional
        Dictionary with 'size_xmin', 'size_xmax', 'duration_xmin', 'duration_xmax'
    """
    
    if manual_bounds is None:
        # Use automatic detection
        print("Using automatic xmin/xmax detection...")
        result = cr.AV_analysis(
            burst=burst, T=T, **AVALANCHE_PARAMS,
            plot=plot_results, saveloc=save_dir
        )
    else:
        # Manual override
        print("Using manual bounds:")
        print(f"  Size: {manual_bounds.get('size_xmin', 'auto')} to {manual_bounds.get('size_xmax', 'auto')}")
        print(f"  Duration: {manual_bounds.get('duration_xmin', 'auto')} to {manual_bounds.get('duration_xmax', 'auto')}")
        
        # First, get automatic results to extract the exponents
        auto_result = cr.AV_analysis(
            burst=burst, T=T, flag=1, plot=False,
            bm=AVALANCHE_PARAMS['size_bm'],
            tm=AVALANCHE_PARAMS['duration_tm']
        )
        
        # Now manually set bounds and recalculate
        result = {}
        
        # Size distribution with manual bounds
        if 'size_xmin' in manual_bounds and 'size_xmax' in manual_bounds:
            xmin = manual_bounds['size_xmin']
            xmax = manual_bounds['size_xmax']
            
            # Filter data
            mask = (burst >= xmin) & (burst <= xmax)
            burst_filtered = burst[mask]
            
            # Fit power law in this range
            from scipy.stats import linregress
            bins = np.logspace(np.log10(xmin), np.log10(xmax), 50)
            hist, bin_edges = np.histogram(burst_filtered, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Only fit non-zero bins
            nonzero = hist > 0
            if np.sum(nonzero) > 2:
                slope, intercept, _, _, _ = linregress(
                    np.log10(bin_centers[nonzero]), 
                    np.log10(hist[nonzero])
                )
                result['alpha'] = -slope
                result['xmin'] = xmin
                result['xmax'] = xmax
            else:
                result['alpha'] = auto_result.get('alpha', np.nan)
        else:
            # Use automatic results
            result['alpha'] = auto_result.get('alpha', np.nan)
            result['xmin'] = auto_result.get('xmin', np.nan)
            result['xmax'] = auto_result.get('xmax', np.nan)
        
        # Similar for duration...
        # (Add similar code for duration if needed)
        
        result['burst'] = burst
        result['T'] = T
        
        # Generate plots if requested
        if plot_results:
            scaling_plots(
                result, burst, 
                result.get('xmin', 10), result.get('xmax', 1000),
                result.get('alpha', 1.5),
                T, 5, 500, 2.0,  # You'll need to add duration params
                np.arange(1, max(T)+1), np.ones(max(T)),  # Placeholder
                1.5, [1.5], f'manual_bounds_', save_dir,
                None, None
            )
    
    return result

""" 
#
#
# This is the main driving part of this code
#
#
"""

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
    if(len(h5_files) > 0):

        # Replace the section starting from around line 2240 to 2460 with this:

        # Initialize lists to store individual values from each file
        individual_susc = []
        individual_rho = []
        individual_cv = []
        individual_br_method_1 = []
        individual_br_method_2 = []
        individual_priesman_br = []
        individual_d2 = []
        individual_ar_order = []

        # Process each H5 file
        for file_path in h5_files:
            print(f"\nProcessing: {file_path}")

            try:
                # Load data
                h5 = tables.open_file(file_path, 'r')
                data = h5.root

                # Debug prints to understand the structure
                print("File structure:")
                h5.list_nodes('/')

                # Try different methods to read the data
                pickle_dir = None
                for node in h5.iter_nodes(h5.root.c, 'Array'):
                    if node.name == 'logfilepath':
                        pickle_dir = str(node[0])
                        break

                if data.__contains__('Spikes'):

                    ######################
                    # Raster plot (last_n_spikes)
                    ######################
                    print("Looking at raster in file: ", file_path)
                    last_spikes = data.c.stats.only_last_spikes[0]
                    print("Last Spike (from stats): ", last_spikes)

                    # Check actual data dimensions
                    actual_data_length = data.Spikes.shape[2]
                    print("Actual data length: ", actual_data_length)

                    # Define the actual end point using real data dimensions
                    actual_end = min(end_time_point, actual_data_length) if end_time_point else actual_data_length

                    print(f"Time window: {starting_time_point} to {actual_end}")
                    print(f"Requested end point: {end_time_point}")
                    print(f"Available data ends at: {actual_data_length}")

                    # Load ONLY the time window directly from HDF5
                    raster = data.Spikes[0, :, starting_time_point:actual_end]         
                    print(f"Raster shape for {file_path}: {raster.shape}")
                    print(f"Time window: {starting_time_point} to {actual_end}")
                    print(f"Requested end point: {end_time_point}")
                    print(f"Available data ends at: {last_spikes}")
                    print(f"Actually analyzing {raster.shape[1]} time steps")
                    print(f"Percentage of requested window analyzed: {(actual_end - starting_time_point) / (end_time_point - starting_time_point) * 100:.1f}%")
                    
                    if True:  # Set to True to enable plotting
                        exc_raster, inh_raster = extract_and_plot_rasters(
                        file_path, 
                        time_window=(starting_time_point, actual_end),
                        save_plot=True,
                        saveloc=base_dir
                    )   
                    
                    ######################
                    # Per raster stats
                    ######################
                    single_susc = susceptibility(raster)
                    susc.append(single_susc)
                    individual_susc.append(single_susc)
                    print("-- Susceptibility: ", single_susc)

                    single_rho = rho(raster)
                    rho_values.append(single_rho)
                    individual_rho.append(single_rho)
                    print("-- Rho: ", single_rho)

                    single_cv = cv(raster)
                    cv_values.append(single_cv)
                    individual_cv.append(single_cv)
                    print("-- CV: ", single_cv)
                    
                    single_br_method_1 = branchparam(raster, lverbose=0)
                    br_method_1.append(single_br_method_1)
                    individual_br_method_1.append(single_br_method_1)
                    print("-- Branching param [method 1]: ", single_br_method_1)

                    max_eig, exp_params = correlationsMoreOptimized(raster, lag=1, plot_eigenvalues=False, plot_weights=False, 
                                                   save_plot=False, saveloc='', pltname='analysis', state = '')
                    print(f"Maximum eigenvalue: {max_eig}")
                    pearson_kappa.append(max_eig)

                    ######################
                    # D2 Analysis 
                    ######################
                    print("Computing d2 correlation dimension...")
                    d2_results = compute_d2_analysis(raster, 
                                                    auto_select_order=True,
                                                    manual_order=5,
                                                    plot_diagnostics=False,
                                                    save_plot=True,
                                                    saveloc=base_dir,
                                                    pltname=f'd2_analysis_mu_{mu}')

                    single_d2 = d2_results['d2_value']
                    single_ar_order = d2_results['ar_order']

                    print(f"-- d2 correlation dimension: {single_d2:.4f}")
                    print(f"-- AR order used: {single_ar_order}")
                    print(f"-- {d2_results['interpretation']}")

                    d2_values.append(single_d2)
                    ar_orders.append(single_ar_order)
                    individual_d2.append(single_d2)
                    individual_ar_order.append(single_ar_order)

                    single_results = cr.get_avalanches(
                    raster, 
                    perc=AVALANCHE_PARAMS['perc_threshold'],
                    ncells=-1, 
                    const_threshold=AVALANCHE_PARAMS['const_threshold']
                )

                    # Add custom filtering if bounds are specified
                    if AVALANCHE_PARAMS['size_xmin'] or AVALANCHE_PARAMS['size_xmax']:
                        size_mask = np.ones(len(single_results['S']), dtype=bool)
                        
                        if AVALANCHE_PARAMS['size_xmin']:
                            size_mask &= single_results['S'] >= AVALANCHE_PARAMS['size_xmin']
                            
                        if AVALANCHE_PARAMS['size_xmax']:
                            size_mask &= single_results['S'] <= AVALANCHE_PARAMS['size_xmax']
                        
                        # Apply the mask
                        for key in ['S', 'T']:
                            if key in single_results:
                                single_results[key] = single_results[key][size_mask]
                        
                        if 'shapes' in single_results:
                            single_results['shapes'] = [single_results['shapes'][i] for i in np.where(size_mask)[0]]
                        
                            print(f"After custom filtering: {np.sum(size_mask)}/{len(size_mask)} avalanches kept")
                        
                    if 'S' in single_results and len(single_results['S']) > 0:
                        print(f"[INFO] Found {len(single_results['S'])} avalanches")
                        
                        # Concatenating avalanche statistics
                        all_burst = np.concatenate((all_burst, single_results['S']))
                        all_T = np.concatenate((all_T, single_results['T']))
                        
                        # Since cr.get_avalanches doesn't return shapes, use our own function
                        print("[INFO] Computing avalanche shapes using find_avalanches...")
                        manual_shapes = find_avalanches(raster)
                        
                        if len(manual_shapes) > 0:
                            all_shapes.extend(manual_shapes)
                            print(f"[INFO] Found {len(manual_shapes)} avalanche shapes")
                            
                            # Calculate branching ratios using the shapes
                            try:
                                single_results_br = branching_ratios(manual_shapes)
                                single_br_method_2 = single_results_br['avgBR']
                                print(f"-- Branching param [method 2]: {single_br_method_2}")
                            except Exception as e:
                                print(f"[WARNING] Error calculating branching ratios: {e}")
                                single_br_method_2 = np.nan
                        else:
                            print("[WARNING] No avalanche shapes found")
                            single_br_method_2 = np.nan
                    else:
                        print("[INFO] No avalanches found")
                        single_br_method_2 = 0

                    br_method_2.append(single_br_method_2)
                    individual_br_method_2.append(single_br_method_2)

                    activity_array = np.sum(raster, axis=0) - rho(raster)
                    single_priesman_br = calc_BR(activity_array, 100, 0.001, os.path.join(base_dir, f'priesman_br_mu_{mu}.pdf'))
                    br_priesman.append(single_priesman_br)
                    individual_priesman_br.append(single_priesman_br)
                    print("-- Branching param [method Priesman]: ", single_priesman_br)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
            finally:
                try:
                    h5.close()
                except:
                    pass
        
        # Save individual metrics to a `.txt` file AFTER processing all files
        text_output_path = os.path.join(base_dir, f'Stats_Mu_{mu:.2f}.txt')
        with open(text_output_path, 'w') as f:
            f.write(f"Metrics Summary for Mu = {mu:.2f}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Number of files processed: {len(individual_susc)}\n")
            f.write("-" * 60 + "\n")
            
            if individual_susc:  # Check if we have any data
                f.write(f"Individual Susceptibility Values: {individual_susc}\n")
                f.write(f"Mean Susceptibility: {np.mean(individual_susc):.4f}\n")
                f.write(f"Individual Rho Values: {individual_rho}\n")
                f.write(f"Mean Rho: {np.mean(individual_rho):.4f}\n")
                f.write(f"Individual CV Values: {individual_cv}\n")
                f.write(f"Mean CV: {np.mean(individual_cv):.4f}\n")
                f.write(f"Individual Branching Ratio (Method 1 - branchparam) Values: {individual_br_method_1}\n")
                f.write(f"Mean Branching Ratio (Method 1 - branchparam): {np.mean(individual_br_method_1):.4f}\n")
                f.write(f"Individual Branching Ratio (Method 2 - get_avalanches) Values: {individual_br_method_2}\n")
                f.write(f"Mean Branching Ratio (Method 2 - get_avalanches): {np.mean(individual_br_method_2):.4f}\n")
                f.write(f"Individual Branching Ratio (Method 3 - Priesman) Values: {individual_priesman_br}\n")
                f.write(f"Mean Branching Ratio (Method 3 - Priesman): {np.mean(individual_priesman_br):.4f}\n")
                f.write(f"Mean Pearson Kappa coefficient: {np.mean(pearson_kappa):.4f}\n")
                f.write(f"Individual d2 Values: {individual_d2}\n")
                f.write(f"Mean d2: {np.mean(individual_d2):.4f}\n")
                f.write(f"Individual AR Orders: {individual_ar_order}\n")
                f.write(f"Mean AR Order: {np.mean(individual_ar_order):.1f}\n")
            else:
                f.write("No valid data processed.\n")
            
            f.write("-" * 60 + "\n")
            print(f"Metrics for Mu = {mu:.2f} saved to: {text_output_path}")

        # Continue with the rest of the code as before...

        overall_susc.append(np.mean(susc))
        overall_rho_values.append(np.mean(rho_values))
        overall_cv_values.append(np.mean(cv_values))
        overall_br_method_1.append(np.mean(br_method_1))
        overall_br_method_2.append(np.mean(br_method_2))
        overall_br_priesman.append(np.mean(br_priesman))
        mu_values.append(mu)
        overall_pearson_kappa.append(np.mean(pearson_kappa))
        overall_d2_values.append(np.mean(d2_values))
        overall_d2_ar_orders.append(np.mean(ar_orders))

        # Save individual values to CSV before moving to avalanche analysis
        detailed_results = []
        for file_idx in range(len(susc)):
            detailed_results.append({
                'Mu': mu,
                'File_Index': file_idx,
                'Susceptibility': susc[file_idx],
                'Rho': rho_values[file_idx],
                'CV': cv_values[file_idx],
                'Branching_Ratio_Method_1': br_method_1[file_idx],
                'Branching_Ratio_Method_2': br_method_2[file_idx],
                'Branching_Ratio_Priesman': br_priesman[file_idx],
                'Pearson_Kappa': pearson_kappa[file_idx],
                'D2_Correlation_Dimension': d2_values[file_idx],
                'D2_AR_Order': ar_orders[file_idx]
            })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(base_dir, f'Individual_Stats_Mu_{mu:.2f}.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Individual file results for Mu = {mu:.2f} saved to: {detailed_csv_path}")

        # Combine all rasters into a single structure if needed
        #if all_rasters:
        if True:
            #combined_raster = np.hstack(all_rasters)
            #print(f"\nCombined raster shape: {combined_raster.shape}")

            # Aggregate avalanches 
            #results = cr.get_avalanches(combined_raster, perc=0.25, ncells=-1, const_threshold=None)

            # compute some stats on avalanche shapes
            avg_results = None
            #avg_results = avshapecollapse(results['shapes'], results['T'], method=None, args=(),plot_flag=True, save_flag=True, filename=f'D:\\Users\\seaco\\SORN\\backup\\test_single\\sanity_check\\AvShapeCollapse_mu={mu}')
            # -> avg_results = avshapecollapse(all_shapes, all_T, method=None, args=(),plot_flag=True, save_flag=True, filename=f'D:\\Users\\seaco\\SORN\\backup\\backup\\test_single\\sanity_check\\AvShapeCollapse_mu={mu}')

            # Handle the results with error checking
            if avg_results is not None:
                av_collapse_exponent.append(avg_results['exponent'])
                av_collapse_secondDer.append(avg_results['secondDer'])
                av_collapse_range.append(avg_results['range'])
                av_collapse_errors.append(avg_results['errors'])
                av_collapse_coefficients.append(avg_results['coefficients'])
                if avg_results['errors'] and avg_results['errors'][-1]:  # Check if errors list is not empty
                    min_error = np.min(avg_results['errors'][-1])
                else:
                    min_error = np.nan  # or some other default value
                av_collapse_min_error.append(min_error)
            else:
                # Append default values when no results are available
                av_collapse_exponent.append(np.nan)  # or None, depending on what you prefer
                av_collapse_secondDer.append(np.nan)
                av_collapse_range.append([])
                av_collapse_errors.append([])
                av_collapse_coefficients.append(None)
                av_collapse_min_error.append(np.nan)
                print(f"[WARNING] No avalanche shape collapse results available for mu={mu}")

            # Plot log-log avalanche stats and crackling noise graph
            # scaling_plots(results['S'], results['T'], pltname=f'scaling_plot_mu={mu}',saveloc=base_dir, show_plot=False, plot_type='pdf')
            #-> scaling_plots(all_burst, all_T, pltname=f'scaling_plot_mu={mu}', saveloc=base_dir, show_plot=False, plot_type='pdf')
            
            # Compute branchig ratio using all aggregate avalanches
            # results_br = branching_ratios(results['shapes'])
            #-> results_br = branching_ratios(all_shapes)
            #-> print("-- Combined Branching param [method 2]: ", results_br['avgBR'])

        else:
            print("No valid rasters processed.")

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
    (overall_pearson_kappa, 'Pearson $\\kappa$'),
    (overall_d2_values, 'D2 Correlation Dimension'),
    (av_collapse_exponent, 'Av_Collapse_exponent 1/($\\sigma$$\\nu$z)'),
    (av_collapse_min_error, 'Av_Collapse_error')
    (av_collapse_exponent, 'Av_Collapse_exponent 1/($\\sigma$$\\nu$z)'),
    (av_collapse_min_error, 'Av_Collapse_error'),
    (overall_av_alpha, 'AV Alpha (size exponent)'),
    (overall_av_beta, 'AV Beta (duration exponent)'),
    (overall_av_df, 'AV Scaling Difference')
]

# Preprocessing: Remove invalid data points (e.g., NaN)
for idx, (data, title) in enumerate(plotting_data):
    if idx < len(axes):
        ax = axes[idx]
        
        # Convert to NumPy arrays for filtering
        mu_clean = np.array(mu_values)
        data_clean = np.array(data)
        
        # Remove NaN or None values
        valid_mask = ~np.isnan(data_clean) & ~np.isnan(mu_clean)
        mu_clean = mu_clean[valid_mask]
        data_clean = data_clean[valid_mask]
        
        # Plot cleaned data
        ax.plot(mu_clean, data_clean, label=title, **plot_params)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel(r'$\mu$', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(frameon=True, fontsize=8)

# Remove any unused subplot slots
for idx in range(len(plotting_data), len(axes)):
    fig.delaxes(axes[idx])

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.2, wspace=0.3)  # Reduced hspace for less vertical space between rows

# Save the entire figure as a PDF with high DPI
output_path = os.path.join(base_dir, 'All_Overall_Stats.pdf')
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')

# Show the plots
plt.show()

print(f"Aggregated plot saved as a PDF at {output_path}")

