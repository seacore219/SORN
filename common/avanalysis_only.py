#!/usr/bin/env python3
"""
Simplified Avalanche Analysis Code
Focuses exclusively on avalanche detection and analysis
P-value testing removed
"""

print("[INFO] Starting import of libraries...")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables
from scipy.stats import linregress
from scipy.optimize import curve_fit
import time
from copy import deepcopy as cdc
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy.sparse import csr_matrix
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Memory management
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
gc.collect()
tables.file._open_files.close_all()
print("[INFO] Libraries imported successfully.")

# ============================================
# CONFIGURATION
# ============================================
base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\SORN_PC\\backup\\delpapa_input\\batch_hip0.06_n6_ps1'
starting_time_point = 6000000
end_time_point = 8000000  # Set to None for full length

# Avalanche analysis parameters
AVALANCHE_PARAMS = {
    'perc_threshold': 0.1,   # Percentile threshold for avalanche detection
    'const_threshold': None,  
    'size_bm': 10,           # Increase to start fitting at larger sizes
    'size_nfactor': 0,       # Positive value to shift xmin higher
    'size_tail_cutoff': 0.66, # Decrease to cut off more of the tail
    'duration_tm': 3,       
    'duration_nfactor': 0,   
    'duration_tail_cutoff': 0.6,
    'exclude_burst_min': 18,  # Minimum xmin value - increase if you want to force higher xmin
    'exclude_time_min': 10,  
    'exclude_burst_diff': 12,  # Minimum range (xmax-xmin) - increase for wider fitting range
    'exclude_time_diff': 10, 
    'none_factor': 40,
}

# ============================================
# AVALANCHE DETECTION FUNCTIONS
# ============================================

def get_avalanches(data, perc=0.25, ncells=-1, const_threshold=None):
    """
    Find avalanches in spike data
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
        'Size': np.asarray(burst),
        'Duration': T,
        'shapes': shapes,
        'loc': location,
        'perc_threshold': perc_threshold
    }
    
    ttoc = time.time()
    print(f"Time took in get_avalanches: {ttoc-ttic:.2f} seconds")
    
    return Result

def find_avalanches(array):
    """Simple avalanche detection for shape analysis"""
    activity_array = np.sum(array, axis=0)
    avalanches = []
    current_avalanche = []
    for activity in activity_array:
        if activity > 0:
            current_avalanche.append(activity)
        elif current_avalanche:
            avalanches.append(current_avalanche)
            current_avalanche = []
    if current_avalanche:
        avalanches.append(current_avalanche)
    return avalanches

# ============================================
# AVALANCHE ANALYSIS FUNCTIONS
# ============================================

def EXCLUDE(data, bm, nfactor=0, verbose=True):
    """Find power-law fitting range"""
    import criticality as cr
    from criticality import exclude as ex
    return ex.EXCLUDE(data, bm, nfactor=nfactor, verbose=verbose)

def AV_analysis(burst, T, flag=1, bm=20, tm=10, nfactor_bm=0, nfactor_tm=0,
                nfactor_bm_tail=0.8, nfactor_tm_tail=1.0, none_fact=40,
                verbose=True, exclude=False, 
                exclude_burst=50, exclude_time=20, 
                exclude_diff_b=20, exclude_diff_t=10, 
                plot=True, pltname='', saveloc=''):
    """
    Analyze avalanche distributions
    Returns alpha (size exponent), beta (duration exponent), and scaling difference
    """
    import criticality as cr
    from criticality import exclude as ex
    
    Result = {}
    
    # Analyze size distribution
    if bm is None:
        bm = int(np.max(burst)/none_fact)
    
    burstMax, burstMin, alpha = ex.EXCLUDE(
        burst[burst < np.power(np.max(burst), nfactor_bm_tail)], 
        bm, nfactor=nfactor_bm, verbose=verbose)
    
    idx_burst = np.where(np.logical_and(burst <= burstMax, burst >= burstMin))[0]
    
    if verbose:
        print(f"alpha: {alpha}")
        print(f"burst min: {burstMin}, max: {burstMax}")
    
    Result['burst'] = burst
    Result['alpha'] = alpha
    Result['xmin'] = burstMin
    Result['xmax'] = burstMax
    Result['EX_b'] = False
    
    if exclude:
        if burstMin > exclude_burst or (burstMax-burstMin) < exclude_diff_b:
            print(f'Excluded for burst: xmin {burstMin} diff: {burstMax-burstMin}')
            Result['EX_b'] = True
    
    # Analyze duration distribution
    if tm is None:
        tm = int(np.max(T)/none_fact)
    
    tMax, tMin, beta = ex.EXCLUDE(
        T[T < np.power(np.max(T), nfactor_tm_tail)], 
        tm, nfactor=nfactor_tm, verbose=verbose)
    
    idx_time = np.where(np.logical_and(T >= tMin, T <= tMax))[0]
    
    if verbose:
        print(f"beta: {beta}")
        print(f"time min: {tMin}, max: {tMax}")
    
    Result['T'] = T
    Result['beta'] = beta
    Result['tmin'] = tMin
    Result['tmax'] = tMax
    Result['EX_t'] = False
    
    if exclude:
        if tMin > exclude_time or (tMax-tMin) < exclude_diff_t:
            print(f'Excluded for time: tmin {tMin} diff: {tMax-tMin}')
            Result['EX_t'] = True
    
    # Calculate scaling relation
    TT = np.arange(1, np.max(T) + 1)
    Sm = []
    for i in np.arange(0, np.size(TT)):
        Sm.append(np.mean(burst[np.where(T == TT[i])[0]]))
    Sm = np.asarray(Sm)
    Loc = np.where(Sm > 0)[0]
    TT = TT[Loc]
    Sm = Sm[Loc]
    
    fit_sigma = np.polyfit(
        np.log(TT[np.where(np.logical_and(TT > tMin, TT < tMax))[0]]),
        np.log(Sm[np.where(np.logical_and(TT > tMin, TT < tMax))[0]]), 1)
    
    sigma = (beta - 1) / (alpha - 1)
    
    Result['pre'] = sigma
    Result['fit'] = fit_sigma
    Result['df'] = np.abs(sigma - fit_sigma[0])
    Result['TT'] = TT
    Result['Sm'] = Sm
    
    if plot:
        plot_avalanche_distributions(Result, burst, T, pltname, saveloc)
    
    return Result

def plot_avalanche_distributions(Result, burst, T, pltname, saveloc):
    """Plot avalanche size and duration distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Size distribution
    pdf = np.histogram(burst, bins=np.arange(1, np.max(burst) + 2))[0]
    p = pdf / np.sum(pdf)
    axes[0].loglog(np.arange(1, np.max(burst) + 1), p, 'o', 
                   markersize=5, color='darkorchid', alpha=0.75)
    axes[0].set_xlabel('Avalanche Size')
    axes[0].set_ylabel('PDF(S)')
    axes[0].set_title(f'Size Distribution, α = {Result["alpha"]:.3f}')
    
    # Duration distribution
    tdf = np.histogram(T, bins=np.arange(1, np.max(T) + 2))[0]
    t = tdf / np.sum(tdf)
    axes[1].loglog(np.arange(1, np.max(T) + 1), t, 'o',
                   markersize=5, color='mediumseagreen', alpha=0.75)
    axes[1].set_xlabel('Avalanche Duration')
    axes[1].set_ylabel('PDF(D)')
    axes[1].set_title(f'Duration Distribution, β = {Result["beta"]:.3f}')
    
    # Scaling relation
    axes[2].loglog(Result['TT'], Result['Sm'], 'o', 
                   color='#fb7d07', markersize=5, alpha=0.75)
    axes[2].set_xlabel('Duration')
    axes[2].set_ylabel('<S>')
    axes[2].set_title(f'Scaling Relation, Δ = {Result["df"]:.3f}')
    
    plt.tight_layout()
    savepath = os.path.join(saveloc, f'{pltname}_avalanche_analysis.pdf')
    plt.savefig(savepath, format='pdf', dpi=300)
    plt.close()
    print(f"Avalanche plots saved to: {savepath}")

def compute_eigenvalue_spectrum(raster_e, raster_i=None, lag=1, n_shuffles=100, 
                                plot_spectrum=True, plot_distribution=True, plot_spectral_density=True,
                                save_plots=True, saveloc='', pltname='eigenvalue',
                                verbose=True, use_sparse=False, max_neurons=None):
    """
    OPTIMIZED: Compute eigenvalue spectrum from spike raster data using vectorized operations.
    Now handles both excitatory and inhibitory populations with spectral density heatmap!
    
    Parameters:
    -----------
    raster_e : np.ndarray
        Excitatory spike raster data with shape (n_neurons_e, n_timesteps)
    raster_i : np.ndarray or None
        Inhibitory spike raster data with shape (n_neurons_i, n_timesteps)
    lag : int
        Time lag for computing correlations (default: 1)
    n_shuffles : int
        Number of shuffles for surrogate data to estimate noise floor (default: 100)
    plot_spectrum : bool
        Whether to plot eigenvalue spectrum as density heatmap
    plot_distribution : bool
        Whether to plot eigenvalue distribution histogram
    plot_spectral_density : bool
        Whether to plot additional spectral analysis plots
    save_plots : bool
        Whether to save plots to disk
    saveloc : str
        Directory to save plots
    pltname : str
        Base name for saved plots
    verbose : bool
        Whether to print progress information
    use_sparse : bool
        Whether to use sparse matrix operations (good for very sparse data)
    max_neurons : int or None
        Maximum number of neurons to analyze (randomly sampled if exceeded)
    
    Returns:
    --------
    dict : Dictionary containing analysis results including E-E, E-I, I-E, I-I correlations
    """
    
    if verbose:
        print(f"\n{'='*50}")
        print("EIGENVALUE SPECTRUM ANALYSIS (E/I VERSION)")
        print(f"{'='*50}")
        print(f"Excitatory raster shape: {raster_e.shape}")
        if raster_i is not None:
            print(f"Inhibitory raster shape: {raster_i.shape}")
        print(f"Computing correlations with lag={lag}")
    
    start_time = time.time()
    
    # Ensure rasters are 2D
    if len(raster_e.shape) == 1:
        raster_e = raster_e.reshape(1, -1)
    
    if raster_i is not None and len(raster_i.shape) == 1:
        raster_i = raster_i.reshape(1, -1)
    
    # Combine E and I populations if both exist
    if raster_i is not None:
        # Create combined raster with E neurons first, then I neurons
        raster = np.vstack([raster_e, raster_i])
        n_neurons_e = raster_e.shape[0]
        n_neurons_i = raster_i.shape[0]
        n_neurons = n_neurons_e + n_neurons_i
        
        # Create labels for neurons
        neuron_types = ['E'] * n_neurons_e + ['I'] * n_neurons_i
        
        if verbose:
            print(f"Combined network: {n_neurons_e} excitatory + {n_neurons_i} inhibitory = {n_neurons} total neurons")
    else:
        raster = raster_e
        n_neurons = raster.shape[0]
        n_neurons_e = n_neurons
        n_neurons_i = 0
        neuron_types = ['E'] * n_neurons
    
    t_length = raster.shape[1] - lag
    
    # Subsample neurons if too many (for speed)
    if max_neurons and n_neurons > max_neurons:
        if verbose:
            print(f"Subsampling from {n_neurons} to {max_neurons} neurons for speed...")
        # Sample proportionally from E and I
        if n_neurons_i > 0:
            e_sample = int(max_neurons * n_neurons_e / n_neurons)
            i_sample = max_neurons - e_sample
            idx_e = np.random.choice(n_neurons_e, min(e_sample, n_neurons_e), replace=False)
            idx_i = np.random.choice(range(n_neurons_e, n_neurons), min(i_sample, n_neurons_i), replace=False)
            idx = np.concatenate([idx_e, idx_i])
        else:
            idx = np.random.choice(n_neurons, max_neurons, replace=False)
        
        raster = raster[idx, :]
        neuron_types = [neuron_types[i] for i in idx]
        n_neurons = max_neurons
        n_neurons_e = sum(1 for t in neuron_types if t == 'E')
        n_neurons_i = sum(1 for t in neuron_types if t == 'I')
    
    # Skip if too few neurons or timesteps
    if n_neurons < 2:
        print("WARNING: Need at least 2 neurons for correlation analysis")
        return None
    if t_length < 10:
        print("WARNING: Time series too short after lag adjustment")
        return None
    
    # ============================================
    # OPTIMIZED: Vectorized Correlation Matrix
    # ============================================
    if verbose:
        print(f"\nComputing {n_neurons}x{n_neurons} correlation matrix (vectorized)...")
        if n_neurons_i > 0:
            print(f"  Matrix blocks: E-E ({n_neurons_e}x{n_neurons_e}), E-I ({n_neurons_e}x{n_neurons_i}), "
                  f"I-E ({n_neurons_i}x{n_neurons_e}), I-I ({n_neurons_i}x{n_neurons_i})")
        corr_start = time.time()
    
    # Pre-compute time series
    series_early = raster[:, :t_length].astype(np.float32)
    series_late = raster[:, lag:lag + t_length].astype(np.float32)
    
    # Check which neurons have activity
    has_activity_early = np.any(series_early, axis=1)
    has_activity_late = np.any(series_late, axis=1)
    active_early = np.where(has_activity_early)[0]
    active_late = np.where(has_activity_late)[0]
    
    if verbose:
        print(f"  Active neurons: {len(active_early)} early, {len(active_late)} late")
        if n_neurons_i > 0:
            active_e_early = sum(1 for i in active_early if neuron_types[i] == 'E')
            active_i_early = len(active_early) - active_e_early
            print(f"    Early: {active_e_early} E, {active_i_early} I")
        print(f"  Computing correlations for {len(active_early) * len(active_late)} pairs...")
    
    # Initialize correlation matrix
    correlation_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    
    if len(active_early) > 0 and len(active_late) > 0:
        if verbose:
            print("  Standardizing time series...", end="", flush=True)
        
        # Standardize the active time series
        active_series_early = series_early[active_early]
        means_early = np.mean(active_series_early, axis=1, keepdims=True)
        stds_early = np.std(active_series_early, axis=1, keepdims=True)
        stds_early[stds_early == 0] = 1
        z_early = (active_series_early - means_early) / stds_early
        
        active_series_late = series_late[active_late]
        means_late = np.mean(active_series_late, axis=1, keepdims=True)
        stds_late = np.std(active_series_late, axis=1, keepdims=True)
        stds_late[stds_late == 0] = 1
        z_late = (active_series_late - means_late) / stds_late
        
        if verbose:
            print(" Done!", flush=True)
            print("  Computing correlation matrix via matrix multiplication...", end="", flush=True)
        
        # Compute correlation via matrix multiplication
        corr_submatrix = np.dot(z_early, z_late.T) / t_length
        
        if verbose:
            print(" Done!", flush=True)
            print("  Filling correlation matrix...", end="", flush=True)
        
        # Fill in the correlation matrix
        for i, idx_i in enumerate(active_early):
            for j, idx_j in enumerate(active_late):
                correlation_matrix[idx_i, idx_j] = corr_submatrix[i, j]
        
        if verbose:
            print(" Done!", flush=True)
    
    # ============================================
    # Analyze E/I submatrices
    # ============================================
    if n_neurons_i > 0 and verbose:
        print("\nAnalyzing E/I correlation blocks:")
        
        # Extract submatrices
        corr_ee = correlation_matrix[:n_neurons_e, :n_neurons_e]
        corr_ei = correlation_matrix[:n_neurons_e, n_neurons_e:]
        corr_ie = correlation_matrix[n_neurons_e:, :n_neurons_e]
        corr_ii = correlation_matrix[n_neurons_e:, n_neurons_e:]
        
        # Compute statistics for each block
        print(f"  E-E: mean={np.mean(np.abs(corr_ee)):.4f}, max={np.max(np.abs(corr_ee)):.4f}")
        print(f"  E-I: mean={np.mean(np.abs(corr_ei)):.4f}, max={np.max(np.abs(corr_ei)):.4f}")
        print(f"  I-E: mean={np.mean(np.abs(corr_ie)):.4f}, max={np.max(np.abs(corr_ie)):.4f}")
        print(f"  I-I: mean={np.mean(np.abs(corr_ii)):.4f}, max={np.max(np.abs(corr_ii)):.4f}")
    
    # ============================================
    # Compute Eigenvalues
    # ============================================
    if verbose:
        print("\nComputing eigenvalues...")
        eig_start = time.time()
    
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    
    if verbose:
        eig_elapsed = time.time() - eig_start
        corr_elapsed = time.time() - corr_start
        print(f"  Eigenvalue computation took {eig_elapsed:.2f} seconds")
        print(f"  Total correlation + eigenvalue time: {corr_elapsed:.2f} seconds")
        print(f"\nResults:")
        print(f"  Max eigenvalue: {max_eigenvalue:.4f}")
        print(f"  Number of positive eigenvalues: {np.sum(eigenvalues.real > 0)}")
        print(f"  Number of negative eigenvalues: {np.sum(eigenvalues.real < 0)}")
        print(f"  Eigenvalue range: [{np.min(eigenvalues.real):.4f}, {np.max(eigenvalues.real):.4f}]")
    
    # ============================================
    # OPTIMIZED: Shuffled Data Analysis
    # ============================================
    max_eigenvalues_shuffled = []
    
    if n_shuffles > 0:
        if verbose:
            print(f"\nComputing noise floor with {n_shuffles} shuffles (optimized)...")
            print("Progress: ", end="", flush=True)
        
        shuffle_start = time.time()
        last_print_time = time.time()
        
        for shuffle_idx in range(n_shuffles):
            # Print progress bar
            if verbose:
                current_time = time.time()
                if shuffle_idx == 0 or (current_time - last_print_time) > 0.5:
                    progress_pct = (shuffle_idx / n_shuffles) * 100
                    print(f"\rProgress: [{'=' * int(progress_pct/2):<50}] {progress_pct:.1f}% ({shuffle_idx}/{n_shuffles})", 
                          end="", flush=True)
                    last_print_time = current_time
            
            # Create shuffled correlation matrix
            shuffled_corr = np.zeros((n_neurons, n_neurons), dtype=np.float32)
            
            if len(active_early) > 0 and len(active_late) > 0:
                # Shuffle only active neurons
                shuffled_early = np.zeros_like(active_series_early)
                shuffled_late = np.zeros_like(active_series_late)
                
                for i in range(len(active_early)):
                    perm = np.random.permutation(t_length)
                    shuffled_early[i] = active_series_early[i, perm]
                
                for j in range(len(active_late)):
                    perm = np.random.permutation(t_length)
                    shuffled_late[j] = active_series_late[j, perm]
                
                # Standardize shuffled data
                means_sh_early = np.mean(shuffled_early, axis=1, keepdims=True)
                stds_sh_early = np.std(shuffled_early, axis=1, keepdims=True)
                stds_sh_early[stds_sh_early == 0] = 1
                z_sh_early = (shuffled_early - means_sh_early) / stds_sh_early
                
                means_sh_late = np.mean(shuffled_late, axis=1, keepdims=True)
                stds_sh_late = np.std(shuffled_late, axis=1, keepdims=True)
                stds_sh_late[stds_sh_late == 0] = 1
                z_sh_late = (shuffled_late - means_sh_late) / stds_sh_late
                
                # Compute correlation matrix for shuffled data
                corr_sh_submatrix = np.dot(z_sh_early, z_sh_late.T) / t_length
                
                # Fill shuffled correlation matrix
                for i, idx_i in enumerate(active_early):
                    for j, idx_j in enumerate(active_late):
                        shuffled_corr[idx_i, idx_j] = corr_sh_submatrix[i, j]
            
            # Get max eigenvalue from shuffled data
            if n_neurons > 100:
                # For large matrices, use power iteration
                from scipy.sparse.linalg import eigs
                try:
                    max_eig_shuffled = eigs(shuffled_corr, k=1, which='LM', 
                                           return_eigenvectors=False, maxiter=100)
                    max_eigenvalues_shuffled.append(np.abs(max_eig_shuffled[0]))
                except:
                    eigenvalues_shuffled = np.linalg.eigvals(shuffled_corr)
                    max_eigenvalues_shuffled.append(np.max(np.abs(eigenvalues_shuffled)))
            else:
                eigenvalues_shuffled = np.linalg.eigvals(shuffled_corr)
                max_eigenvalues_shuffled.append(np.max(np.abs(eigenvalues_shuffled)))
        
        max_eigenvalue_shuffled_mean = np.mean(max_eigenvalues_shuffled)
        max_eigenvalue_shuffled_std = np.std(max_eigenvalues_shuffled)
        significance_ratio = max_eigenvalue / max_eigenvalue_shuffled_mean if max_eigenvalue_shuffled_mean > 0 else np.inf
        
        if verbose:
            print(f"\rProgress: [{'=' * 50}] 100.0% ({n_shuffles}/{n_shuffles}) - Complete!", flush=True)
            shuffle_elapsed = time.time() - shuffle_start
            print(f"\n  Shuffling took {shuffle_elapsed:.2f} seconds ({shuffle_elapsed/n_shuffles:.3f} sec/shuffle)")
            print(f"\nShuffled data statistics:")
            print(f"  Mean max eigenvalue: {max_eigenvalue_shuffled_mean:.4f}")
            print(f"  Std max eigenvalue: {max_eigenvalue_shuffled_std:.4f}")
            print(f"  Significance ratio: {significance_ratio:.2f}")
    else:
        max_eigenvalue_shuffled_mean = None
        max_eigenvalue_shuffled_std = None
        significance_ratio = None
    
    if verbose:
        total_elapsed = time.time() - start_time
        print(f"\nTotal analysis time: {total_elapsed:.2f} seconds")
    
    # ============================================
    # Plotting
    # ============================================
    
    # Plot 1: Eigenvalue Spectrum with dots, density overlay, and grid counts
    if plot_spectrum:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine range for both plots
        real_margin = 0.5
        imag_margin = 0.5
        real_range = [np.min(eigenvalues.real) - real_margin, np.max(eigenvalues.real) + real_margin]
        imag_range = [np.min(eigenvalues.imag) - imag_margin, np.max(eigenvalues.imag) + imag_margin]
        
        # Make the range square and centered
        max_range = max(abs(real_range[0]), abs(real_range[1]), abs(imag_range[0]), abs(imag_range[1]))
        real_range = [-max_range, max_range]
        imag_range = [-max_range, max_range]
        
        # ========== LEFT PLOT: Dots with density overlay and grid counts ==========
        
        # First, plot ALL eigenvalues as individual dots
        scatter1 = axes[0].scatter(eigenvalues.real, eigenvalues.imag, 
                                   c='darkblue', s=15, alpha=0.6, edgecolors='none',
                                   zorder=2, label='Individual eigenvalues')
        
        # Create density overlay using 2D histogram
        n_bins_overlay = 50  # For smooth overlay
        H_overlay, xedges_o, yedges_o = np.histogram2d(eigenvalues.real, eigenvalues.imag, 
                                                       bins=n_bins_overlay, range=[real_range, imag_range])
        H_overlay = H_overlay.T
        
        # Apply Gaussian smoothing for nice overlay
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H_overlay, sigma=1.0)
        
        # Create translucent overlay only where there are eigenvalues
        H_masked = np.ma.masked_where(H_smooth < 0.5, H_smooth)
        
        # Add the translucent density overlay
        im = axes[0].imshow(H_masked, interpolation='bilinear', origin='lower',
                           extent=[xedges_o[0], xedges_o[-1], yedges_o[0], yedges_o[-1]],
                           cmap='YlOrRd', aspect='equal', alpha=0.3, vmin=1, 
                           vmax=np.max(H_smooth), zorder=1)
        
        # Create 1x1 grid for counting
        grid_size = 1.0  # 1 unit squares
        x_bins = np.arange(real_range[0], real_range[1] + grid_size, grid_size)
        y_bins = np.arange(imag_range[0], imag_range[1] + grid_size, grid_size)
        H_counts, xedges_c, yedges_c = np.histogram2d(eigenvalues.real, eigenvalues.imag, 
                                                      bins=[x_bins, y_bins])
        
        # Add count labels to grid cells with eigenvalues
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                count = int(H_counts[i, j])
                if count > 0:  # Only show counts where there are eigenvalues
                    x_center = (x_bins[i] + x_bins[i+1]) / 2
                    y_center = (y_bins[j] + y_bins[j+1]) / 2
                    
                    # Choose text color based on count
                    if count == 1:
                        color = 'gray'
                        fontsize = 8
                    elif count <= 5:
                        color = 'orange'
                        fontsize = 9
                    else:
                        color = 'red'
                        fontsize = 10
                        
                    axes[0].text(x_center, y_center, str(count), 
                               ha='center', va='center', color=color, 
                               fontsize=fontsize, fontweight='bold',
                               alpha=0.7, zorder=4)
        
        # Add grid lines
        axes[0].set_xticks(x_bins, minor=True)
        axes[0].set_yticks(y_bins, minor=True)
        axes[0].grid(True, which='minor', alpha=0.2, linewidth=0.5, color='gray')
        axes[0].grid(True, which='major', alpha=0.3, linewidth=1.0, color='gray')
        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = axes[0].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, linewidth=1.2, 
                                   label='Unit circle', zorder=3)
        
        # Add noise floor circle if available
        if n_shuffles > 0 and max_eigenvalue_shuffled_mean:
            circle = plt.Circle((0, 0), max_eigenvalue_shuffled_mean, 
                              color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.5,
                              zorder=3)
            axes[0].add_patch(circle)
            noise_floor = axes[0].plot([], [], 'r--', linewidth=1.5, 
                                       label=f'Noise floor (r={max_eigenvalue_shuffled_mean:.2f})')
        
        # Mark the maximum eigenvalue clearly
        max_idx = np.argmax(np.abs(eigenvalues))
        max_eig = axes[0].scatter(eigenvalues[max_idx].real, eigenvalues[max_idx].imag,
                                  c='lime', s=100, marker='*', edgecolors='black', linewidth=1,
                                  label=f'Max |λ| = {max_eigenvalue:.3f}', zorder=5)
        
        # Add subtle crosshairs
        axes[0].axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=0)
        axes[0].axvline(0, color='black', linewidth=0.8, alpha=0.3, zorder=0)
        
        # Create custom legend with colors
        from matplotlib.patches import Patch
        legend_elements = [
            scatter1,
            Patch(facecolor='yellow', alpha=0.3, label='Low density'),
            Patch(facecolor='red', alpha=0.3, label='High density'),
            unit_circle[0],
        ]
        if n_shuffles > 0 and max_eigenvalue_shuffled_mean:
            legend_elements.append(noise_floor[0])
        legend_elements.append(max_eig)
        
        # Add text color legend
        legend_elements.extend([
            Patch(facecolor='none', edgecolor='none', label='Grid counts:'),
            Patch(facecolor='gray', alpha=0.7, label='1 eigenvalue'),
            Patch(facecolor='orange', alpha=0.7, label='2-5 eigenvalues'),
            Patch(facecolor='red', alpha=0.7, label='>5 eigenvalues'),
        ])
        
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
        
        # Clean up the plot
        axes[0].set_xlabel('Real Part', fontsize=12)
        axes[0].set_ylabel('Imaginary Part', fontsize=12)
        axes[0].set_title('Eigenvalue Spectrum with Density', fontsize=13, fontweight='bold')
        axes[0].set_xlim(real_range)
        axes[0].set_ylim(imag_range)
        axes[0].set_aspect('equal')
        
        # ========== RIGHT PLOT: Pure scatter plot for clarity ==========
        
        # Color by significance if we have shuffle data
        if n_shuffles > 0:
            # Color based on significance
            significant = np.abs(eigenvalues) > (max_eigenvalue_shuffled_mean + 2*max_eigenvalue_shuffled_std)
            
            # Non-significant eigenvalues
            axes[1].scatter(eigenvalues[~significant].real, eigenvalues[~significant].imag, 
                          c='lightgray', s=10, alpha=0.5, edgecolors='none',
                          label=f'Non-significant (n={np.sum(~significant)})')
            
            # Significant eigenvalues
            axes[1].scatter(eigenvalues[significant].real, eigenvalues[significant].imag, 
                          c='darkblue', s=20, alpha=0.8, edgecolors='none',
                          label=f'Significant (n={np.sum(significant)})')
        else:
            # Just plot all eigenvalues uniformly
            axes[1].scatter(eigenvalues.real, eigenvalues.imag, 
                          c='darkblue', s=15, alpha=0.6, edgecolors='none',
                          label=f'All eigenvalues (n={len(eigenvalues)})')
        
        # Add the same 1x1 grid
        axes[1].set_xticks(x_bins, minor=True)
        axes[1].set_yticks(y_bins, minor=True)
        axes[1].grid(True, which='minor', alpha=0.15, linewidth=0.5, color='gray')
        axes[1].grid(True, which='major', alpha=0.25, linewidth=1.0, color='gray')
        
        # Add unit circle
        axes[1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, linewidth=1.2, label='Unit circle')
        
        # Add noise floor
        if n_shuffles > 0 and max_eigenvalue_shuffled_mean:
            circle2 = plt.Circle((0, 0), max_eigenvalue_shuffled_mean, 
                               color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.5)
            axes[1].add_patch(circle2)
        
        # Mark max eigenvalue
        axes[1].scatter(eigenvalues[max_idx].real, eigenvalues[max_idx].imag,
                       c='lime', s=100, marker='*', edgecolors='black', linewidth=1,
                       label=f'Max |λ| = {max_eigenvalue:.3f}', zorder=5)
        
        # Subtle crosshairs
        axes[1].axhline(0, color='black', linewidth=0.8, alpha=0.3)
        axes[1].axvline(0, color='black', linewidth=0.8, alpha=0.3)
        
        axes[1].set_xlabel('Real Part', fontsize=12)
        axes[1].set_ylabel('Imaginary Part', fontsize=12)
        axes[1].set_title('Individual Eigenvalues', fontsize=13, fontweight='bold')
        axes[1].set_xlim(real_range)
        axes[1].set_ylim(imag_range)
        axes[1].set_aspect('equal')
        axes[1].legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Overall title
        network_type = f'E/I Network (E={n_neurons_e}, I={n_neurons_i})' if n_neurons_i > 0 else f'Network (n={n_neurons})'
        plt.suptitle(f'Eigenvalue Analysis - {network_type}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            savepath = os.path.join(saveloc, f'{pltname}_spectrum.pdf')
            plt.savefig(savepath, format='pdf', dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Spectrum plot saved to: {savepath}")
        else:
            plt.show()
        plt.close()
    
    # Plot 2: Eigenvalue Distribution and Comparison with Shuffled
    if plot_distribution and n_shuffles > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Eigenvalue magnitude distribution
        eigenvalue_mags = np.abs(eigenvalues)
        axes[0].hist(eigenvalue_mags, bins=30, color='darkblue', alpha=0.7, 
                    edgecolor='black', label='Original')
        
        if n_shuffles > 0:
            axes[0].axvline(max_eigenvalue_shuffled_mean, color='red', 
                          linestyle='--', linewidth=2, label='Shuffled mean')
            axes[0].axvspan(max_eigenvalue_shuffled_mean - max_eigenvalue_shuffled_std,
                          max_eigenvalue_shuffled_mean + max_eigenvalue_shuffled_std,
                          color='red', alpha=0.2, label='Shuffled ±1 std')
        
        axes[0].axvline(max_eigenvalue, color='darkgreen', 
                       linestyle='-', linewidth=2, label=f'Max = {max_eigenvalue:.3f}')
        
        axes[0].set_xlabel('|Eigenvalue|', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Eigenvalue Magnitudes', fontsize=13)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right: Sorted eigenvalues (log scale)
        sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        axes[1].semilogy(range(len(sorted_eigenvalues)), sorted_eigenvalues, 
                        'o-', color='darkblue', markersize=6, alpha=0.7,
                        label='Original eigenvalues')
        
        if n_shuffles > 0:
            axes[1].axhline(max_eigenvalue_shuffled_mean, color='red', 
                          linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Noise floor (μ={max_eigenvalue_shuffled_mean:.3f})')
            axes[1].fill_between(range(len(sorted_eigenvalues)),
                                max_eigenvalue_shuffled_mean - max_eigenvalue_shuffled_std,
                                max_eigenvalue_shuffled_mean + max_eigenvalue_shuffled_std,
                                color='red', alpha=0.2)
        
        axes[1].set_xlabel('Rank', fontsize=12)
        axes[1].set_ylabel('|Eigenvalue| (log scale)', fontsize=12)
        axes[1].set_title('Ranked Eigenvalue Spectrum', fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')
        
        network_type = f'E/I Network' if n_neurons_i > 0 else 'Excitatory Network'
        plt.suptitle(f'Eigenvalue Analysis - {network_type} (lag={lag})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            savepath = os.path.join(saveloc, f'{pltname}_distribution.pdf')
            plt.savefig(savepath, format='pdf', dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Distribution plot saved to: {savepath}")
        else:
            plt.show()
        plt.close()
    
    # Plot 3: Correlation Matrix and E/I Block Analysis
    if plot_spectral_density:
        n_subplots = 3 if n_neurons_i > 0 else 2
        fig, axes = plt.subplots(1, n_subplots, figsize=(5*n_subplots, 5))
        
        # First subplot: Correlation matrix heatmap
        vmax = np.percentile(np.abs(correlation_matrix), 95)
        im1 = axes[0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            aspect='auto', interpolation='nearest')
        
        if n_neurons_i > 0:
            # Add lines to separate E/I blocks
            axes[0].axhline(n_neurons_e - 0.5, color='black', linewidth=2, alpha=0.7)
            axes[0].axvline(n_neurons_e - 0.5, color='black', linewidth=2, alpha=0.7)
            
            # Add labels
            axes[0].text(n_neurons_e/2, -n_neurons*0.02, 'E', ha='center', fontsize=12, fontweight='bold')
            axes[0].text(n_neurons_e + n_neurons_i/2, -n_neurons*0.02, 'I', ha='center', fontsize=12, fontweight='bold')
            axes[0].text(-n_neurons*0.02, n_neurons_e/2, 'E', va='center', rotation=90, fontsize=12, fontweight='bold')
            axes[0].text(-n_neurons*0.02, n_neurons_e + n_neurons_i/2, 'I', va='center', rotation=90, fontsize=12, fontweight='bold')
        
        axes[0].set_xlabel('Neuron Index', fontsize=11)
        axes[0].set_ylabel('Neuron Index', fontsize=11)
        axes[0].set_title('Correlation Matrix', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Correlation', fraction=0.046)
        
        # Second subplot: Eigenvalue real/imaginary parts distribution
        axes[1].hist(eigenvalues.real, bins=30, alpha=0.5, color='blue', label='Real parts')
        axes[1].hist(eigenvalues.imag, bins=30, alpha=0.5, color='red', label='Imaginary parts')
        axes[1].axvline(0, color='black', linewidth=0.5, alpha=0.5)
        axes[1].set_xlabel('Eigenvalue Component', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Real vs Imaginary Components', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Third subplot (if E/I): Block-wise correlation strengths
        if n_neurons_i > 0:
            # Extract submatrices
            corr_ee = correlation_matrix[:n_neurons_e, :n_neurons_e]
            corr_ei = correlation_matrix[:n_neurons_e, n_neurons_e:]
            corr_ie = correlation_matrix[n_neurons_e:, :n_neurons_e]
            corr_ii = correlation_matrix[n_neurons_e:, n_neurons_e:]
            
            # Compute mean absolute correlations for each block
            block_means = np.array([[np.mean(np.abs(corr_ee)), np.mean(np.abs(corr_ei))],
                                   [np.mean(np.abs(corr_ie)), np.mean(np.abs(corr_ii))]])
            
            im2 = axes[2].imshow(block_means, cmap='viridis', aspect='auto', vmin=0)
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = axes[2].text(j, i, f'{block_means[i, j]:.3f}',
                                      ha="center", va="center", color="white", fontsize=14, fontweight='bold')
            
            axes[2].set_xticks([0, 1])
            axes[2].set_yticks([0, 1])
            axes[2].set_xticklabels(['E', 'I'], fontsize=12)
            axes[2].set_yticklabels(['E', 'I'], fontsize=12)
            axes[2].set_xlabel('Target Population', fontsize=11)
            axes[2].set_ylabel('Source Population', fontsize=11)
            axes[2].set_title('Mean |Correlation| by Block', fontsize=12)
            plt.colorbar(im2, ax=axes[2], label='Mean |r|', fraction=0.046)
        
        plt.suptitle('Correlation Structure Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            savepath = os.path.join(saveloc, f'{pltname}_correlation_analysis.pdf')
            plt.savefig(savepath, format='pdf', dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Correlation analysis plot saved to: {savepath}")
        else:
            plt.show()
        plt.close()
    
    # ============================================
    # Return Results
    # ============================================
    results = {
        'eigenvalues': eigenvalues,
        'max_eigenvalue': max_eigenvalue,
        'correlation_matrix': correlation_matrix,
        'max_eigenvalue_shuffled_mean': max_eigenvalue_shuffled_mean,
        'max_eigenvalue_shuffled_std': max_eigenvalue_shuffled_std,
        'significance_ratio': significance_ratio,
        'n_neurons': n_neurons,
        'n_neurons_e': n_neurons_e,
        'n_neurons_i': n_neurons_i,
        'lag': lag
    }
    
    # Add E/I specific results if applicable
    if n_neurons_i > 0:
        results['corr_ee_mean'] = np.mean(np.abs(correlation_matrix[:n_neurons_e, :n_neurons_e]))
        results['corr_ei_mean'] = np.mean(np.abs(correlation_matrix[:n_neurons_e, n_neurons_e:]))
        results['corr_ie_mean'] = np.mean(np.abs(correlation_matrix[n_neurons_e:, :n_neurons_e]))
        results['corr_ii_mean'] = np.mean(np.abs(correlation_matrix[n_neurons_e:, n_neurons_e:]))
    
    return results


# Helper function to extract E and I rasters from your data
def extract_ei_rasters(h5_file_path, starting_time, end_time, n_exc=200, n_inh=40):
    """
    Extract excitatory and inhibitory rasters from SORN data.
    
    NOTE: When using simulation_ExtraInputNew.py, typically only excitatory neurons 
    are saved to reduce file size. Inhibitory neurons are simulated but not recorded.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the H5 file
    starting_time : int
        Start time point
    end_time : int
        End time point  
    n_exc : int
        Number of excitatory neurons (default: 200)
    n_inh : int
        Number of inhibitory neurons (default: 40)
    
    Returns:
    --------
    tuple : (raster_e, raster_i) - excitatory and inhibitory rasters
            Note: raster_i will be None if inhibitory spikes weren't saved
    """
    try:
        h5 = tables.open_file(h5_file_path, 'r')
        data = h5.root
        
        # Get network parameters from the h5 file
        if hasattr(h5.root.c, 'N_e') and hasattr(h5.root.c, 'N_i'):
            n_exc = h5.root.c.N_e[0]
            n_inh = h5.root.c.N_i[0]
            print(f"Network parameters from file: N_e={n_exc}, N_i={n_inh}")
        
        # Initialize rasters
        raster_e = None
        raster_i = None
        
        # Check for main Spikes data (excitatory or combined)
        if data.__contains__('Spikes'):
            spike_shape = data.Spikes.shape
            print(f"Spikes array shape: {spike_shape}")
            
            # Get actual data dimensions
            actual_data_length = spike_shape[2]
            actual_end = min(end_time, actual_data_length) if end_time else actual_data_length
            
            # Load the spike data - note the [0] to get first (and only) dimension
            full_raster = data.Spikes[0, :, starting_time:actual_end]
            print(f"Loaded raster shape: {full_raster.shape}")
            
            # Check total neurons
            total_neurons = full_raster.shape[0]
            print(f"Total neurons in Spikes data: {total_neurons}")
            
            if total_neurons == n_exc + n_inh:
                # Both E and I neurons are saved together
                raster_e = full_raster[:n_exc, :]
                raster_i = full_raster[n_exc:n_exc+n_inh, :]
                print(f"Found both E and I neurons in Spikes array")
                print(f"Split into E: {raster_e.shape}, I: {raster_i.shape}")
            elif total_neurons == n_exc:
                # Only excitatory neurons saved (COMMON CASE for ExtraInput experiments)
                raster_e = full_raster
                print("Only excitatory neurons found in Spikes array")
                print("This is expected for ExtraInput experiments to reduce file size")
            else:
                print(f"WARNING: Unexpected number of neurons: {total_neurons}")
                print(f"Expected {n_exc} (E only) or {n_exc + n_inh} (E+I)")
                raster_e = full_raster[:min(n_exc, total_neurons), :]
                if total_neurons > n_exc:
                    raster_i = full_raster[n_exc:, :]
        
        # Check for separate inhibitory spikes (rare but possible)
        if data.__contains__('SpikesInh') and raster_i is None:
            print("Found separate SpikesInh array")
            inh_shape = data.SpikesInh.shape
            actual_data_length = inh_shape[2]
            actual_end = min(end_time, actual_data_length) if end_time else actual_data_length
            raster_i = data.SpikesInh[0, :, starting_time:actual_end]
            print(f"Loaded inhibitory raster shape: {raster_i.shape}")
        
        # Final status report
        if raster_e is not None and raster_i is not None:
            print(f"\n✓ Successfully loaded both E and I rasters")
            print(f"  Excitatory: {raster_e.shape}")
            print(f"  Inhibitory: {raster_i.shape}")
        elif raster_e is not None:
            print(f"\n✓ Successfully loaded excitatory raster: {raster_e.shape}")
            print("ℹ No inhibitory raster found (typical for ExtraInput experiments)")
            print("  Analysis will proceed with excitatory neurons only")
        else:
            print("\n✗ No spike data found in file")
        
        h5.close()
        return raster_e, raster_i
            
    except Exception as e:
        print(f"Error loading E/I rasters: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# FAST VERSION for very large rasters - uses only max eigenvalue
def compute_max_eigenvalue_fast(raster, lag=1, verbose=True):
    """
    Ultra-fast version that only computes the maximum eigenvalue using power iteration.
    Good for quick checks on very large rasters.
    
    Parameters:
    -----------
    raster : np.ndarray
        Spike raster data with shape (n_neurons, n_timesteps)
    lag : int
        Time lag for computing correlations
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    float : Maximum eigenvalue
    """
    from scipy.sparse.linalg import eigs
    
    if verbose:
        print("Computing max eigenvalue (fast method)...")
    
    # Ensure 2D
    if len(raster.shape) == 1:
        raster = raster.reshape(1, -1)
    
    n_neurons = raster.shape[0]
    t_length = raster.shape[1] - lag
    
    # Get active neurons only
    series_early = raster[:, :t_length].astype(np.float32)
    series_late = raster[:, lag:lag + t_length].astype(np.float32)
    
    active = np.where(np.any(series_early, axis=1) & np.any(series_late, axis=1))[0]
    
    if len(active) < 2:
        return 0.0
    
    # Work only with active neurons
    active_early = series_early[active]
    active_late = series_late[active]
    
    # Standardize
    means_e = np.mean(active_early, axis=1, keepdims=True)
    stds_e = np.std(active_early, axis=1, keepdims=True)
    stds_e[stds_e == 0] = 1
    z_early = (active_early - means_e) / stds_e
    
    means_l = np.mean(active_late, axis=1, keepdims=True)
    stds_l = np.std(active_late, axis=1, keepdims=True)
    stds_l[stds_l == 0] = 1
    z_late = (active_late - means_l) / stds_l
    
    # Correlation matrix (only for active neurons)
    corr_matrix = np.dot(z_early, z_late.T) / t_length
    
    # Get just the largest eigenvalue
    try:
        max_eig = eigs(corr_matrix, k=1, which='LM', return_eigenvectors=False, maxiter=100)
        return np.abs(max_eig[0])
    except:
        # Fallback
        eigenvalues = np.linalg.eigvals(corr_matrix)
        return np.max(np.abs(eigenvalues))


# Function to combine eigenvalues from multiple simulations
def analyze_eigenvalues_batch(h5_files, starting_time, end_time, base_dir, 
                             combine_analysis=True, n_exc=200, n_inh=40,
                             lag=1, n_shuffles=50, verbose=True):
    """
    Analyze eigenvalues from multiple simulations.
    
    Parameters:
    -----------
    h5_files : list
        List of paths to H5 files
    starting_time : int
        Start time point
    end_time : int
        End time point
    base_dir : str
        Directory to save results
    combine_analysis : bool
        If True, combine all eigenvalues into one plot
        If False, analyze each file separately
    n_exc : int
        Number of excitatory neurons
    n_inh : int
        Number of inhibitory neurons
    lag : int
        Time lag for correlations
    n_shuffles : int
        Number of shuffles for noise floor
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict : Combined results or list of individual results
    """
    
    all_eigenvalues = []
    all_max_eigenvalues = []
    individual_results = []
    
    print(f"\n{'='*60}")
    print(f"BATCH EIGENVALUE ANALYSIS")
    print(f"Processing {len(h5_files)} simulations")
    print(f"{'='*60}")
    
    for file_idx, file_path in enumerate(h5_files):
        print(f"\n--- Simulation {file_idx+1}/{len(h5_files)} ---")
        
        # Extract rasters
        raster_e, raster_i = extract_ei_rasters(
            file_path, starting_time, end_time, n_exc, n_inh
        )
        
        if raster_e is not None:
            if not combine_analysis:
                # Analyze each file separately with full plots
                results = compute_eigenvalue_spectrum(
                    raster_e=raster_e,
                    raster_i=raster_i,
                    lag=lag,
                    n_shuffles=n_shuffles if file_idx == 0 else 0,  # Only shuffle for first file to save time
                    plot_spectrum=True,
                    plot_distribution=True,
                    plot_spectral_density=True,
                    save_plots=True,
                    saveloc=base_dir,
                    pltname=f'eigenvalue_sim{file_idx:02d}',
                    verbose=verbose
                )
                individual_results.append(results)
                all_eigenvalues.extend(results['eigenvalues'])
                all_max_eigenvalues.append(results['max_eigenvalue'])
            else:
                # Just compute eigenvalues, save plotting for combined analysis
                results = compute_eigenvalue_spectrum(
                    raster_e=raster_e,
                    raster_i=raster_i,
                    lag=lag,
                    n_shuffles=0,  # Skip shuffling for individual files
                    plot_spectrum=False,
                    plot_distribution=False,
                    plot_spectral_density=False,
                    save_plots=False,
                    verbose=False
                )
                individual_results.append(results)
                all_eigenvalues.extend(results['eigenvalues'])
                all_max_eigenvalues.append(results['max_eigenvalue'])
                
                print(f"  Max eigenvalue: {results['max_eigenvalue']:.4f}")
    
    # Combined analysis
    if combine_analysis and len(all_eigenvalues) > 0:
        print(f"\n{'='*60}")
        print(f"COMBINED ANALYSIS")
        print(f"{'='*60}")
        print(f"Total eigenvalues collected: {len(all_eigenvalues)}")
        print(f"From {len(h5_files)} simulations")
        print(f"Max eigenvalues range: [{np.min(all_max_eigenvalues):.4f}, {np.max(all_max_eigenvalues):.4f}]")
        print(f"Mean max eigenvalue: {np.mean(all_max_eigenvalues):.4f} ± {np.std(all_max_eigenvalues):.4f}")
        
        # Convert to numpy array
        all_eigenvalues = np.array(all_eigenvalues)
        
        # Create combined plot
        plot_combined_eigenvalue_spectrum(
            all_eigenvalues, 
            all_max_eigenvalues,
            n_simulations=len(h5_files),
            n_neurons_per_sim=n_exc,  # Assuming only excitatory
            saveloc=base_dir,
            pltname='eigenvalue_combined',
            verbose=verbose
        )
        
        return {
            'all_eigenvalues': all_eigenvalues,
            'all_max_eigenvalues': all_max_eigenvalues,
            'individual_results': individual_results,
            'mean_max_eigenvalue': np.mean(all_max_eigenvalues),
            'std_max_eigenvalue': np.std(all_max_eigenvalues)
        }
    
    return individual_results


def plot_combined_eigenvalue_spectrum(all_eigenvalues, all_max_eigenvalues, 
                                     n_simulations, n_neurons_per_sim,
                                     saveloc='', pltname='eigenvalue_combined', 
                                     verbose=True):
    """
    Create combined plots for eigenvalues from multiple simulations.
    EXACTLY like individual plots but with ALL simulations combined.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ========== TOP LEFT: Main combined plot with ALL dots visible ==========
    ax_main = fig.add_subplot(gs[0, :2])  # Span 2 columns
    
    # Determine range
    real_margin = 0.5
    imag_margin = 0.5
    real_range = [np.min(all_eigenvalues.real) - real_margin, np.max(all_eigenvalues.real) + real_margin]
    imag_range = [np.min(all_eigenvalues.imag) - imag_margin, np.max(all_eigenvalues.imag) + imag_margin]
    
    # Make square
    max_range = max(abs(real_range[0]), abs(real_range[1]), abs(imag_range[0]), abs(imag_range[1]))
    real_range = [-max_range, max_range]
    imag_range = [-max_range, max_range]
    
    # PLOT ALL INDIVIDUAL EIGENVALUES AS DOTS
    scatter1 = ax_main.scatter(all_eigenvalues.real, all_eigenvalues.imag, 
                               c='darkblue', s=8, alpha=0.4, edgecolors='none',
                               zorder=2, label=f'All eigenvalues (n={len(all_eigenvalues)})')
    
    # Create density overlay
    n_bins_overlay = 50
    H_overlay, xedges_o, yedges_o = np.histogram2d(all_eigenvalues.real, all_eigenvalues.imag, 
                                                   bins=n_bins_overlay, range=[real_range, imag_range])
    H_overlay = H_overlay.T
    
    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    H_smooth = gaussian_filter(H_overlay, sigma=1.5)
    
    # Translucent overlay only where there are eigenvalues
    H_masked = np.ma.masked_where(H_smooth < 0.5, H_smooth)
    
    # Add the translucent density overlay
    im = ax_main.imshow(H_masked, interpolation='bilinear', origin='lower',
                       extent=[xedges_o[0], xedges_o[-1], yedges_o[0], yedges_o[-1]],
                       cmap='YlOrRd', aspect='equal', alpha=0.25, vmin=1, 
                       vmax=np.max(H_smooth), zorder=1)
    
    # Create 1x1 grid for counting
    grid_size = 1.0
    x_bins = np.arange(real_range[0], real_range[1] + grid_size, grid_size)
    y_bins = np.arange(imag_range[0], imag_range[1] + grid_size, grid_size)
    H_counts, xedges_c, yedges_c = np.histogram2d(all_eigenvalues.real, all_eigenvalues.imag, 
                                                  bins=[x_bins, y_bins])
    
    # Add count labels to grid cells
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            count = int(H_counts[i, j])
            if count > 0:
                x_center = (x_bins[i] + x_bins[i+1]) / 2
                y_center = (y_bins[j] + y_bins[j+1]) / 2
                
                # Color based on count
                if count <= 5:
                    color = 'gray'
                    fontsize = 7
                elif count <= 20:
                    color = 'orange'
                    fontsize = 8
                else:
                    color = 'red'
                    fontsize = 9
                    
                ax_main.text(x_center, y_center, str(count), 
                           ha='center', va='center', color=color, 
                           fontsize=fontsize, fontweight='bold',
                           alpha=0.6, zorder=4)
    
    # Add grid lines
    ax_main.set_xticks(x_bins, minor=True)
    ax_main.set_yticks(y_bins, minor=True)
    ax_main.grid(True, which='minor', alpha=0.15, linewidth=0.5, color='gray')
    ax_main.grid(True, which='major', alpha=0.25, linewidth=1.0, color='gray')
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax_main.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, linewidth=1.2, zorder=3)
    
    # Mark overall max eigenvalue
    overall_max_idx = np.argmax(np.abs(all_eigenvalues))
    ax_main.scatter(all_eigenvalues[overall_max_idx].real, all_eigenvalues[overall_max_idx].imag,
                   c='lime', s=150, marker='*', edgecolors='black', linewidth=2,
                   zorder=10)
    
    # Crosshairs
    ax_main.axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=0)
    ax_main.axvline(0, color='black', linewidth=0.8, alpha=0.3, zorder=0)
    
    # Labels and title
    ax_main.set_xlabel('Real Part', fontsize=12)
    ax_main.set_ylabel('Imaginary Part', fontsize=12)
    ax_main.set_title(f'Combined Eigenvalue Spectrum - {n_simulations} Simulations', 
                     fontsize=14, fontweight='bold')
    ax_main.set_xlim(real_range)
    ax_main.set_ylim(imag_range)
    ax_main.set_aspect('equal')
    
    # Add legend with all elements
    from matplotlib.patches import Patch
    legend_elements = [
        scatter1,
        Patch(facecolor='yellow', alpha=0.25, label='Low density'),
        Patch(facecolor='red', alpha=0.25, label='High density'),
        Patch(facecolor='none', edgecolor='none', label='Grid counts:'),
        Patch(facecolor='gray', alpha=0.6, label='1-5 eigenvalues'),
        Patch(facecolor='orange', alpha=0.6, label='6-20 eigenvalues'),
        Patch(facecolor='red', alpha=0.6, label='>20 eigenvalues'),
    ]
    ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Stats text
    stats_text = f'Total: {len(all_eigenvalues)} eigenvalues\n'
    stats_text += f'From {n_simulations} simulations\n'
    stats_text += f'Max |λ| = {np.max(np.abs(all_eigenvalues)):.3f}'
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # ========== TOP RIGHT: Max eigenvalue distribution ==========
    ax_hist = fig.add_subplot(gs[0, 2])
    
    ax_hist.hist(all_max_eigenvalues, bins=min(20, n_simulations), 
                color='darkblue', alpha=0.7, edgecolor='black')
    ax_hist.axvline(np.mean(all_max_eigenvalues), color='red', linestyle='--', 
                   linewidth=2, label=f'μ={np.mean(all_max_eigenvalues):.2f}')
    ax_hist.axvspan(np.mean(all_max_eigenvalues) - np.std(all_max_eigenvalues),
                   np.mean(all_max_eigenvalues) + np.std(all_max_eigenvalues),
                   color='red', alpha=0.2)
    
    ax_hist.set_xlabel('Max Eigenvalue per Sim', fontsize=11)
    ax_hist.set_ylabel('Count', fontsize=11)
    ax_hist.set_title('Max λ Distribution', fontsize=12)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)
    
    # ========== BOTTOM LEFT: Pure scatter (no overlay) ==========
    ax_scatter = fig.add_subplot(gs[1, :2])
    
    # Just dots, no overlay for clarity
    ax_scatter.scatter(all_eigenvalues.real, all_eigenvalues.imag, 
                      c='darkblue', s=8, alpha=0.4, edgecolors='none')
    
    # Add the same grid
    ax_scatter.set_xticks(x_bins, minor=True)
    ax_scatter.set_yticks(y_bins, minor=True)
    ax_scatter.grid(True, which='minor', alpha=0.1, linewidth=0.5, color='gray')
    ax_scatter.grid(True, which='major', alpha=0.2, linewidth=1.0, color='gray')
    
    # Unit circle
    ax_scatter.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, linewidth=1.2)
    
    # Max eigenvalue
    ax_scatter.scatter(all_eigenvalues[overall_max_idx].real, all_eigenvalues[overall_max_idx].imag,
                      c='lime', s=150, marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    # Crosshairs
    ax_scatter.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    ax_scatter.axvline(0, color='black', linewidth=0.8, alpha=0.3)
    
    ax_scatter.set_xlabel('Real Part', fontsize=12)
    ax_scatter.set_ylabel('Imaginary Part', fontsize=12)
    ax_scatter.set_title('All Eigenvalues (No Overlay)', fontsize=12, fontweight='bold')
    ax_scatter.set_xlim(real_range)
    ax_scatter.set_ylim(imag_range)
    ax_scatter.set_aspect('equal')
    
    # ========== BOTTOM RIGHT: Eigenvalue magnitude ranking ==========
    ax_rank = fig.add_subplot(gs[1, 2])
    
    eigenvalue_mags = np.abs(all_eigenvalues)
    sorted_mags = np.sort(eigenvalue_mags)[::-1]
    
    ax_rank.semilogy(range(len(sorted_mags)), sorted_mags, 
                    'b-', linewidth=0.5, alpha=0.7)
    ax_rank.set_xlabel('Rank', fontsize=11)
    ax_rank.set_ylabel('|λ| (log)', fontsize=11)
    ax_rank.set_title('Ranked Magnitudes', fontsize=12)
    ax_rank.grid(True, alpha=0.3, which='both')
    
    # Overall title
    plt.suptitle(f'Combined Eigenvalue Analysis - {n_simulations} Simulations × {n_neurons_per_sim} Neurons', 
               fontsize=15, fontweight='bold')
    
    # Save
    savepath = os.path.join(saveloc, f'{pltname}.pdf')
    plt.savefig(savepath, format='pdf', dpi=300, bbox_inches='tight')
    if verbose:
        print(f"Combined plot saved to: {savepath}")
    plt.show()
    plt.close()

# ============================================
# FILE HANDLING FUNCTIONS
# ============================================

def get_h5_files(backup_path):
    """Get paths to all result.h5 files"""
    all_folders = [f for f in os.listdir(backup_path) 
                   if os.path.isdir(os.path.join(backup_path, f))]
    date_folders = [f for f in all_folders 
                    if f.startswith('202') or f.startswith('sim')]
    date_folders.sort()
    
    h5_files = []
    for folder in date_folders:
        h5_path = os.path.join(backup_path, folder, 'common', 'result.h5')
        if os.path.exists(h5_path):
            h5_files.append(h5_path)
            print(f"Found H5 file in: {folder}")
    
    return h5_files

def process_h5_file(file_path, starting_time, end_time):
    """Process a single .h5 file to extract raster data"""
    try:
        h5 = tables.open_file(file_path, 'r')
        data = h5.root
        
        if data.__contains__('Spikes'):
            print(f"Processing: {file_path}")
            
            # Get actual data dimensions
            actual_data_length = data.Spikes.shape[2]
            print(f"Data length: {actual_data_length}")
            
            # Define time window
            actual_end = min(end_time, actual_data_length) if end_time else actual_data_length
            
            # Load only the time window
            raster = data.Spikes[0, :, starting_time:actual_end]
            print(f"Loaded raster shape: {raster.shape}")
            
            return raster
        else:
            print(f"No 'Spikes' data found in {file_path}")
            return None
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None
    finally:
        try:
            h5.close()
        except:
            pass

#!/usr/bin/env python3
"""
Activity Matrix Visualization Functions
Visualize spike raster patterns and activity matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import tables
import os
try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    # Fallback if scipy is not available or import fails
    print("Warning: gaussian_filter1d not available, using simple smoothing")
    def gaussian_filter1d(array, sigma=1):
        # Simple moving average as fallback
        window_size = int(sigma * 3)
        if window_size < 1:
            window_size = 1
        kernel = np.ones(window_size) / window_size
        if len(array.shape) == 1:
            return np.convolve(array, kernel, mode='same')
        else:
            result = np.zeros_like(array)
            for i in range(array.shape[0]):
                result[i] = np.convolve(array[i], kernel, mode='same')
            return result

def plot_activity_matrix(raster, time_window=None, neuron_subset=None, 
                         bin_size=1, smooth=False, title="Neural Activity Matrix",
                         save_path=None, figsize=(15, 8), cmap='hot',
                         show_avalanches=False, avalanche_results=None):
    """
    Plot the activity matrix (spike raster) with various visualization options.
    
    Parameters:
    -----------
    raster : np.ndarray
        Spike raster data (n_neurons x n_timesteps)
    time_window : tuple or None
        (start, end) time points to display
    neuron_subset : tuple or None
        (start, end) neurons to display
    bin_size : int
        Temporal binning factor for visualization
    smooth : bool
        Apply smoothing for better visualization
    title : str
        Plot title
    save_path : str or None
        Path to save figure
    figsize : tuple
        Figure size
    cmap : str
        Colormap to use
    show_avalanches : bool
        Highlight avalanche periods
    avalanche_results : dict
        Results from avalanche detection
    """
    
    # Apply time window if specified
    if time_window:
        raster = raster[:, time_window[0]:time_window[1]]
    
    # Apply neuron subset if specified
    if neuron_subset:
        raster = raster[neuron_subset[0]:neuron_subset[1], :]
    
    n_neurons, n_timesteps = raster.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 1], width_ratios=[4, 1, 0.5],
                         hspace=0.02, wspace=0.02)
    
    # Main raster plot
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Apply temporal binning if requested
    if bin_size > 1:
        n_bins = n_timesteps // bin_size
        binned_raster = np.zeros((n_neurons, n_bins))
        for i in range(n_bins):
            binned_raster[:, i] = np.sum(raster[:, i*bin_size:(i+1)*bin_size], axis=1)
        display_raster = binned_raster
        time_axis = np.arange(n_bins) * bin_size
    else:
        display_raster = raster.copy()
        time_axis = np.arange(n_timesteps)
    
    # Apply smoothing if requested
    if smooth:
        # Smooth each neuron's activity independently
        try:
            for i in range(n_neurons):
                display_raster[i, :] = gaussian_filter1d(display_raster[i, :], sigma=1.5)
        except:
            # Fallback to simple moving average
            window_size = 3
            kernel = np.ones(window_size) / window_size
            for i in range(n_neurons):
                display_raster[i, :] = np.convolve(display_raster[i, :], kernel, mode='same')
    
    # Plot the main raster
    im = ax_main.imshow(display_raster, aspect='auto', cmap=cmap,
                        interpolation='nearest', origin='lower')
    
    # Add avalanche highlights if requested
    if show_avalanches and avalanche_results:
        avalanche_locs = avalanche_results.get('loc', [])
        for loc in avalanche_locs:
            if time_window:
                if loc >= time_window[0] and loc < time_window[1]:
                    ax_main.axvline(loc - time_window[0], color='cyan', 
                                   alpha=0.3, linewidth=0.5)
            else:
                if loc < n_timesteps:
                    ax_main.axvline(loc, color='cyan', alpha=0.3, linewidth=0.5)
    
    ax_main.set_xlabel('Time (steps)', fontsize=11)
    ax_main.set_ylabel('Neuron Index', fontsize=11)
    ax_main.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Population activity (top)
    ax_pop = fig.add_subplot(gs[0, 0], sharex=ax_main)
    population_activity = np.sum(display_raster, axis=0)
    ax_pop.plot(time_axis, population_activity, color='darkblue', linewidth=1)
    ax_pop.fill_between(time_axis, 0, population_activity, alpha=0.3, color='darkblue')
    ax_pop.set_ylabel('Pop.\nActivity', fontsize=9)
    ax_pop.set_xlim([0, len(time_axis)-1])
    ax_pop.grid(True, alpha=0.3)
    plt.setp(ax_pop.get_xticklabels(), visible=False)
    
    # Mean firing rate per neuron (right)
    ax_rate = fig.add_subplot(gs[1, 1], sharey=ax_main)
    mean_rates = np.mean(display_raster, axis=1)
    ax_rate.plot(mean_rates, np.arange(n_neurons), color='darkgreen', linewidth=1)
    ax_rate.fill_betweenx(np.arange(n_neurons), 0, mean_rates, alpha=0.3, color='darkgreen')
    ax_rate.set_xlabel('Mean\nRate', fontsize=9)
    ax_rate.set_ylim([0, n_neurons-1])
    ax_rate.grid(True, alpha=0.3)
    plt.setp(ax_rate.get_yticklabels(), visible=False)
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    plt.colorbar(im, cax=ax_cbar, label='Activity')
    
    # Statistics box (bottom)
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_stats.axis('off')
    
    # Calculate statistics
    total_spikes = np.sum(raster)
    mean_rate = total_spikes / (n_neurons * n_timesteps)
    active_neurons = np.sum(np.any(raster > 0, axis=1))
    sparsity = 1 - (np.count_nonzero(raster) / raster.size)
    
    stats_text = (f"Total spikes: {int(total_spikes):,} | "
                 f"Mean rate: {mean_rate:.3f} | "
                 f"Active neurons: {active_neurons}/{n_neurons} | "
                 f"Sparsity: {sparsity:.1%}")
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(f'Neural Activity Matrix - {n_neurons} neurons × {n_timesteps} timesteps',
                fontsize=12, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activity matrix plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_activity_matrix_detailed(raster, time_window=None, 
                                 save_path=None, title_prefix=""):
    """
    Create a detailed multi-panel visualization of the activity matrix.
    
    Parameters:
    -----------
    raster : np.ndarray
        Spike raster data (n_neurons x n_timesteps)
    time_window : tuple or None
        (start, end) time points to display
    save_path : str or None
        Path to save figure
    title_prefix : str
        Prefix for the title
    """
    
    # Apply time window if specified
    if time_window:
        raster = raster[:, time_window[0]:time_window[1]]
    
    n_neurons, n_timesteps = raster.shape
    
    # Create figure with multiple visualization styles
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Raw spike raster (binary)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(raster, aspect='auto', cmap='Greys', interpolation='nearest',
              origin='lower', vmin=0, vmax=1)
    ax1.set_title('Raw Spike Raster', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=9)
    ax1.set_ylabel('Neuron', fontsize=9)
    
    # 2. Smoothed activity (filtered)
    ax2 = fig.add_subplot(gs[0, 1])
    # Create smoothed version
    smoothed = np.zeros_like(raster, dtype=float)
    for i in range(n_neurons):
        # Simple moving average smoothing
        window_size = 5
        kernel = np.ones(window_size) / window_size
        smoothed[i, :] = np.convolve(raster[i, :].astype(float), kernel, mode='same')
    im2 = ax2.imshow(smoothed, aspect='auto', cmap='hot', interpolation='bilinear',
                    origin='lower')
    ax2.set_title('Smoothed Activity', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=9)
    ax2.set_ylabel('Neuron', fontsize=9)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. Binned activity (10ms bins)
    ax3 = fig.add_subplot(gs[0, 2])
    bin_size = 10
    n_bins = n_timesteps // bin_size
    binned = np.zeros((n_neurons, n_bins))
    for i in range(n_bins):
        binned[:, i] = np.sum(raster[:, i*bin_size:(i+1)*bin_size], axis=1)
    im3 = ax3.imshow(binned, aspect='auto', cmap='viridis', interpolation='nearest',
                    origin='lower')
    ax3.set_title(f'Binned Activity ({bin_size}ms bins)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (bins)', fontsize=9)
    ax3.set_ylabel('Neuron', fontsize=9)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Population activity over time
    ax4 = fig.add_subplot(gs[1, :])
    pop_activity = np.sum(raster, axis=0)
    ax4.plot(pop_activity, color='darkblue', linewidth=0.5, alpha=0.8)
    ax4.fill_between(range(len(pop_activity)), 0, pop_activity, 
                     alpha=0.3, color='darkblue')
    
    # Mark high activity periods
    threshold = np.percentile(pop_activity, 90)
    high_activity = pop_activity > threshold
    ax4.fill_between(range(len(pop_activity)), 0, np.max(pop_activity),
                     where=high_activity, alpha=0.2, color='red', 
                     label=f'High activity (>P90={threshold:.0f})')
    
    ax4.set_xlabel('Time (steps)', fontsize=11)
    ax4.set_ylabel('Population Activity\n(# active neurons)', fontsize=11)
    ax4.set_title('Population Activity Timeline', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # 5. Firing rate distribution
    ax5 = fig.add_subplot(gs[2, 0])
    firing_rates = np.mean(raster, axis=1) * 1000  # Convert to Hz if needed
    ax5.hist(firing_rates, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(firing_rates), color='red', linestyle='--', 
               label=f'Mean: {np.mean(firing_rates):.2f}')
    ax5.set_xlabel('Firing Rate', fontsize=10)
    ax5.set_ylabel('# Neurons', fontsize=10)
    ax5.set_title('Firing Rate Distribution', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Inter-spike interval distribution
    ax6 = fig.add_subplot(gs[2, 1])
    isis = []
    for i in range(n_neurons):
        spike_times = np.where(raster[i, :] > 0)[0]
        if len(spike_times) > 1:
            isis.extend(np.diff(spike_times))
    
    if isis:
        ax6.hist(isis, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Inter-Spike Interval (steps)', fontsize=10)
        ax6.set_ylabel('Count', fontsize=10)
        ax6.set_title('ISI Distribution', fontsize=11, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
    
    # 7. Correlation matrix (subset for visibility)
    ax7 = fig.add_subplot(gs[2, 2])
    # Take a subset of neurons for correlation calculation
    subset_size = min(50, n_neurons)
    subset_raster = raster[:subset_size, :]
    corr_matrix = np.corrcoef(subset_raster)
    im7 = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                    aspect='auto', interpolation='nearest')
    ax7.set_title(f'Correlation Matrix (first {subset_size} neurons)', 
                 fontsize=11, fontweight='bold')
    ax7.set_xlabel('Neuron', fontsize=9)
    ax7.set_ylabel('Neuron', fontsize=9)
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    # Overall title
    suptitle = f'{title_prefix} Activity Matrix Analysis' if title_prefix else 'Activity Matrix Analysis'
    plt.suptitle(suptitle + f' - {n_neurons} neurons × {n_timesteps} timesteps',
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed activity matrix plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_activity_comparison(rasters_list, titles=None, time_window=None,
                            save_path=None, main_title="Activity Comparison"):
    """
    Compare activity matrices from multiple simulations side by side.
    
    Parameters:
    -----------
    rasters_list : list of np.ndarray
        List of spike rasters to compare
    titles : list of str or None
        Titles for each raster
    time_window : tuple or None
        (start, end) time window to display
    save_path : str or None
        Path to save figure
    main_title : str
        Main title for the figure
    """
    
    n_rasters = len(rasters_list)
    if titles is None:
        titles = [f'Simulation {i+1}' for i in range(n_rasters)]
    
    # Create figure
    fig, axes = plt.subplots(2, n_rasters, figsize=(5*n_rasters, 8),
                            gridspec_kw={'height_ratios': [3, 1]})
    
    if n_rasters == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, raster in enumerate(rasters_list):
        # Apply time window if specified
        if time_window:
            raster = raster[:, time_window[0]:time_window[1]]
        
        n_neurons, n_timesteps = raster.shape
        
        # Plot raster
        ax_raster = axes[0, idx]
        im = ax_raster.imshow(raster, aspect='auto', cmap='hot',
                             interpolation='nearest', origin='lower')
        ax_raster.set_title(titles[idx], fontsize=11, fontweight='bold')
        ax_raster.set_xlabel('Time', fontsize=10)
        if idx == 0:
            ax_raster.set_ylabel('Neuron Index', fontsize=10)
        
        # Plot population activity
        ax_pop = axes[1, idx]
        pop_activity = np.sum(raster, axis=0)
        ax_pop.plot(pop_activity, color='darkblue', linewidth=0.8)
        ax_pop.fill_between(range(len(pop_activity)), 0, pop_activity,
                          alpha=0.3, color='darkblue')
        ax_pop.set_xlabel('Time', fontsize=10)
        if idx == 0:
            ax_pop.set_ylabel('Population Activity', fontsize=10)
        ax_pop.grid(True, alpha=0.3)
        
        # Add statistics
        mean_rate = np.mean(raster)
        sparsity = 1 - np.count_nonzero(raster) / raster.size
        stats_text = f'Rate: {mean_rate:.3f}\nSparsity: {sparsity:.1%}'
        ax_raster.text(0.02, 0.98, stats_text, transform=ax_raster.transAxes,
                      fontsize=9, verticalalignment='top', color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.suptitle(main_title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activity comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_avalanche_highlighted_matrix(raster, avalanche_results, 
                                     time_window=None, save_path=None):
    """
    Plot activity matrix with avalanches highlighted and annotated.
    
    Parameters:
    -----------
    raster : np.ndarray
        Spike raster data
    avalanche_results : dict
        Results from avalanche detection
    time_window : tuple or None
        Time window to display
    save_path : str or None
        Path to save figure
    """
    
    if time_window:
        raster = raster[:, time_window[0]:time_window[1]]
    
    n_neurons, n_timesteps = raster.shape
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 3, 1], width_ratios=[3, 1],
                         hspace=0.1, wspace=0.2)
    
    # Main raster with avalanche highlights
    ax_main = fig.add_subplot(gs[1, 0])
    im = ax_main.imshow(raster, aspect='auto', cmap='Greys',
                       interpolation='nearest', origin='lower', alpha=0.8)
    
    # Overlay avalanche periods
    if 'loc' in avalanche_results:
        avalanche_locs = avalanche_results['loc']
        avalanche_sizes = avalanche_results['Size']
        avalanche_durations = avalanche_results['Duration']
        
        # Color avalanches by size
        size_colors = plt.cm.jet(np.log(avalanche_sizes) / np.log(np.max(avalanche_sizes)))
        
        for i, (loc, size, duration) in enumerate(zip(avalanche_locs[:100], 
                                                       avalanche_sizes[:100],
                                                       avalanche_durations[:100])):
            if time_window:
                if loc >= time_window[0] and loc + duration <= time_window[1]:
                    loc_adj = loc - time_window[0]
                    rect = patches.Rectangle((loc_adj, 0), duration, n_neurons,
                                           linewidth=0, edgecolor='none',
                                           facecolor=size_colors[i], alpha=0.3)
                    ax_main.add_patch(rect)
            else:
                if loc + duration <= n_timesteps:
                    rect = patches.Rectangle((loc, 0), duration, n_neurons,
                                           linewidth=0, edgecolor='none',
                                           facecolor=size_colors[i], alpha=0.3)
                    ax_main.add_patch(rect)
    
    ax_main.set_xlabel('Time (steps)', fontsize=11)
    ax_main.set_ylabel('Neuron Index', fontsize=11)
    ax_main.set_title('Neural Activity with Avalanche Periods Highlighted', 
                     fontsize=12, fontweight='bold')
    
    # Population activity with avalanche markers (top)
    ax_pop = fig.add_subplot(gs[0, 0], sharex=ax_main)
    pop_activity = np.sum(raster, axis=0)
    ax_pop.plot(pop_activity, color='black', linewidth=0.8)
    
    # Mark avalanche starts
    if 'loc' in avalanche_results:
        for loc in avalanche_locs[:100]:
            if time_window:
                if loc >= time_window[0] and loc < time_window[1]:
                    ax_pop.axvline(loc - time_window[0], color='red', 
                                  alpha=0.5, linewidth=0.5)
            else:
                if loc < n_timesteps:
                    ax_pop.axvline(loc, color='red', alpha=0.5, linewidth=0.5)
    
    ax_pop.set_ylabel('Population\nActivity', fontsize=10)
    ax_pop.grid(True, alpha=0.3)
    plt.setp(ax_pop.get_xticklabels(), visible=False)
    
    # Avalanche size distribution (right top)
    ax_size = fig.add_subplot(gs[0, 1])
    if 'Size' in avalanche_results:
        sizes = avalanche_results['Size']
        ax_size.hist(np.log10(sizes + 1), bins=30, color='darkred', 
                    alpha=0.7, edgecolor='black')
        ax_size.set_xlabel('log₁₀(Size)', fontsize=10)
        ax_size.set_ylabel('Count', fontsize=10)
        ax_size.set_title('Avalanche Sizes', fontsize=11)
        ax_size.grid(True, alpha=0.3)
    
    # Avalanche duration distribution (right middle)
    ax_dur = fig.add_subplot(gs[1, 1])
    if 'T' in avalanche_results:
        durations = avalanche_results['T']
        ax_dur.hist(np.log10(durations + 1), bins=30, color='darkgreen',
                   alpha=0.7, edgecolor='black')
        ax_dur.set_xlabel('log₁₀(Duration)', fontsize=10)
        ax_dur.set_ylabel('Count', fontsize=10)
        ax_dur.set_title('Avalanche Durations', fontsize=11)
        ax_dur.grid(True, alpha=0.3)
    
    # Statistics (bottom)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')

    stats_text = f"Total avalanches: {len(avalanche_results['Size'])} | "
    stats_text += f"Mean size: {np.mean(avalanche_results['Size']):.1f} | "
    stats_text += f"Max size: {np.max(avalanche_results['Size']):.0f} | "
    stats_text += f"Mean duration: {np.mean(avalanche_results['Duration']):.1f} | "
    stats_text += f"Max duration: {np.max(avalanche_results['Duration']):.0f}"

    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                 ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Avalanche Analysis on Activity Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Avalanche-highlighted matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


#!/usr/bin/env python3
"""
Complete integration of activity matrix visualization into your main code.
Add these functions to your existing script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# First, add this simple smoothing function if scipy is not available
def simple_smooth(array, window_size=3):
    """Simple moving average smoothing"""
    kernel = np.ones(window_size) / window_size
    if len(array.shape) == 1:
        return np.convolve(array, kernel, mode='same')
    else:
        result = np.zeros_like(array)
        for i in range(array.shape[0]):
            result[i] = np.convolve(array[i], kernel, mode='same')
        return result

def plot_simple_activity_matrix(raster, title="Neural Activity", save_path=None):
    """
    Simple but effective activity matrix visualization.
    This is a minimal version that definitely works.
    """
    n_neurons, n_timesteps = raster.shape
    
    # Limit display to manageable size
    max_time_display = min(10000, n_timesteps)
    display_raster = raster[:, :max_time_display]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                            gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Spike raster
    im = axes[0].imshow(display_raster, aspect='auto', cmap='hot',
                       interpolation='nearest', origin='lower')
    axes[0].set_xlabel('Time (steps)', fontsize=11)
    axes[0].set_ylabel('Neuron Index', fontsize=11)
    axes[0].set_title(title, fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0], fraction=0.02)
    cbar.set_label('Activity', fontsize=10)
    
    # Plot 2: Population activity
    pop_activity = np.sum(display_raster, axis=0)
    axes[1].plot(pop_activity, color='darkblue', linewidth=0.8)
    axes[1].fill_between(range(len(pop_activity)), 0, pop_activity,
                        alpha=0.3, color='darkblue')
    axes[1].set_xlabel('Time (steps)', fontsize=11)
    axes[1].set_ylabel('Population Activity', fontsize=11)
    axes[1].set_xlim([0, max_time_display])
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    total_spikes = np.sum(display_raster)
    mean_rate = total_spikes / (n_neurons * max_time_display)
    active_neurons = np.sum(np.any(display_raster > 0, axis=1))
    
    stats_text = (f"Neurons: {n_neurons} | Time: {max_time_display} steps | "
                 f"Total spikes: {int(total_spikes):,} | "
                 f"Mean rate: {mean_rate:.4f} | "
                 f"Active neurons: {active_neurons}/{n_neurons}")
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
        plt.close()
    else:
        plt.show()

def plot_activity_with_avalanches(raster, avalanche_results, title="Activity with Avalanches", 
                                 save_path=None):
    """
    Plot activity matrix with avalanche periods highlighted.
    """
    n_neurons, n_timesteps = raster.shape
    max_time_display = min(5000, n_timesteps)
    display_raster = raster[:, :max_time_display]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                           gridspec_kw={'height_ratios': [1, 3, 1]})
    
    # Top: Population activity with avalanche markers
    pop_activity = np.sum(display_raster, axis=0)
    axes[0].plot(pop_activity, color='black', linewidth=0.8)
    axes[0].fill_between(range(len(pop_activity)), 0, pop_activity,
                        alpha=0.3, color='gray')
    
    # Mark avalanche starts if available
    if avalanche_results and 'loc' in avalanche_results:
        for loc in avalanche_results['loc'][:100]:  # Show first 100
            if loc < max_time_display:
                axes[0].axvline(loc, color='red', alpha=0.3, linewidth=0.5)
    
    axes[0].set_ylabel('Population Activity', fontsize=10)
    axes[0].set_xlim([0, max_time_display])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=12, fontweight='bold')
    
    # Middle: Spike raster with avalanche overlay
    im = axes[1].imshow(display_raster, aspect='auto', cmap='gray_r',
                       interpolation='nearest', origin='lower', alpha=0.8)
    
    # Overlay avalanche periods with colors
    if avalanche_results and all(k in avalanche_results for k in ['loc', 'Size', 'Duration']):
        avalanche_locs = avalanche_results['loc']
        avalanche_sizes = avalanche_results['Size']
        avalanche_durations = avalanche_results['Duration']

        # Normalize sizes for coloring
        if len(avalanche_sizes) > 0:
            size_norm = avalanche_sizes / np.max(avalanche_sizes)
            
            for i, (loc, size, duration) in enumerate(zip(
                avalanche_locs[:50], avalanche_sizes[:50], avalanche_durations[:50]
            )):
                if loc + duration <= max_time_display:
                    # Create colored rectangle for each avalanche
                    color = plt.cm.jet(size_norm[i])
                    rect = patches.Rectangle((loc, 0), duration, n_neurons,
                                           linewidth=0, edgecolor='none',
                                           facecolor=color, alpha=0.2)
                    axes[1].add_patch(rect)
    
    axes[1].set_xlabel('Time (steps)', fontsize=11)
    axes[1].set_ylabel('Neuron Index', fontsize=11)
    axes[1].set_xlim([0, max_time_display])
    
    # Bottom: Avalanche size timeline
    if avalanche_results and 'loc' in avalanche_results and 'Size' in avalanche_results:
        locs = avalanche_results['loc']
        sizes = avalanche_results['Size']

        # Only plot avalanches within display window
        mask = locs < max_time_display
        display_locs = locs[mask][:100]  # Limit to first 100
        display_sizes = sizes[mask][:100]
        
        if len(display_locs) > 0:
            axes[2].scatter(display_locs, display_sizes, alpha=0.5, s=20, c='darkred')
            axes[2].set_yscale('log')
            axes[2].set_xlabel('Time (steps)', fontsize=11)
            axes[2].set_ylabel('Avalanche Size', fontsize=10)
            axes[2].set_xlim([0, max_time_display])
            axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No avalanche data available', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_xlabel('Time (steps)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
        plt.close()
    else:
        plt.show()

def plot_activity_comparison_simple(rasters_list, titles=None, save_path=None):
    """
    Compare multiple activity matrices side by side.
    """
    n_rasters = min(len(rasters_list), 4)  # Max 4 for visibility
    
    if titles is None:
        titles = [f'Simulation {i+1}' for i in range(n_rasters)]
    
    fig, axes = plt.subplots(2, n_rasters, figsize=(5*n_rasters, 8),
                            gridspec_kw={'height_ratios': [3, 1]})
    
    if n_rasters == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_rasters):
        raster = rasters_list[idx]
        n_neurons, n_timesteps = raster.shape
        max_time = min(5000, n_timesteps)
        display_raster = raster[:, :max_time]
        
        # Spike raster
        im = axes[0, idx].imshow(display_raster, aspect='auto', cmap='hot',
                                interpolation='nearest', origin='lower')
        axes[0, idx].set_title(titles[idx], fontsize=11, fontweight='bold')
        if idx == 0:
            axes[0, idx].set_ylabel('Neuron Index', fontsize=10)
        axes[0, idx].set_xlabel('Time', fontsize=10)
        
        # Population activity
        pop_activity = np.sum(display_raster, axis=0)
        axes[1, idx].plot(pop_activity, color='darkblue', linewidth=0.8)
        axes[1, idx].fill_between(range(len(pop_activity)), 0, pop_activity,
                                 alpha=0.3, color='darkblue')
        if idx == 0:
            axes[1, idx].set_ylabel('Pop. Activity', fontsize=10)
        axes[1, idx].set_xlabel('Time', fontsize=10)
        axes[1, idx].grid(True, alpha=0.3)
        
        # Add basic stats
        mean_rate = np.mean(display_raster)
        active = np.sum(np.any(display_raster > 0, axis=1))
        axes[0, idx].text(0.02, 0.98, f'Rate: {mean_rate:.3f}\nActive: {active}/{n_neurons}',
                         transform=axes[0, idx].transAxes, fontsize=8,
                         verticalalignment='top', color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.suptitle('Activity Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
        plt.close()
    else:
        plt.show()

def visualize_activity_matrices(h5_files, starting_time, end_time, base_dir,
                               avalanche_results=None):
    """
    Main function to visualize activity matrices from all files.
    This is what you call from your main() function.
    """
    print("\nVisualizing activity matrices...")
    
    all_rasters = []
    
    for idx, h5_file in enumerate(h5_files):
        print(f"\nProcessing visualization for simulation {idx+1}/{len(h5_files)}")
        
        # Load raster data using your existing function
        raster = process_h5_file(h5_file, starting_time, end_time)
        
        if raster is not None:
            all_rasters.append(raster)
            
            # Create simple activity matrix plot
            save_path = os.path.join(base_dir, f'activity_matrix_sim{idx:02d}.pdf')
            plot_simple_activity_matrix(
                raster=raster,
                title=f"Simulation {idx+1} - Neural Activity",
                save_path=save_path
            )
            
            # If avalanche results exist, create avalanche-highlighted plot
            if avalanche_results is not None:
                save_path_aval = os.path.join(base_dir, f'activity_avalanches_sim{idx:02d}.pdf')
                plot_activity_with_avalanches(
                    raster=raster,
                    avalanche_results=avalanche_results,
                    title=f"Simulation {idx+1} - Activity with Avalanches",
                    save_path=save_path_aval
                )
    
    # Create comparison plot if multiple simulations
    if len(all_rasters) > 1:
        print(f"\nCreating comparison plot for {len(all_rasters)} simulations...")
        save_path_comp = os.path.join(base_dir, 'activity_comparison.pdf')
        plot_activity_comparison_simple(
            rasters_list=all_rasters,
            save_path=save_path_comp
        )
    
    print(f"\nActivity visualization complete! Created {len(all_rasters)*2 + (1 if len(all_rasters)>1 else 0)} plots")
    
    return all_rasters  # Return for potential further analysis


# ============================================
# MAIN ANALYSIS
# ============================================

# Updated main() function - add this to your existing code
# This shows how to integrate the network visualization

def main():
    print(f"\n{'='*60}")
    print("SIMPLIFIED AVALANCHE ANALYSIS WITH NETWORK VISUALIZATION")
    print(f"{'='*60}\n")
    
    print(f"Base directory: {base_dir}")
    print(f"Time window: {starting_time_point} to {end_time_point}")
    
    # Initialize storage for all avalanches
    all_burst = np.array([])
    all_T = np.array([], dtype=int)
    all_shapes = []
    
    # Get list of H5 files
    h5_files = get_h5_files(base_dir)
    print(f"\nFound {len(h5_files)} H5 files to analyze")
    
    if len(h5_files) == 0:
        print("No H5 files found. Exiting.")
        return
    
    # ============================================
    # AVALANCHE ANALYSIS (EXISTING)
    # ============================================
    print(f"\n{'='*60}")
    print("AVALANCHE DETECTION AND ANALYSIS")
    print(f"{'='*60}")
    
    # Process each file for avalanches
    for file_idx, file_path in enumerate(h5_files):
        print(f"\n{'='*50}")
        print(f"Processing file {file_idx+1}/{len(h5_files)} for avalanches")
        print(f"{'='*50}")
        
        # Load raster data
        raster = process_h5_file(file_path, starting_time_point, end_time_point)
        
        if raster is not None:
            # Find avalanches
            print("\nDetecting avalanches...")
            results = get_avalanches(
                raster, 
                perc=AVALANCHE_PARAMS['perc_threshold'],
                const_threshold=AVALANCHE_PARAMS['const_threshold']
            )

            if len(results['Size']) > 0:
                print(f"Found {len(results['Size'])} avalanches")

                # Accumulate results
                all_burst = np.concatenate((all_burst, results['Size']))
                all_T = np.concatenate((all_T, results['Duration']))

                # Get shapes
                shapes = find_avalanches(raster)
                all_shapes.extend(shapes)
                print(f"Found {len(shapes)} avalanche shapes")
            else:
                print("No avalanches found in this file")
    
    # Analyze combined avalanches
    print(f"\n{'='*60}")
    print("COMBINED AVALANCHE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total avalanches: {len(all_burst)}")
    
    if len(all_burst) > 0:
        # Run full analysis
        print("\nAnalyzing avalanche distributions...")
        
        # Import criticality module only when needed
        try:
            import criticality as cr
            
            AV_Result = cr.AV_analysis(
                burst=all_burst,
                T=all_T,
                flag=1,
                bm=AVALANCHE_PARAMS['size_bm'],
                tm=AVALANCHE_PARAMS['duration_tm'],
                nfactor_bm=AVALANCHE_PARAMS['size_nfactor'],
                nfactor_tm=AVALANCHE_PARAMS['duration_nfactor'],
                nfactor_bm_tail=AVALANCHE_PARAMS['size_tail_cutoff'],
                nfactor_tm_tail=AVALANCHE_PARAMS['duration_tail_cutoff'],
                none_fact=AVALANCHE_PARAMS['none_factor'],
                verbose=True,
                exclude=True,
                exclude_burst=AVALANCHE_PARAMS['exclude_burst_min'],
                exclude_time=AVALANCHE_PARAMS['exclude_time_min'],
                exclude_diff_b=AVALANCHE_PARAMS['exclude_burst_diff'],
                exclude_diff_t=AVALANCHE_PARAMS['exclude_time_diff'],
                plot=True,
                pltname='avalanche_analysis',
                saveloc=base_dir
            )
            
            # Print results
            print("\n" + "="*40)
            print("ANALYSIS RESULTS")
            print("="*40)
            print(f"Alpha (size exponent): {AV_Result['alpha']:.3f}")
            print(f"Beta (duration exponent): {AV_Result['beta']:.3f}")
            print(f"Size range: {AV_Result['xmin']:.0f} to {AV_Result['xmax']:.0f}")
            print(f"Duration range: {AV_Result['tmin']:.0f} to {AV_Result['tmax']:.0f}")
            print(f"Scaling relation difference: {AV_Result['df']:.3f}")
            
            # Save results
            results_df = pd.DataFrame({
                'Metric': ['Alpha', 'Beta', 'Size_xmin', 'Size_xmax', 
                          'Duration_tmin', 'Duration_tmax', 'Scaling_diff'],
                'Value': [AV_Result['alpha'], AV_Result['beta'], 
                         AV_Result['xmin'], AV_Result['xmax'],
                         AV_Result['tmin'], AV_Result['tmax'], 
                         AV_Result['df']]
            })
            
            csv_path = os.path.join(base_dir, 'avalanche_analysis_results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
            
        except ImportError:
            print("\nWARNING: criticality module not found. Using simplified analysis.")
            AV_Result = AV_analysis(
                burst=all_burst,
                T=all_T,
                flag=1,
                **{k: v for k, v in AVALANCHE_PARAMS.items() 
                   if k not in ['perc_threshold', 'const_threshold']},
                plot=True,
                pltname='avalanche_analysis_simple',
                saveloc=base_dir
            )
    else:
        print("No avalanches found in any files.")

    # ============================================
    # ACTIVITY MATRIX VISUALIZATION (ADD THIS HERE!)
    # ============================================
    print(f"\n{'='*60}")
    print("ACTIVITY MATRIX VISUALIZATION")
    print(f"{'='*60}")
    
    # Visualize the activity matrices
    all_rasters = visualize_activity_matrices(
        h5_files=h5_files,
        starting_time=starting_time_point,
        end_time=end_time_point,
        base_dir=base_dir,
        avalanche_results=AV_Result  # Pass avalanche results if available
    )
    
    # ============================================
    # EIGENVALUE SPECTRUM ANALYSIS (EXISTING)
    # ============================================
    if len(h5_files) > 0:
        print(f"\n{'='*60}")
        print("EIGENVALUE SPECTRUM ANALYSIS")
        print(f"{'='*60}")

        combined_results = analyze_eigenvalues_batch(
            h5_files=h5_files,
            starting_time=starting_time_point,
            end_time=end_time_point,
            base_dir=base_dir,
            combine_analysis=True,
            n_exc=200,
            n_inh=40,
            lag=1,
            n_shuffles=50,
            verbose=True
        )
        
        if combined_results and 'all_eigenvalues' in combined_results:
            print(f"\nCombined Analysis Complete:")
            print(f"  Total eigenvalues: {len(combined_results['all_eigenvalues'])}")
            print(f"  Mean max eigenvalue: {combined_results['mean_max_eigenvalue']:.4f}")
            print(f"  Std max eigenvalue: {combined_results['std_max_eigenvalue']:.4f}")
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE!")
    print("="*60)

# Integration function to add to your main code
def visualize_activity_matrices(h5_files, starting_time, end_time, base_dir,
                               avalanche_results=None):
    """
    Add this function to your main code to visualize activity matrices.
    
    Parameters:
    -----------
    h5_files : list
        List of H5 file paths
    starting_time : int
        Start time point
    end_time : int  
        End time point
    base_dir : str
        Directory to save plots
    avalanche_results : dict or None
        Avalanche detection results if available
    """
    
    print(f"\n{'='*60}")
    print("ACTIVITY MATRIX VISUALIZATION")
    print(f"{'='*60}")
    
    all_rasters = []
    
    for idx, h5_file in enumerate(h5_files):
        print(f"\nProcessing activity matrix for simulation {idx+1}/{len(h5_files)}")
        
        # Load raster data
        raster = process_h5_file(h5_file, starting_time, end_time)
        
        if raster is not None:
            all_rasters.append(raster)
            
            # Plot individual activity matrix
            save_path = os.path.join(base_dir, f'activity_matrix_sim{idx:02d}.pdf')
            plot_activity_matrix(
                raster=raster,
                time_window=(0, min(5000, raster.shape[1])),  # Show first 5000 steps
                title=f"Simulation {idx+1} - Neural Activity",
                save_path=save_path,
                show_avalanches=(avalanche_results is not None)
            )
            
            # Plot detailed analysis
            save_path_detail = os.path.join(base_dir, f'activity_detailed_sim{idx:02d}.pdf')
            plot_activity_matrix_detailed(
                raster=raster,
                time_window=(0, min(10000, raster.shape[1])),
                save_path=save_path_detail,
                title_prefix=f"Simulation {idx+1}"
            )
            
            # If we have avalanche results, create highlighted plot
            if avalanche_results:
                save_path_aval = os.path.join(base_dir, f'activity_avalanches_sim{idx:02d}.pdf')
                plot_avalanche_highlighted_matrix(
                    raster=raster,
                    avalanche_results=avalanche_results,
                    time_window=(0, min(5000, raster.shape[1])),
                    save_path=save_path_aval
                )
    
    # Create comparison plot if multiple simulations
    if len(all_rasters) > 1:
        save_path_comp = os.path.join(base_dir, 'activity_comparison.pdf')
        plot_activity_comparison(
            rasters_list=all_rasters[:4],  # Compare up to 4 simulations
            time_window=(0, 5000),
            save_path=save_path_comp,
            main_title="Activity Matrix Comparison Across Simulations"
        )
    
    print(f"\nActivity matrix visualization complete!")
    print(f"Plots saved to: {base_dir}")
    
    return all_rasters

def create_network_comparison_plot(all_stats, base_dir):
    """Create comparison plot for network statistics across simulations."""
    
    print(f"\n{'='*40}")
    print("NETWORK COMPARISON ACROSS SIMULATIONS")
    print(f"{'='*40}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract statistics for plotting
    sparsities = [s.get('sparsity', 0) for s in all_stats]
    mean_weights = [s.get('mean_weight', 0) for s in all_stats]
    spectral_radii = [s.get('spectral_radius', 0) for s in all_stats if s.get('spectral_radius')]
    mean_degrees = [s.get('mean_in_degree', 0) for s in all_stats]
    max_weights = [s.get('max_weight', 0) for s in all_stats]
    n_connections = [s.get('n_connections', 0) for s in all_stats]
    
    # Plot statistics
    sim_indices = range(len(all_stats))
    
    # Sparsity
    axes[0, 0].bar(sim_indices, sparsities, color='darkblue', alpha=0.7)
    axes[0, 0].set_xlabel('Simulation', fontsize=11)
    axes[0, 0].set_ylabel('Sparsity', fontsize=11)
    axes[0, 0].set_title('Network Sparsity', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean Weight
    axes[0, 1].bar(sim_indices, mean_weights, color='darkgreen', alpha=0.7)
    axes[0, 1].set_xlabel('Simulation', fontsize=11)
    axes[0, 1].set_ylabel('Mean Weight', fontsize=11)
    axes[0, 1].set_title('Average Connection Weight', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectral Radius
    if spectral_radii:
        axes[0, 2].bar(range(len(spectral_radii)), spectral_radii, color='darkred', alpha=0.7)
        axes[0, 2].set_xlabel('Simulation', fontsize=11)
        axes[0, 2].set_ylabel('Spectral Radius', fontsize=11)
        axes[0, 2].set_title('Weight Matrix Spectral Radius', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)
    
    # Mean Degree
    axes[1, 0].bar(sim_indices, mean_degrees, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Simulation', fontsize=11)
    axes[1, 0].set_ylabel('Mean Degree', fontsize=11)
    axes[1, 0].set_title('Average Connectivity', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max Weight
    axes[1, 1].bar(sim_indices, max_weights, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Simulation', fontsize=11)
    axes[1, 1].set_ylabel('Max |Weight|', fontsize=11)
    axes[1, 1].set_title('Maximum Weight Magnitude', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Number of Connections
    axes[1, 2].bar(sim_indices, n_connections, color='teal', alpha=0.7)
    axes[1, 2].set_xlabel('Simulation', fontsize=11)
    axes[1, 2].set_ylabel('# Connections', fontsize=11)
    axes[1, 2].set_title('Total Connections', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Network Topology Comparison Across Simulations', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(base_dir, 'network_comparison.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Network comparison plot saved to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary across {len(all_stats)} simulations:")
    print(f"Mean sparsity: {np.mean(sparsities):.2%} ± {np.std(sparsities):.2%}")
    print(f"Mean weight: {np.mean(mean_weights):.4f} ± {np.std(mean_weights):.4f}")
    print(f"Mean degree: {np.mean(mean_degrees):.1f} ± {np.std(mean_degrees):.1f}")
    if spectral_radii:
        print(f"Mean spectral radius: {np.mean(spectral_radii):.3f} ± {np.std(spectral_radii):.3f}")


def create_combined_summary_plot(avalanche_results, eigenvalue_results, 
                                network_stats, base_dir):
    """Create a combined summary plot showing all three analyses together."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Avalanche distributions
    if avalanche_results:
        # Size distribution
        ax2 = fig.add_subplot(gs[0, 1])
        burst = avalanche_results['burst']
        pdf = np.histogram(burst, bins=np.arange(1, np.max(burst) + 2))[0]
        p = pdf / np.sum(pdf)
        ax2.loglog(np.arange(1, np.max(burst) + 1), p, 'o', 
                  markersize=4, color='darkorchid', alpha=0.75)
        ax2.set_xlabel('Avalanche Size', fontsize=10)
        ax2.set_ylabel('PDF(S)', fontsize=10)
        ax2.set_title(f"α = {avalanche_results['alpha']:.3f}", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Duration distribution
        ax3 = fig.add_subplot(gs[0, 2])
        T = avalanche_results['T']
        tdf = np.histogram(T, bins=np.arange(1, np.max(T) + 2))[0]
        t = tdf / np.sum(tdf)
        ax3.loglog(np.arange(1, np.max(T) + 1), t, 'o',
                  markersize=4, color='mediumseagreen', alpha=0.75)
        ax3.set_xlabel('Avalanche Duration', fontsize=10)
        ax3.set_ylabel('PDF(D)', fontsize=10)
        ax3.set_title(f"β = {avalanche_results['beta']:.3f}", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Eigenvalue spectrum
    if eigenvalue_results and 'all_eigenvalues' in eigenvalue_results:
        ax4 = fig.add_subplot(gs[1, 0])
        eigenvalues = eigenvalue_results['all_eigenvalues']
        ax4.scatter(eigenvalues.real, eigenvalues.imag, 
                   c='darkblue', s=5, alpha=0.4, edgecolors='none')
        theta = np.linspace(0, 2*np.pi, 100)
        ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.4, linewidth=1)
        ax4.set_xlabel('Real Part', fontsize=10)
        ax4.set_ylabel('Imaginary Part', fontsize=10)
        ax4.set_title('Eigenvalue Spectrum', fontsize=12, fontweight='bold')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
    
    # Network statistics summary
    if network_stats and len(network_stats) > 0:
        ax5 = fig.add_subplot(gs[1, 1])
        sparsities = [s.get('sparsity', 0) for s in network_stats]
        mean_weights = [s.get('mean_weight', 0) for s in network_stats]
        
        x = range(len(network_stats))
        ax5_twin = ax5.twinx()
        
        bars1 = ax5.bar(x, sparsities, color='darkblue', alpha=0.5, 
                       label='Sparsity', width=0.4, align='edge')
        bars2 = ax5_twin.bar([i+0.4 for i in x], mean_weights, 
                            color='darkgreen', alpha=0.5, 
                            label='Mean Weight', width=0.4, align='edge')
        
        ax5.set_xlabel('Simulation', fontsize=10)
        ax5.set_ylabel('Sparsity', color='darkblue', fontsize=10)
        ax5_twin.set_ylabel('Mean Weight', color='darkgreen', fontsize=10)
        ax5.set_title('Network Properties', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='y', labelcolor='darkblue')
        ax5_twin.tick_params(axis='y', labelcolor='darkgreen')
        ax5.grid(True, alpha=0.3)
    
    # Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = "ANALYSIS SUMMARY\n" + "="*30 + "\n\n"
    
    if avalanche_results:
        summary_text += "Avalanche Analysis:\n"
        summary_text += f"  α = {avalanche_results['alpha']:.3f}\n"
        summary_text += f"  β = {avalanche_results['beta']:.3f}\n"
        summary_text += f"  Δ = {avalanche_results['df']:.3f}\n\n"
    
    if eigenvalue_results:
        summary_text += "Eigenvalue Analysis:\n"
        summary_text += f"  Mean max λ = {eigenvalue_results['mean_max_eigenvalue']:.3f}\n"
        summary_text += f"  Std max λ = {eigenvalue_results['std_max_eigenvalue']:.3f}\n\n"
    
    if network_stats:
        mean_sparsity = np.mean([s.get('sparsity', 0) for s in network_stats])
        mean_weight = np.mean([s.get('mean_weight', 0) for s in network_stats])
        summary_text += "Network Properties:\n"
        summary_text += f"  Mean sparsity = {mean_sparsity:.2%}\n"
        summary_text += f"  Mean weight = {mean_weight:.4f}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Combined Network, Avalanche, and Eigenvalue Analysis', 
                fontsize=15, fontweight='bold')
    
    save_path = os.path.join(base_dir, 'combined_summary.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Combined summary plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()