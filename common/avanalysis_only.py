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

# Memory management
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
gc.collect()
tables.file._open_files.close_all()
print("[INFO] Libraries imported successfully.")

# ============================================
# CONFIGURATION
# ============================================
base_dir = 'C:\\Users\\seaco\\OneDrive\\Documents\\Charles\\SORN_PC\\backup\\delpapa_input\\batch_hip0.06_n4_ps1'
starting_time_point = 5000000
end_time_point = 6000000  # Set to None for full length

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
        'S': np.asarray(burst),
        'T': T,
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

# ============================================
# MAIN ANALYSIS
# ============================================

def main():
    print(f"\n{'='*60}")
    print("SIMPLIFIED AVALANCHE ANALYSIS")
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
    
    # Process each file
    for file_idx, file_path in enumerate(h5_files):
        print(f"\n{'='*50}")
        print(f"Processing file {file_idx+1}/{len(h5_files)}")
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
            
            if len(results['S']) > 0:
                print(f"Found {len(results['S'])} avalanches")
                
                # Accumulate results
                all_burst = np.concatenate((all_burst, results['S']))
                all_T = np.concatenate((all_T, results['T']))
                
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
                flag=1,  # Simple analysis without p-values
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
            # Use simplified analysis without criticality module
            AV_Result = AV_analysis(
                burst=all_burst,
                T=all_T,
                flag=1,  # Simple analysis without p-values
                **{k: v for k, v in AVALANCHE_PARAMS.items() 
                   if k not in ['perc_threshold', 'const_threshold']},
                plot=True,
                pltname='avalanche_analysis_simple',
                saveloc=base_dir
            )
            
    else:
        print("No avalanches found in any files.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()