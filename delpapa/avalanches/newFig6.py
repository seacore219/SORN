#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recreate Figure 6 from SORN paper using saved h5 files
This analyzes avalanche distributions in three regimes:
- Normal (before input)
- Extra input start (onset of input)
- Extra input end (readaptation)
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tables
import os
import sys
import glob
import re

# Parameters
section_steps = int(2e6)
extrainput_steps = 20  # Duration of extra input phase

# Figure parameters
width = 7
height = 3
letter_size = 10
letter_size_panel = 12
line_width = 1.5
line_width_fit = 2.0

def get_h5_files(batch_path):
    """Find all result.h5 files in the simulation directories"""
    h5_files = []
    
    # Look for directories with date pattern or sim_XX pattern
    for folder in os.listdir(batch_path):
        folder_path = os.path.join(batch_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Check if it matches date pattern (YYYY-MM-DD HH-MM-SS) or sim pattern
        date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}'
        sim_pattern = r'sim_\d+_\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}'
        
        if re.match(date_pattern, folder) or re.match(sim_pattern, folder):
            h5_path = os.path.join(folder_path, 'common', 'result.h5')
            if os.path.exists(h5_path):
                h5_files.append(h5_path)
                print(f"Found: {folder}/common/result.h5")
    
    print(f"\nTotal h5 files found: {len(h5_files)}")
    return sorted(h5_files)

def simple_avalanche_analysis(activity, threshold):
    """
    Simple avalanche detection
    An avalanche starts when activity exceeds threshold and ends when it goes below
    """
    avalanches = []
    in_avalanche = False
    current_duration = 0
    current_size = 0
    
    for act in activity:
        if act > threshold:
            in_avalanche = True
            current_duration += 1
            current_size += act
        else:
            if in_avalanche:
                avalanches.append((current_duration, current_size))
                in_avalanche = False
                current_duration = 0
                current_size = 0
    
    if in_avalanche:  # Handle case where data ends during avalanche
        avalanches.append((current_duration, current_size))
    
    durations = np.array([a[0] for a in avalanches if a[0] > 0])
    sizes = np.array([a[1] for a in avalanches if a[1] > 0])
    
    return durations, sizes

def plot_power_law_pdf(data, ax, color='k', linewidth=1.5, label=None, bins=50):
    """Plot probability density function on log-log scale"""
    # Remove zeros and invalid values
    data = data[data > 0]
    if len(data) == 0:
        return
    
    # Create log-spaced bins
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot only non-zero values
    mask = hist > 0
    ax.plot(bin_centers[mask], hist[mask], color=color, linewidth=linewidth, label=label)

def main():
    # Specify the batch folder you want to analyze
    batch_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\batch_ExtraInput_20250731_234502"
    
    print(f"Analyzing batch: {os.path.basename(batch_path)}")
    print(f"Full path: {batch_path}")
    
    if not os.path.exists(batch_path):
        print(f"Error: Batch path does not exist: {batch_path}")
        return
    
    h5_files = get_h5_files(batch_path)
    
    if not h5_files:
        print("No h5 files found!")
        return
    
    # Initialize figure
    fig = plt.figure(1, figsize=(width, height))
    
    possible_regimes = ['normal', 'extrainput_start', 'extrainput_end']
    
    # Storage for all trials
    results = {regime: {'T_all': [], 'S_all': [], 'activity': []} for regime in possible_regimes}
    
    # Process each h5 file
    for file_idx, h5_file in enumerate(h5_files):
        sim_folder = os.path.basename(os.path.dirname(os.path.dirname(h5_file)))
        print(f"\nProcessing file {file_idx + 1}/{len(h5_files)}: {sim_folder}")
        
        try:
            h5 = tables.open_file(h5_file, 'r')
            
            # Debug: List all available datasets
            if file_idx == 0:  # Only print for first file
                print("\n  Available datasets in h5 file:")
                for node in h5.walk_nodes("/", "Array"):
                    print(f"    {node._v_pathname}: shape {node.shape}")
                print()
            
            # Get activity data
            if hasattr(h5.root, 'activity'):
                activity = h5.root.activity[:]
                print(f"  Activity shape: {activity.shape}")
                
                # Handle different array shapes
                if activity.ndim > 1:
                    # If activity has shape (1, N_steps), flatten it
                    activity = activity.flatten()
                    print(f"  Flattened activity shape: {activity.shape}")
                
                # Get N_e from parameters if available
                N_e = 200  # Default value
                    
                # Convert activity to actual neuron counts if needed
                # The activity is usually stored as fraction of active neurons
                if np.max(activity) <= 1.0:  # Likely a fraction
                    activity = activity * N_e
                
                print(f"  Activity range: [{np.min(activity):.3f}, {np.max(activity):.3f}]")
                print(f"  Activity mean: {np.mean(activity):.3f}")
                
                # Process each regime
                # Normal: section 1 (before input)
                if len(activity) >= 2 * section_steps:
                    normal_activity = activity[section_steps:2*section_steps]
                    results['normal']['activity'].append(normal_activity)
                    print(f"  Normal phase: {len(normal_activity)} steps")
                
                # Extra input start: right after 2*section_steps
                if len(activity) >= 2*section_steps + 10*extrainput_steps:
                    start_activity = activity[2*section_steps:2*section_steps + extrainput_steps]
                    results['extrainput_start']['activity'].append(start_activity)
                    print(f"  Extra input start: {len(start_activity)} steps")
                
                # Extra input end: after the extra input phase
                if len(activity) >= 2*section_steps + extrainput_steps:
                    end_activity = activity[2*section_steps + extrainput_steps:]
                    # Limit to reasonable length for analysis
                    if len(end_activity) > section_steps - extrainput_steps:
                        end_activity = end_activity[:section_steps - extrainput_steps]
                    results['extrainput_end']['activity'].append(end_activity)
                    print(f"  Extra input end: {len(end_activity)} steps")
            else:
                print("  Warning: No activity data found in h5 file")
            
            h5.close()
            
        except Exception as e:
            print(f"  Error processing {h5_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate avalanches for each regime
    colors = {'normal': 'k', 'extrainput_start': 'r', 'extrainput_end': 'cyan'}
    labels = {'normal': 'Before input', 'extrainput_start': 'Input onset', 'extrainput_end': 'Readaptation'}
    
    # Create subplots
    ax1 = plt.subplot(121)  # Duration distributions
    ax2 = plt.subplot(122)  # Size distributions
    
    for regime in possible_regimes:
        if not results[regime]['activity']:
            print(f"\nNo data for regime: {regime}")
            continue
        
        print(f"\nAnalyzing {regime}...")
        
        # Concatenate all activity data for this regime
        all_activity = np.concatenate(results[regime]['activity'])
        
        # Calculate threshold (half of mean activity)
        threshold = np.mean(all_activity) / 2.0
        print(f"  Mean activity: {np.mean(all_activity):.3f}, Std: {np.std(all_activity):.3f}")
        print(f"  Threshold: {threshold:.3f}")
        
        # Detect avalanches for each trial and combine
        T_all = []
        S_all = []
        
        for activity in results[regime]['activity']:
            T, S = simple_avalanche_analysis(activity, threshold)
            if len(T) > 0:
                T_all.extend(T)
                S_all.extend(S)
        
        T_all = np.array(T_all)
        S_all = np.array(S_all)
        
        print(f"  Found {len(T_all)} avalanches")
        if len(T_all) > 0:
            print(f"  Duration range: [{np.min(T_all)}, {np.max(T_all)}]")
            print(f"  Size range: [{np.min(S_all):.1f}, {np.max(S_all):.1f}]")
        
        # Plot distributions
        if len(T_all) > 10:  # Need enough avalanches for meaningful distribution
            plot_power_law_pdf(T_all, ax1, color=colors[regime], 
                             linewidth=line_width if regime == 'normal' else line_width_fit,
                             label=None)
            plot_power_law_pdf(S_all, ax2, color=colors[regime], 
                             linewidth=line_width if regime == 'normal' else line_width_fit,
                             label=labels[regime])
    
    # Format subplot 1 (Duration)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$T$', fontsize=letter_size)
    ax1.set_ylabel(r'$f(T)$', fontsize=letter_size)
    ax1.set_xlim([1, 300])
    ax1.set_ylim([0.0001, 1])
    ax1.set_xticks([1, 10, 100])
    ax1.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$'])
    ax1.set_yticks([1, 0.01, 0.0001])
    ax1.set_yticklabels(['$10^0$', '$10^{-2}$', '$10^{-4}$'])
    ax1.tick_params(labelsize=letter_size)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    # Format subplot 2 (Size)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$S$', fontsize=letter_size)
    ax2.set_ylabel(r'$f(S)$', fontsize=letter_size)
    ax2.set_xlim([1, 3000])
    ax2.set_ylim([0.00001, 0.1])
    ax2.set_xticks([1, 10, 100, 1000])
    ax2.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
    ax2.set_yticks([0.1, 0.001, 0.00001])
    ax2.set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
    
    # Only add legend if we have data
    handles, labels_list = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(loc=(0.5, 0.8), prop={'size': letter_size}, frameon=False)
    
    # Add panel labels
    ax1.annotate('A', xy=(-0.15, 0.9), xycoords='axes fraction',
                fontsize=letter_size_panel, fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom')
    ax2.annotate('B', xy=(-0.15, 0.9), xycoords='axes fraction',
                fontsize=letter_size_panel, fontweight='bold',
                horizontalalignment='right', verticalalignment='bottom')
    
    # Adjust layout
    plt.gcf().subplots_adjust(bottom=0.17, wspace=0.4)
    
    # Save figure
    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, 'Fig6_recreated.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    # Also save as PNG for easier viewing
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()