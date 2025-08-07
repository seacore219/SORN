#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recreate Figure 6 exactly as in the paper
Modified to work with SORN raster data from specific directory structure
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import tables
import os
import sys
try:
    import powerlaw as pl
except ImportError:
    print("Warning: powerlaw library not found. Install with: pip install powerlaw")
    pl = None

# Parameters from original Fig6.py
section_steps = int(2e6)
extrainput_steps = 20  # Analysis window for onset

def detect_avalanches_with_transient(activity, threshold, transient_steps=0):
    """
    Detect avalanches but only count those that START after transient_steps
    This matches the original paper's approach for input onset
    """
    avalanches = []
    in_avalanche = False
    start_time = 0
    current_duration = 0
    current_size = 0
    
    for t, act in enumerate(activity):
        if act > threshold:
            if not in_avalanche:
                in_avalanche = True
                start_time = t
                current_duration = 0
                current_size = 0
            current_duration += 1
            current_size += act
        else:
            if in_avalanche:
                # Only include avalanche if it started after transient
                if start_time >= transient_steps:
                    avalanches.append((current_duration, current_size))
                in_avalanche = False
    
    if in_avalanche and start_time >= transient_steps:
        avalanches.append((current_duration, current_size))
    
    durations = np.array([a[0] for a in avalanches if a[0] > 0])
    sizes = np.array([a[1] for a in avalanches if a[1] > 0])
    
    return durations, sizes

def detect_avalanches_threshold_percent(activity, theta_percent):
    """
    Detect avalanches using a percentage of mean activity as threshold
    Used for confidence intervals
    """
    threshold = int(np.mean(activity) * theta_percent / 100.0) + 1
    return detect_avalanches_with_transient(activity, threshold)

def analyze_simulation(h5_file):
    """Analyze a single simulation file"""
    h5 = tables.open_file(h5_file, 'r')
    
    # Get activity
    activity = h5.root.activity[:]
    if activity.ndim > 1:
        activity = activity.flatten()
    
    # Get N_e
    N_e = 200
    if hasattr(h5.root, 'c') and hasattr(h5.root.c, 'N_e'):
        N_e = h5.root.c.N_e
    
    # Convert to neuron counts
    if np.max(activity) <= 1.0:
        activity = np.round(activity * N_e)
    
    h5.close()
    
    results = {}
    
    # 1. Normal regime (section 2: steps 2M to 4M)
    normal_activity = activity[section_steps:2*section_steps]
    threshold_normal = int(np.mean(normal_activity) / 2.0) + 1  # Rounding as in original
    T_normal, S_normal = detect_avalanches_with_transient(normal_activity, threshold_normal)
    
    # Also get avalanches at 5% and 25% thresholds for confidence intervals
    T_normal_5, S_normal_5 = detect_avalanches_threshold_percent(normal_activity, 5)
    T_normal_25, S_normal_25 = detect_avalanches_threshold_percent(normal_activity, 25)
    
    results['normal'] = {
        'T': T_normal, 'S': S_normal, 
        'T_5': T_normal_5, 'S_5': S_normal_5,
        'T_25': T_normal_25, 'S_25': S_normal_25,
        'threshold': threshold_normal
    }
    
    # 2. Input onset (first 20 steps of section 3, but analyze 200 steps)
    # Activity from 4M to 4M+200, but only count avalanches starting in first 20
    onset_activity = activity[2*section_steps:2*section_steps + 10*extrainput_steps]
    T_onset, S_onset = detect_avalanches_with_transient(onset_activity, 
                                                        threshold_normal, 
                                                        transient_steps=extrainput_steps)
    
    # For onset, use fixed thresholds as in original
    T_onset_9, S_onset_9 = detect_avalanches_with_transient(onset_activity, 9, 
                                                           transient_steps=extrainput_steps)
    T_onset_12, S_onset_12 = detect_avalanches_with_transient(onset_activity, 12, 
                                                             transient_steps=extrainput_steps)
    
    results['onset'] = {
        'T': T_onset, 'S': S_onset,
        'T_9': T_onset_9, 'S_9': S_onset_9,
        'T_12': T_onset_12, 'S_12': S_onset_12
    }
    
    # 3. Readaptation (rest of section 3: from 4M+20 to 6M)
    readapt_activity = activity[2*section_steps + extrainput_steps:]
    T_readapt, S_readapt = detect_avalanches_with_transient(readapt_activity, threshold_normal)
    
    T_readapt_5, S_readapt_5 = detect_avalanches_threshold_percent(readapt_activity, 5)
    T_readapt_25, S_readapt_25 = detect_avalanches_threshold_percent(readapt_activity, 25)
    
    results['readapt'] = {
        'T': T_readapt, 'S': S_readapt,
        'T_5': T_readapt_5, 'S_5': S_readapt_5,
        'T_25': T_readapt_25, 'S_25': S_readapt_25
    }
    
    return results

def plot_with_powerlaw(data, ax, color='k', linewidth=1.5, label=None):
    """Plot using powerlaw library if available, otherwise use histogram"""
    data = data[data > 0]
    if len(data) < 10:
        return
    
    if pl is not None:
        # Use powerlaw library for smooth fitting
        pl.plot_pdf(data, color=color, linewidth=linewidth, label=label, ax=ax)
    else:
        # Fallback to histogram
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 30)
        hist, edges = np.histogram(data, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        mask = hist > 0
        ax.plot(centers[mask], hist[mask], color=color, linewidth=linewidth, label=label)

def plot_confidence_interval(data1, data2, ax, color, alpha=0.2):
    """Plot confidence interval between two datasets"""
    if pl is not None:
        # Use powerlaw library's pdf function
        pdf1 = pl.pdf(data1, 10)
        pdf2 = pl.pdf(data2, 10)
        bin_centers1 = (pdf1[0][:-1] + pdf1[0][1:]) / 2.
        bin_centers2 = (pdf2[0][:-1] + pdf2[0][1:]) / 2.
        x_max = pdf1[0].max()
        interp1 = np.interp(np.arange(x_max), bin_centers1, pdf1[1])
        interp2 = np.interp(np.arange(x_max), bin_centers2, pdf2[1])
        ax.fill_between(np.linspace(0, x_max, len(interp2)), interp2, interp1, 
                       facecolor=color, alpha=alpha)

def create_figure6(batch_path):
    """Create Figure 6 from batch of simulations"""
    
    # Find all folders starting with "202" (for years 2020-2029)
    folders = []
    if os.path.exists(batch_path):
        for item in os.listdir(batch_path):
            if item.startswith("202") and os.path.isdir(os.path.join(batch_path, item)):
                folders.append(item)
    
    # Sort folders to process them in chronological order
    folders.sort()
    
    print(f"Found {len(folders)} simulation folders starting with '202'")
    
    # Build paths to result.h5 files
    h5_files = []
    for folder in folders:
        h5_path = os.path.join(batch_path, folder, "common", "result.h5")
        if os.path.exists(h5_path):
            h5_files.append(h5_path)
        else:
            print(f"Warning: Could not find {h5_path}")
    
    print(f"Found {len(h5_files)} simulations")
    
    # Collect results from all simulations
    all_results = {
        'normal': {'T': [], 'S': [], 'T_5': [], 'S_5': [], 'T_25': [], 'S_25': []},
        'onset': {'T': [], 'S': [], 'T_9': [], 'S_9': [], 'T_12': [], 'S_12': []},
        'readapt': {'T': [], 'S': [], 'T_5': [], 'S_5': [], 'T_25': [], 'S_25': []}
    }
    
    for i, h5_file in enumerate(h5_files):
        print(f"Processing {i+1}/{len(h5_files)}: {os.path.basename(os.path.dirname(os.path.dirname(h5_file)))}")
        try:
            results = analyze_simulation(h5_file)
            for regime in ['normal', 'onset', 'readapt']:
                for key in results[regime]:
                    if key != 'threshold':
                        all_results[regime][key].extend(results[regime][key])
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Convert to arrays
    for regime in all_results:
        for key in all_results[regime]:
            all_results[regime][key] = np.array(all_results[regime][key])
    
    # Create figure
    fig = plt.figure(figsize=(7, 3))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    # Define line widths
    line_width = 1.5
    line_width_fit = 2.0
    
    # Plot distributions
    # Panel A: Duration
    plot_confidence_interval(all_results['normal']['T_5'], all_results['normal']['T_25'], 
                           ax1, 'k', alpha=0.2)
    plot_with_powerlaw(all_results['normal']['T'], ax1, 'k', line_width_fit, None)
    
    plot_confidence_interval(all_results['onset']['T_9'], all_results['onset']['T_12'], 
                           ax1, 'r', alpha=0.2)
    plot_with_powerlaw(all_results['onset']['T'], ax1, 'r', line_width, None)
    
    plot_confidence_interval(all_results['readapt']['T_5'], all_results['readapt']['T_25'], 
                           ax1, 'cyan', alpha=0.2)
    plot_with_powerlaw(all_results['readapt']['T'], ax1, 'cyan', line_width_fit, None)
    
    # Panel B: Size
    plot_confidence_interval(all_results['normal']['S_5'], all_results['normal']['S_25'], 
                           ax2, 'k', alpha=0.2)
    plot_with_powerlaw(all_results['normal']['S'], ax2, 'k', line_width_fit, 'Before input')
    
    plot_confidence_interval(all_results['onset']['S_9'], all_results['onset']['S_12'], 
                           ax2, 'r', alpha=0.2)
    plot_with_powerlaw(all_results['onset']['S'], ax2, 'r', line_width, 'Input onset')
    
    plot_confidence_interval(all_results['readapt']['S_5'], all_results['readapt']['S_25'], 
                           ax2, 'cyan', alpha=0.2)
    plot_with_powerlaw(all_results['readapt']['S'], ax2, 'cyan', line_width_fit, 'Readaptation')
    
    # Format axes exactly as in paper
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    
    # Panel A formatting
    ax1.set_xlabel(r'$T$', fontsize=10)
    ax1.set_ylabel(r'$f(T)$', fontsize=10)
    ax1.set_xlim([1, 300])
    ax1.set_ylim([0.0001, 1])
    ax1.set_xticks([1, 10, 100])
    ax1.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$'])
    ax1.set_yticks([1, 0.01, 0.0001])
    ax1.set_yticklabels(['$10^0$', '$10^{-2}$', '$10^{-4}$'])
    
    # Panel B formatting
    ax2.set_xlabel(r'$S$', fontsize=10)
    ax2.set_ylabel(r'$f(S)$', fontsize=10)
    ax2.set_xlim([1, 3000])
    ax2.set_ylim([0.00001, 0.1])
    ax2.set_xticks([1, 10, 100, 1000])
    ax2.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
    ax2.set_yticks([0.1, 0.001, 0.00001])
    ax2.set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
    
    # Legend
    ax2.legend(loc=(0.5, 0.8), prop={'size': 10}, frameon=False)
    
    # Panel labels
    ax1.annotate('A', xy=(-0.15, 0.9), xycoords='axes fraction',
                fontsize=12, fontweight='bold')
    ax2.annotate('B', xy=(-0.15, 0.9), xycoords='axes fraction',
                fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.17, wspace=0.4)
    
    # Save
    output_path = os.path.join(batch_path, 'Fig6_exact.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Set the path directly here
    batch_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\delpapa_input\batch_hip0.06_n4_ps1"
    
    create_figure6(batch_path)