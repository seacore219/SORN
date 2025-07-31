#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot raster plots from saved h5 files showing both excitatory and inhibitory spikes
"""

import numpy as np
import matplotlib.pyplot as plt
import tables
import os
import sys

def plot_raster_from_h5(h5_file_path, time_window=None, save_plot=True):
    """
    Create raster plot from h5 file
    
    Parameters:
        h5_file_path: Path to result.h5 file
        time_window: (start, end) tuple for time window in ms, None for full data
        save_plot: Whether to save the plot as PDF/PNG
    """
    
    print(f"Loading data from: {h5_file_path}")
    
    try:
        h5 = tables.open_file(h5_file_path, 'r')
        
        # Check what's available
        print("\nAvailable datasets:")
        for node in h5.walk_nodes("/", "Array"):
            print(f"  {node._v_pathname}: shape {node.shape}")
        
        # Get spike data
        has_e_spikes = hasattr(h5.root, 'Spikes')
        has_i_spikes = hasattr(h5.root, 'SpikesInh')
        
        if not has_e_spikes and not has_i_spikes:
            print("No spike data found in h5 file!")
            h5.close()
            return
        
        # Load excitatory spikes
        if has_e_spikes:
            e_spikes = h5.root.Spikes[:]
            print(f"\nExcitatory spikes shape: {e_spikes.shape}")
            N_e = e_spikes.shape[0]
            N_steps = e_spikes.shape[1]
        
        # Load inhibitory spikes
        if has_i_spikes:
            i_spikes = h5.root.SpikesInh[:]
            print(f"Inhibitory spikes shape: {i_spikes.shape}")
            N_i = i_spikes.shape[0]
            if has_e_spikes:
                assert i_spikes.shape[1] == N_steps, "E and I spike arrays have different lengths!"
            else:
                N_steps = i_spikes.shape[1]
        
        # Get parameters if available
        if hasattr(h5.root, 'c'):
            c_data = h5.root.c[:]
            # Parameters are stored as a serialized bunch/dict
        
        h5.close()
        
        # Apply time window if specified
        if time_window is not None:
            start_idx = int(time_window[0])
            end_idx = int(time_window[1])
            if has_e_spikes:
                e_spikes = e_spikes[:, start_idx:end_idx]
            if has_i_spikes:
                i_spikes = i_spikes[:, start_idx:end_idx]
            time_steps = end_idx - start_idx
        else:
            time_steps = N_steps
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot excitatory spikes
        if has_e_spikes:
            ax_e = axes[0]
            # Find spike times for each neuron
            for neuron_idx in range(N_e):
                spike_times = np.where(e_spikes[neuron_idx, :] > 0)[0]
                if len(spike_times) > 0:
                    ax_e.scatter(spike_times, np.ones_like(spike_times) * neuron_idx,
                               s=0.5, c='black', marker='|')
            
            ax_e.set_ylabel('Excitatory Neuron', fontsize=12)
            ax_e.set_ylim(-0.5, N_e - 0.5)
            ax_e.set_title(f'Excitatory Neurons (N={N_e})', fontsize=14)
            ax_e.grid(True, alpha=0.3)
        
        # Plot inhibitory spikes
        if has_i_spikes:
            ax_i = axes[1]
            # Find spike times for each neuron
            for neuron_idx in range(N_i):
                spike_times = np.where(i_spikes[neuron_idx, :] > 0)[0]
                if len(spike_times) > 0:
                    ax_i.scatter(spike_times, np.ones_like(spike_times) * neuron_idx,
                               s=0.5, c='red', marker='|')
            
            ax_i.set_ylabel('Inhibitory Neuron', fontsize=12)
            ax_i.set_ylim(-0.5, N_i - 0.5)
            ax_i.set_title(f'Inhibitory Neurons (N={N_i})', fontsize=14)
            ax_i.set_xlabel('Time Step', fontsize=12)
            ax_i.grid(True, alpha=0.3)
        
        plt.suptitle(f'Raster Plot - {os.path.basename(os.path.dirname(os.path.dirname(h5_file_path)))}',
                    fontsize=16)
        plt.tight_layout()
        
        # Save if requested
        if save_plot:
            output_dir = os.path.dirname(h5_file_path)
            output_base = os.path.join(output_dir, 'raster_plot')
            plt.savefig(output_base + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(output_base + '.png', dpi=150, bbox_inches='tight')
            print(f"\nPlots saved to:\n  {output_base}.pdf\n  {output_base}.png")
        
        # Also create a plot showing firing rates over time
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate firing rates in bins
        bin_size = 1000  # steps
        n_bins = time_steps // bin_size
        
        if has_e_spikes and n_bins > 0:
            e_rates = []
            for i in range(n_bins):
                start = i * bin_size
                end = (i + 1) * bin_size
                rate = np.mean(e_spikes[:, start:end]) * 1000  # Hz (assuming 1ms timestep)
                e_rates.append(rate)
            
            time_bins = np.arange(n_bins) * bin_size
            ax.plot(time_bins, e_rates, 'b-', label='Excitatory', linewidth=2)
        
        if has_i_spikes and n_bins > 0:
            i_rates = []
            for i in range(n_bins):
                start = i * bin_size
                end = (i + 1) * bin_size
                rate = np.mean(i_spikes[:, start:end]) * 1000  # Hz
                i_rates.append(rate)
            
            ax.plot(time_bins, i_rates, 'r-', label='Inhibitory', linewidth=2)
        
        # Mark phase transitions for ExtraInput experiment
        if time_steps >= 6e6:  # Full simulation
            ax.axvline(2e6, color='gray', linestyle='--', alpha=0.5, label='Phase boundary')
            ax.axvline(4e6, color='gray', linestyle='--', alpha=0.5)
            ax.text(1e6, ax.get_ylim()[1]*0.9, 'Transient', ha='center')
            ax.text(3e6, ax.get_ylim()[1]*0.9, 'No Input', ha='center')
            ax.text(5e6, ax.get_ylim()[1]*0.9, 'With Input', ha='center')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title('Population Firing Rates Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_base2 = os.path.join(output_dir, 'firing_rates')
            plt.savefig(output_base2 + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(output_base2 + '.png', dpi=150, bbox_inches='tight')
            print(f"  {output_base2}.pdf\n  {output_base2}.png")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")
        import traceback
        traceback.print_exc()

def plot_all_rasters_in_batch(batch_path, time_window=None):
    """
    Plot raster plots for all simulations in a batch folder
    """
    import glob
    
    # Find all result.h5 files
    h5_files = glob.glob(os.path.join(batch_path, "*/common/result.h5"))
    h5_files.extend(glob.glob(os.path.join(batch_path, "sim_*/common/result.h5")))
    
    print(f"Found {len(h5_files)} h5 files in batch folder")
    
    for i, h5_file in enumerate(h5_files):
        print(f"\n{'='*60}")
        print(f"Processing file {i+1}/{len(h5_files)}")
        print(f"{'='*60}")
        plot_raster_from_h5(h5_file, time_window=time_window)

if __name__ == "__main__":
    # Example usage:
    
    # Plot a single file
    # h5_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\batch_ExtraInput_hip0.100_gain1.0_noise0.050\sim_01_2025-01-15 10-00-00\common\result.h5"
    # plot_raster_from_h5(h5_path)
    
    # Plot all files in a batch
    batch_path = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\batch_ExtraInput_hip0.100_gain1.0_noise0.050"
    
    # Plot full simulation
    plot_all_rasters_in_batch(batch_path)
    
    # Or plot just a time window (e.g., first 10000 steps)
    # plot_all_rasters_in_batch(batch_path, time_window=(0, 10000))
    
    # Or plot the transition to external input (around 4M steps)
    # plot_all_rasters_in_batch(batch_path, time_window=(3990000, 4010000))