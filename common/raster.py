import tables
import numpy as np
import matplotlib.pyplot as plt

def extract_and_plot_rasters(h5_path, time_window=None):
    """
    Extract and plot both excitatory and inhibitory neuron rasters
    
    Args:
        h5_path: Path to result.h5 file
        time_window: Optional tuple (start, end) to plot specific time range
    """
    h5 = tables.open_file(h5_path, 'r')
    
    # Get network parameters
    N_e = h5.root.c.N_e[0]  # Number of excitatory neurons
    N_i = h5.root.c.N_i[0]  # Number of inhibitory neurons
    
    print(f"Network: {N_e} excitatory, {N_i} inhibitory neurons")
    
    # Check what spike data exists
    excitatory_raster = None
    inhibitory_raster = None
    
    # Check for main Spikes data
    if hasattr(h5.root, 'Spikes'):
        spike_data = h5.root.Spikes[0, :, :]
        print(f"Found Spikes data: shape {spike_data.shape}")
        
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
        inhibitory_raster = h5.root.SpikesInh[0, :, :]
        print(f"Found SpikesInh data: shape {inhibitory_raster.shape}")
    
    h5.close()
    
    # Get initial dimensions to check if we need to reduce the time window
    if excitatory_raster is not None:
        original_time_steps = excitatory_raster.shape[1]
    elif inhibitory_raster is not None:
        original_time_steps = inhibitory_raster.shape[1]
    else:
        original_time_steps = 1000  # fallback
    
    max_neurons = max(N_e, N_i)
    
    # Calculate maximum time steps that would create a reasonable figure
    max_width = 50  # Maximum figure width in inches
    base_height = 4
    max_time_steps_for_display = int(max_width * max_neurons / base_height)
    
    # Auto-reduce time window if dataset is too large and no specific window was given
    if time_window is None and original_time_steps > max_time_steps_for_display:
        # Use the first portion of the data that fits reasonably
        auto_end = min(max_time_steps_for_display, original_time_steps)
        time_window = (0, auto_end)
        print(f"Dataset too large ({original_time_steps} time steps). Auto-reducing to first {auto_end} time steps for display.")
        print(f"To view other portions, specify time_window parameter (e.g., time_window=({auto_end}, {min(auto_end*2, original_time_steps)}))")
    
    # Apply time window (either user-specified or auto-generated)
    if time_window is not None:
        start, end = time_window
        if excitatory_raster is not None:
            excitatory_raster = excitatory_raster[:, start:end]
        if inhibitory_raster is not None:
            inhibitory_raster = inhibitory_raster[:, start:end]
    
    # Calculate dimensions for proper scaling
    if excitatory_raster is not None:
        time_steps = excitatory_raster.shape[1]
    elif inhibitory_raster is not None:
        time_steps = inhibitory_raster.shape[1]
    else:
        time_steps = 1000  # fallback
    
    # Calculate figure size to maintain 1:1 aspect ratio
    aspect_ratio = time_steps / max_neurons
    
    # Set a reasonable base size and scale accordingly
    fig_width = base_height * aspect_ratio
    fig_height = base_height * 2  # Two subplots
    
    print(f"Displaying: {time_steps} time steps, {max_neurons} max neurons")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    print(f"Figure size: {fig_width:.1f} x {fig_height:.1f} inches")
    
    # Create figure with calculated dimensions
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    
    # Plot excitatory neurons
    if excitatory_raster is not None:
        spike_times_e, neuron_ids_e = np.where(excitatory_raster)
        axes[0].scatter(spike_times_e, neuron_ids_e, s=0.5, c='red', marker='.', alpha=0.5)
        axes[0].set_ylabel('Excitatory Neuron ID')
        axes[0].set_title(f'Excitatory Neurons (N={N_e})')
        axes[0].set_ylim(-1, N_e)
        # Set equal aspect ratio (now that we've controlled the data size)
        axes[0].set_aspect('equal', adjustable='box')
    else:
        axes[0].text(0.5, 0.5, 'No excitatory spike data found', 
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Excitatory Neurons - NO DATA')
    
    # Plot inhibitory neurons
    if inhibitory_raster is not None:
        spike_times_i, neuron_ids_i = np.where(inhibitory_raster)
        axes[1].scatter(spike_times_i, neuron_ids_i, s=0.5, c='blue', marker='.', alpha=0.5)
        axes[1].set_ylabel('Inhibitory Neuron ID')
        axes[1].set_title(f'Inhibitory Neurons (N={N_i})')
        axes[1].set_ylim(-1, N_i)
        # Set equal aspect ratio (now that we've controlled the data size)
        axes[1].set_aspect('equal', adjustable='box')
    else:
        axes[1].text(0.5, 0.5, 'No inhibitory spike data found', 
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Inhibitory Neurons - NO DATA')
    
    # Add time window info to x-axis label if data was windowed
    xlabel = 'Time (ms)'
    if time_window is not None:
        xlabel += f' (showing {time_window[0]} to {time_window[1]})'
    axes[1].set_xlabel(xlabel)
    
    plt.tight_layout()
    plt.show()
    
    return excitatory_raster, inhibitory_raster

# Use it like this:
h5_file = r"C:\Users\seaco\OneDrive\Documents\Charles\SORN_PC\backup\batch_hip0.0700_fp0.00_cde0.00_cdi0.00\sim_03_2025-06-24 16-19-57\common\result.h5"
# Plot full raster
exc_raster, inh_raster = extract_and_plot_rasters(h5_file)

# Or plot a specific time window (e.g., timesteps 100000 to 110000)
# exc_raster, inh_raster = extract_and_plot_rasters(h5_file, time_window=(100000, 110000))

# Save the rasters if needed
if exc_raster is not None:
    np.save('excitatory_raster.npy', exc_raster)
if inh_raster is not None:
    np.save('inhibitory_raster.npy', inh_raster)