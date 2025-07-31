import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional, List
import os
import h5py

class SORN:
    """
    Self-Organizing Recurrent Network (SORN) implementation
    Based on: Zheng & Triesch (2014) - A robust self-organizing neural network model
    
    Fixed to be more faithful to the original paper implementation
    """
    
    def __init__(self, 
                 N_E: int = 200,
                 N_I: int = 40,
                 N_U: int = 0,  # Set to 0 as per paper (no external input by default)
                 eta_stdp: float = 0.004,
                 eta_istdp: float = 0.001,
                 eta_ip: float = 0.01,
                 h_ip: float = 0.1,  # Target firing rate (called h_ip in paper)
                 lambda_: float = 20,  # Expected number of connections per neuron
                 T_e_min: float = 0.5,  # Minimum excitatory threshold
                 T_e_max: float = 1.0,  # Maximum excitatory threshold
                 T_i_min: float = 0.5,  # Minimum inhibitory threshold
                 T_i_max: float = 1.0,  # Maximum inhibitory threshold
                 noise_sig: float = np.sqrt(0.05),  # Noise standard deviation
                 W_ee_initial: float = 0.001,  # Initial weight for new connections
                 W_min: float = 0.0,  # Minimum weight value
                 W_max: float = 1.0,  # Maximum weight value
                 W_ei_min: float = 0.001):  # Minimum weight for W_EI (never fully remove inhibition)
        """
        Initialize SORN network with parameters closer to the original paper
        
        Parameters match those in Zheng & Triesch (2014)
        """
        
        # Network size
        self.N_E = N_E
        self.N_I = N_I
        self.N_U = N_U
        
        # Learning rates
        self.eta_stdp = eta_stdp
        self.eta_istdp = eta_istdp
        self.eta_ip = eta_ip
        self.h_ip = h_ip  # Target firing rate
        
        # Network parameters
        self.lambda_ = lambda_
        self.p_ee = lambda_ / N_E  # Connection probability for E->E
        self.p_ei = 0.2  # Connection probability for I->E (20% as in paper)
        self.p_ie = 1.0  # Connection probability for E->I (100% as in paper)
        
        # Structural plasticity probability
        # From the paper and Implementation 2: 
        # p_sp = N_E * (N_E - 1) * (0.1 / (200 * 199))
        # This ensures the expected number of new connections balances removals
        expected_connections = self.p_ee * N_E * (N_E - 1)  # Expected total connections
        total_possible = N_E * (N_E - 1)  # Total possible connections
        # Probability chosen so that on average 1 new connection is created per timestep
        self.p_sp = expected_connections / (200.0 * 199.0)  # Scaled for 200x200 network
        
        # Weight bounds
        self.W_ee_initial = W_ee_initial
        self.W_min = W_min
        self.W_max = W_max
        self.W_ei_min = W_ei_min
        
        # Threshold bounds
        self.T_e_min = T_e_min
        self.T_e_max = T_e_max
        self.T_i_min = T_i_min
        self.T_i_max = T_i_max
        
        # Noise parameter
        self.noise_sig = noise_sig
        
        # Initialize network components
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize all network components following the paper"""
        
        # Initialize state vectors
        # Excitatory neurons start with probability h_ip of being active
        self.x = np.random.rand(self.N_E) < self.h_ip
        self.y = np.zeros(self.N_I, dtype=bool)  # Inhibitory neurons start silent
        
        # Input (if any)
        self.u = np.zeros(self.N_U) if self.N_U > 0 else np.array([])
        
        # Pre-threshold activities
        self.R_x = np.zeros(self.N_E)
        self.R_y = np.zeros(self.N_I)
        
        # Initialize thresholds uniformly distributed
        self.T_E = np.random.uniform(self.T_e_min, self.T_e_max, self.N_E)
        self.T_I = np.random.uniform(self.T_i_min, self.T_i_max, self.N_I)
        
        # Initialize weight matrices
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize synaptic weight matrices with proper sparsity"""
        
        # W_EE: Excitatory to excitatory (sparse)
        self.W_EE = self._create_sparse_matrix(
            self.N_E, self.N_E, self.p_ee, avoid_self=True
        )
        
        # W_EI: Inhibitory to excitatory (20% connectivity)
        self.W_EI = self._create_sparse_matrix(
            self.N_E, self.N_I, self.p_ei, avoid_self=False
        )
        
        # W_IE: Excitatory to inhibitory (full connectivity)
        self.W_IE = self._create_sparse_matrix(
            self.N_I, self.N_E, self.p_ie, avoid_self=False
        )
        
        # W_EU: Input to excitatory (only if input exists)
        if self.N_U > 0:
            self.W_EU = np.ones((self.N_E, self.N_U))
        else:
            self.W_EU = np.zeros((self.N_E, 0))
        
        # Normalize all weight matrices
        self._normalize_weights()
        
    def _create_sparse_matrix(self, rows: int, cols: int, p: float, 
                            avoid_self: bool = False) -> np.ndarray:
        """
        Create sparse weight matrix with connection probability p
        Ensures each neuron has at least one input connection
        """
        
        # Create connection mask
        mask = np.random.rand(rows, cols) < p
        
        if avoid_self and rows == cols:
            np.fill_diagonal(mask, False)
        
        # Ensure each neuron has at least one input (for normalization)
        for i in range(rows):
            if not mask[i].any():
                # Add random connection
                j = np.random.randint(cols)
                if avoid_self and i == j:
                    j = (j + 1) % cols
                mask[i, j] = True
        
        # Initialize weights uniformly between 0 and 1 where connections exist
        W = np.zeros((rows, cols))
        W[mask] = np.random.rand(np.sum(mask))
        
        return W
        
    def _normalize_weights(self):
        """
        Normalize all incoming weights to sum to 1 for each neuron
        This maintains the balance of excitation and inhibition
        """
        
        # Normalize W_EE (sum of incoming weights = 1)
        row_sums = self.W_EE.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.W_EE = self.W_EE / row_sums[:, np.newaxis]
        
        # Normalize W_EI
        row_sums = self.W_EI.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.W_EI = self.W_EI / row_sums[:, np.newaxis]
        
        # Normalize W_IE
        row_sums = self.W_IE.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.W_IE = self.W_IE / row_sums[:, np.newaxis]
        
        # Normalize W_EU (if input exists)
        if self.N_U > 0:
            row_sums = self.W_EU.sum(axis=1)
            row_sums[row_sums == 0] = 1
            self.W_EU = self.W_EU / row_sums[:, np.newaxis]
    
    def step(self, u_new: Optional[np.ndarray] = None):
        """
        Perform one update step of the SORN network
        
        Parameters:
        -----------
        u_new : np.ndarray, optional
            New input vector (if None, uses zero input)
        """
        
        # Store previous states for plasticity
        x_prev = self.x.copy()
        y_prev = self.y.copy()
        
        # Update input
        if u_new is not None and self.N_U > 0:
            self.u = u_new
        else:
            self.u = np.zeros(self.N_U) if self.N_U > 0 else np.array([])
        
        # Compute pre-threshold excitatory activity
        # R_x = W_EE * x - W_EI * y - T_E + noise
        self.R_x = (self.W_EE @ self.x.astype(float) - 
                    self.W_EI @ self.y.astype(float) - 
                    self.T_E)
        
        # Add input if present
        if self.N_U > 0:
            self.R_x += self.W_EU @ self.u
        
        # Add noise
        self.R_x += self.noise_sig * np.random.randn(self.N_E)
        
        # Apply threshold function (Heaviside)
        x_new = (self.R_x >= 0).astype(float)
        
        # Compute pre-threshold inhibitory activity
        # R_y = W_IE * x - T_I + noise
        self.R_y = (self.W_IE @ self.x.astype(float) - 
                    self.T_I + 
                    self.noise_sig * np.random.randn(self.N_I))
        
        # Apply threshold function
        y_new = (self.R_y >= 0).astype(float)
        
        # Apply plasticity rules
        self._apply_plasticity(x_prev, x_new, y_prev, y_new)
        
        # Update states
        self.x = x_new.astype(bool)
        self.y = y_new.astype(bool)
        
    def _apply_plasticity(self, x_prev: np.ndarray, x_new: np.ndarray, 
                         y_prev: np.ndarray, y_new: np.ndarray):
        """Apply all plasticity rules in the correct order"""
        
        # 1. Intrinsic plasticity (always applied)
        self._ip(x_new)
        
        # 2. STDP
        self._stdp(x_prev, x_new)
        
        # 3. iSTDP
        self._istdp(y_prev, x_new)
        
        # 4. Structural plasticity
        self._structural_plasticity()
        
        # 5. Synaptic normalization (must be done after all weight updates)
        self._normalize_weights()
    
    def _ip(self, x_new: np.ndarray):
        """
        Intrinsic Plasticity: Adjust thresholds to maintain target firing rate
        ΔT_i^E = η_IP * (x_i - h_IP)
        """
        
        # Update thresholds
        self.T_E += self.eta_ip * (x_new - self.h_ip)
        
        # Keep thresholds in valid range
        self.T_E = np.clip(self.T_E, self.T_e_min, self.T_e_max)
        
    def _stdp(self, x_prev: np.ndarray, x_new: np.ndarray):
        """
        Spike-Timing Dependent Plasticity
        ΔW_ij^EE = η_STDP * [x_i(t) * x_j(t-1) - x_i(t-1) * x_j(t)]
        """
        
        # Compute weight changes
        # Potentiation: post fires now, pre fired before
        potentiation = np.outer(x_new, x_prev)
        # Depression: post fired before, pre fires now  
        depression = np.outer(x_prev, x_new)
        
        # Update weights
        dW = self.eta_stdp * (potentiation - depression)
        
        # Only update existing connections
        mask = self.W_EE > 0
        self.W_EE[mask] += dW[mask]
        
        # Apply weight bounds
        self.W_EE = np.clip(self.W_EE, self.W_min, self.W_max)
        
        # Important: Only remove connections that are effectively zero
        # This prevents over-pruning of the network
        self.W_EE[self.W_EE < 1e-6] = 0
        
    def _istdp(self, y_prev: np.ndarray, x_new: np.ndarray):
        """
        Inhibitory STDP
        ΔW_ij^EI = -η_istdp * y_j(t-1) * [1 - x_i(t) * (1 + 1/h_IP)]
        """
        
        # Compute weight changes
        factor = 1 - x_new * (1 + 1/self.h_ip)
        dW = -self.eta_istdp * np.outer(factor, y_prev)
        
        # Update weights
        self.W_EI += dW
        
        # Apply weight bounds with minimum to prevent complete removal of inhibition
        self.W_EI = np.clip(self.W_EI, self.W_ei_min, self.W_max)
        
    def _structural_plasticity(self):
        """
        Structural Plasticity: Create new synapses with probability p_sp
        New synapses start with weight W_ee_initial
        """
        
        if np.random.rand() < self.p_sp:
            # Find zero entries in W_EE (potential new connections)
            zero_mask = self.W_EE == 0
            # Exclude diagonal
            np.fill_diagonal(zero_mask, False)
            
            # Get indices of potential new connections
            zero_indices = np.argwhere(zero_mask)
            
            if len(zero_indices) > 0:
                # Select random location for new connection
                idx = np.random.randint(len(zero_indices))
                i, j = zero_indices[idx]
                
                # Create new connection with initial weight
                self.W_EE[i, j] = self.W_ee_initial
                
    def get_connection_fraction(self) -> float:
        """Calculate fraction of active connections in W_EE"""
        # Exclude diagonal from calculation
        mask = ~np.eye(self.N_E, dtype=bool)
        return (self.W_EE[mask] > 0).sum() / (self.N_E * (self.N_E - 1))
    
    def get_mean_weight(self) -> float:
        """Calculate mean weight of active connections"""
        active_weights = self.W_EE[self.W_EE > 0]
        return np.mean(active_weights) if len(active_weights) > 0 else 0
        
    def simulate(self, steps: int, input_pattern: Optional[np.ndarray] = None,
                 record_interval: int = 1000) -> Dict[str, np.ndarray]:
        """
        Run simulation for specified number of steps
        
        Parameters:
        -----------
        steps : int
            Number of simulation steps
        input_pattern : np.ndarray, optional
            Input pattern (steps × N_U). If None, zero input is used.
        record_interval : int
            Interval for recording connection fraction
            
        Returns:
        --------
        history : dict
            Dictionary containing state histories
        """
        
        # Initialize input
        if input_pattern is None or self.N_U == 0:
            input_pattern = np.zeros((steps, self.N_U)) if self.N_U > 0 else None
            
        # Initialize history
        history = {
            'x': np.zeros((steps, self.N_E)),
            'y': np.zeros((steps, self.N_I)),
            'T_E': np.zeros((steps, self.N_E)),
            'activity_E': np.zeros(steps),
            'activity_I': np.zeros(steps),
            'connection_fraction': [],
            'mean_weight': []
        }
        
        # Run simulation
        for t in range(steps):
            # Get input for this timestep
            if input_pattern is not None and self.N_U > 0:
                u_t = input_pattern[t] if t < len(input_pattern) else np.zeros(self.N_U)
            else:
                u_t = None
            
            # Update network
            self.step(u_t)
            
            # Record states
            history['x'][t] = self.x
            history['y'][t] = self.y
            history['T_E'][t] = self.T_E
            history['activity_E'][t] = np.mean(self.x)  # Mean activity
            history['activity_I'][t] = np.mean(self.y)
            
            # Record connection fraction and mean weight at intervals
            if t % record_interval == 0:
                history['connection_fraction'].append(self.get_connection_fraction())
                history['mean_weight'].append(self.get_mean_weight())
            
            # Progress report
            if t % 100000 == 0:
                print(f"Step {t}/{steps}, Activity: {history['activity_E'][t]:.3f}, "
                      f"Conn. Frac: {self.get_connection_fraction():.3f}")
        
        history['connection_fraction'] = np.array(history['connection_fraction'])
        history['mean_weight'] = np.array(history['mean_weight'])
        
        return history


def save_spiking_data(history: Dict[str, np.ndarray], filename: str = "spiking_data.h5"):
    """
    Save spiking data to HDF5 file
    
    Parameters:
    -----------
    history : dict
        Simulation history containing spike data
    filename : str
        Name of the HDF5 file to save
    """
    with h5py.File(filename, 'w') as f:
        # Save excitatory spikes (main raster data)
        f.create_dataset('spikes_E', data=history['x'], compression='gzip')
        
        # Save inhibitory spikes
        f.create_dataset('spikes_I', data=history['y'], compression='gzip')
        
        # Save thresholds
        f.create_dataset('thresholds_E', data=history['T_E'], compression='gzip')
        
        # Save activity
        f.create_dataset('activity_E', data=history['activity_E'])
        f.create_dataset('activity_I', data=history['activity_I'])
        
        # Save network statistics
        f.create_dataset('connection_fraction', data=history['connection_fraction'])
        f.create_dataset('mean_weight', data=history['mean_weight'])
        
        # Save metadata
        f.attrs['N_E'] = history['x'].shape[1]
        f.attrs['N_I'] = history['y'].shape[1]
        f.attrs['n_timesteps'] = history['x'].shape[0]
        
    print(f"Spiking data saved to {filename}")


def plot_raster_full_resolution(history: Dict[str, np.ndarray], 
                               save_path: str = "raster_full.pdf",
                               chunk_size: int = 100000):
    """
    Create a full-resolution raster plot where each spike is exactly 1 unit wide
    Saves as PDF to maintain perfect resolution
    
    Parameters:
    -----------
    history : dict
        Simulation history containing spike data
    save_path : str
        Path to save the raster plot (should be .pdf)
    chunk_size : int
        Process data in chunks to manage memory
    """
    
    print("Creating full-resolution raster plot...")
    
    # Get dimensions
    n_timesteps, n_neurons = history['x'].shape
    
    # Calculate figure dimensions
    # Each timestep = 1 point, each neuron = 1 point
    # Convert points to inches (72 points = 1 inch)
    width_inches = n_timesteps / 72.0  # This will be very long!
    height_inches = n_neurons / 72.0
    
    print(f"Creating figure: {width_inches:.1f} x {height_inches:.1f} inches")
    print(f"({n_timesteps} timesteps x {n_neurons} neurons)")
    
    # Create figure with exact dimensions
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=72)
    
    # Process in chunks to manage memory
    print("Processing spikes...")
    for chunk_start in range(0, n_timesteps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_timesteps)
        
        # Get spike data for this chunk
        spike_chunk = history['x'][chunk_start:chunk_end]
        
        # Find spikes in this chunk
        spike_times, neuron_ids = np.where(spike_chunk)
        
        if len(spike_times) > 0:
            # Plot spikes as vertical lines of exactly 1 unit width
            # Using vlines for perfect vertical lines
            ax.vlines(spike_times + chunk_start, 
                     neuron_ids - 0.5, 
                     neuron_ids + 0.5,
                     colors='black',
                     linewidth=1.0,
                     antialiased=False)  # No antialiasing for sharp lines
        
        if chunk_start % 500000 == 0:
            print(f"  Processed {chunk_start:,} / {n_timesteps:,} timesteps")
    
    # Set exact limits
    ax.set_xlim(0, n_timesteps)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    
    # Remove all margins and padding
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes
    
    # Remove all white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as PDF
    print(f"Saving to {save_path}...")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close()
    
    print(f"Full-resolution raster plot saved to {save_path}")
    print(f"File shows all {n_timesteps:,} timesteps with no information loss")


def plot_raster_with_activity(history: Dict[str, np.ndarray], 
                             save_path: str = "raster_with_activity.pdf",
                             activity_height: float = 2.0):
    """
    Create a full-resolution raster plot with activity trace below
    Each spike is exactly 1 unit wide, includes mean activity plot
    
    Parameters:
    -----------
    history : dict
        Simulation history containing spike data
    save_path : str
        Path to save the raster plot (should be .pdf)
    activity_height : float
        Height of activity plot in inches
    """
    
    print("Creating full-resolution raster plot with activity...")
    
    # Get dimensions
    n_timesteps, n_neurons = history['x'].shape
    
    # Calculate figure dimensions
    width_inches = n_timesteps / 72.0
    raster_height = n_neurons / 72.0
    total_height = raster_height + activity_height + 0.5  # 0.5 inch gap
    
    print(f"Creating figure: {width_inches:.1f} x {total_height:.1f} inches")
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(width_inches, total_height), dpi=72)
    
    # Calculate height ratios
    height_ratio_raster = raster_height / total_height
    height_ratio_activity = activity_height / total_height
    height_ratio_gap = 0.5 / total_height
    
    # Create axes manually for precise control
    ax_raster = plt.axes([0, height_ratio_activity + height_ratio_gap, 
                         1, height_ratio_raster])
    ax_activity = plt.axes([0, 0, 1, height_ratio_activity])
    
    # Plot raster
    print("Plotting raster...")
    chunk_size = 100000
    for chunk_start in range(0, n_timesteps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_timesteps)
        spike_chunk = history['x'][chunk_start:chunk_end]
        spike_times, neuron_ids = np.where(spike_chunk)
        
        if len(spike_times) > 0:
            ax_raster.vlines(spike_times + chunk_start, 
                           neuron_ids - 0.5, 
                           neuron_ids + 0.5,
                           colors='black',
                           linewidth=1.0,
                           antialiased=False)
        
        if chunk_start % 500000 == 0:
            print(f"  Processed {chunk_start:,} / {n_timesteps:,} timesteps")
    
    # Set raster limits
    ax_raster.set_xlim(0, n_timesteps)
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.axis('off')
    
    # Plot activity
    print("Plotting activity...")
    # Downsample activity for plotting if needed (but keep all information)
    ax_activity.fill_between(np.arange(n_timesteps), 
                           0, 
                           history['activity_E'],
                           color='black',
                           alpha=0.7,
                           linewidth=0)
    
    # Add target activity line
    ax_activity.axhline(y=0.1, color='red', linewidth=1, alpha=0.8)
    
    # Set activity limits
    ax_activity.set_xlim(0, n_timesteps)
    ax_activity.set_ylim(0, max(0.2, history['activity_E'].max() * 1.1))
    
    # Only show x-axis on activity plot
    ax_activity.spines['top'].set_visible(False)
    ax_activity.spines['right'].set_visible(False)
    ax_activity.set_xlabel('Time Steps', fontsize=10)
    ax_activity.set_ylabel('Mean Activity', fontsize=10)
    
    # Add tick marks every million timesteps
    tick_positions = np.arange(0, n_timesteps + 1, 1000000)
    ax_activity.set_xticks(tick_positions)
    ax_activity.set_xticklabels([f'{t/1e6:.0f}M' for t in tick_positions], fontsize=8)
    
    # Save as PDF
    print(f"Saving to {save_path}...")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=72)
    plt.close()
    
    print(f"Full-resolution raster plot with activity saved to {save_path}")


def plot_network_statistics(history: Dict[str, np.ndarray], save_path: str = "network_stats.png"):
    """
    Plot network statistics over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Connection fraction
    ax = axes[0, 0]
    steps = np.arange(len(history['connection_fraction'])) * 1000  # Assuming record_interval=1000
    ax.plot(steps, history['connection_fraction'] * 100, 'b-', linewidth=2)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Active Connections (%)')
    ax.set_title('Connection Fraction Evolution')
    ax.grid(True, alpha=0.3)
    
    # Mean weight
    ax = axes[0, 1]
    ax.plot(steps, history['mean_weight'], 'g-', linewidth=2)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mean Weight')
    ax.set_title('Mean Weight Evolution')
    ax.grid(True, alpha=0.3)
    
    # Activity over time (smoothed)
    ax = axes[1, 0]
    window = 1000
    activity_smooth = np.convolve(history['activity_E'], np.ones(window)/window, mode='valid')
    ax.plot(activity_smooth, 'r-', alpha=0.7, linewidth=1)
    ax.axhline(y=0.1, color='k', linestyle='--', label='Target (h_ip=0.1)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mean Activity')
    ax.set_title('Excitatory Activity (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Threshold distribution
    ax = axes[1, 1]
    # Show initial and final threshold distributions
    ax.hist(history['T_E'][0], bins=30, alpha=0.5, label='Initial', color='blue')
    ax.hist(history['T_E'][-1], bins=30, alpha=0.5, label='Final', color='red')
    ax.set_xlabel('Threshold Value')
    ax.set_ylabel('Count')
    ax.set_title('Threshold Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Network statistics saved to {save_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Create SORN network with parameters from the paper
    print("Creating SORN network...")
    sorn = SORN(
        N_E=200,
        N_I=40,
        N_U=0,  # No external input as in spontaneous condition
        eta_stdp=0.004,
        eta_istdp=0.001,
        eta_ip=0.01,
        h_ip=0.1,
        lambda_=20
    )
    
    # Run simulation
    print("\nRunning simulation...")
    steps = 3000000  # 3 million steps as in the paper
    history = sorn.simulate(steps, record_interval=1000)
    
    print(f"\nSimulation complete!")
    print(f"Final mean excitatory activity: {history['activity_E'][-10000:].mean():.3f}")
    print(f"Final mean inhibitory activity: {history['activity_I'][-10000:].mean():.3f}")
    print(f"Final mean threshold: {sorn.T_E.mean():.3f}")
    print(f"Final connection fraction: {history['connection_fraction'][-1]:.3f}")
    print(f"Final mean weight: {history['mean_weight'][-1]:.3f}")
    
    # Save spiking data
    print("\nSaving spiking data...")
    save_spiking_data(history, "plots/spiking_data.h5")
    
    # Create plots
    print("\nCreating plots...")
    
    # Full resolution raster plot of entire simulation
    plot_raster_full_resolution(history, save_path="plots/raster_full_3M.pdf")
    
    # Full resolution raster with activity trace
    plot_raster_with_activity(history, save_path="plots/raster_with_activity_3M.pdf")
    
    # Also create zoomed-in versions for easier viewing
    print("\nCreating zoomed raster plots...")
    
    # Create a subset for a zoomed view (last 100k timesteps)
    zoom_start = steps - 100000
    zoom_history = {
        'x': history['x'][zoom_start:],
        'activity_E': history['activity_E'][zoom_start:]
    }
    plot_raster_full_resolution(zoom_history, save_path="plots/raster_zoom_last100k.pdf")
    
    # Network statistics
    plot_network_statistics(history, save_path="plots/network_stats.png")
    
    print("\nAll plots saved to 'plots' directory!")