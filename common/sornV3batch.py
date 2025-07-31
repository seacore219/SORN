import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional, List, Any
import os
import h5py
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time  

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
        
    def simulate_incremental(self, steps: int, 
                           input_pattern: Optional[np.ndarray] = None,
                           save_interval: int = 10000,
                           record_interval: int = 1000,
                           filename: str = "spiking_data.h5") -> Dict[str, List]:
        """
        Run simulation with incremental saving to HDF5
        
        Parameters:
        -----------
        steps : int
            Number of simulation steps
        input_pattern : np.ndarray, optional
            Input pattern (steps × N_U). If None, zero input is used.
        save_interval : int
            Interval for saving data to disk (default: 10000)
        record_interval : int
            Interval for recording connection fraction
        filename : str
            HDF5 filename for saving data
            
        Returns:
        --------
        summary : dict
            Dictionary containing summary statistics (not full history)
        """
        
        # Initialize input
        if input_pattern is None or self.N_U == 0:
            input_pattern = np.zeros((steps, self.N_U)) if self.N_U > 0 else None
        
        # Create/open HDF5 file with appropriate datasets
        with h5py.File(filename, 'w') as f:
            # Create datasets with appropriate chunk sizes for efficient writing
            chunk_size = min(save_interval, 10000)
            
            # Main spike data
            dset_x = f.create_dataset('spikes_E', 
                                    shape=(steps, self.N_E),
                                    dtype='bool',
                                    chunks=(chunk_size, self.N_E),
                                    compression='gzip')
            
            dset_y = f.create_dataset('spikes_I',
                                    shape=(steps, self.N_I),
                                    dtype='bool', 
                                    chunks=(chunk_size, self.N_I),
                                    compression='gzip')
            
            # Thresholds (save less frequently to save space)
            threshold_save_interval = save_interval * 10
            n_threshold_saves = steps // threshold_save_interval + 1
            dset_T_E = f.create_dataset('thresholds_E',
                                      shape=(n_threshold_saves, self.N_E),
                                      dtype='float32',
                                      compression='gzip')
            
            # Activity (1D arrays)
            dset_act_E = f.create_dataset('activity_E',
                                        shape=(steps,),
                                        dtype='float32',
                                        chunks=(chunk_size,),
                                        compression='gzip')
            
            dset_act_I = f.create_dataset('activity_I',
                                        shape=(steps,),
                                        dtype='float32',
                                        chunks=(chunk_size,),
                                        compression='gzip')
            
            # Network statistics (recorded less frequently)
            n_stat_records = steps // record_interval + 1
            dset_conn_frac = f.create_dataset('connection_fraction',
                                            shape=(n_stat_records,),
                                            dtype='float32')
            
            dset_mean_weight = f.create_dataset('mean_weight',
                                              shape=(n_stat_records,),
                                              dtype='float32')
            
            # Save metadata
            f.attrs['N_E'] = self.N_E
            f.attrs['N_I'] = self.N_I
            f.attrs['n_timesteps'] = steps
            f.attrs['save_interval'] = save_interval
            f.attrs['record_interval'] = record_interval
            
            # Temporary buffers for batch writing
            buffer_x = np.zeros((save_interval, self.N_E), dtype=bool)
            buffer_y = np.zeros((save_interval, self.N_I), dtype=bool)
            buffer_act_E = np.zeros(save_interval, dtype=np.float32)
            buffer_act_I = np.zeros(save_interval, dtype=np.float32)
            
            # Summary statistics to return
            summary = {
                'final_activity_E': 0,
                'final_activity_I': 0,
                'final_mean_threshold': 0,
                'final_connection_fraction': 0,
                'final_mean_weight': 0,
                'activity_E_history': [],  # Keep a downsampled version in memory
                'connection_fraction_history': []
            }
            
            # Run simulation
            buffer_idx = 0
            stat_idx = 0
            threshold_idx = 0
            
            for t in range(steps):
                # Get input for this timestep
                if input_pattern is not None and self.N_U > 0:
                    u_t = input_pattern[t] if t < len(input_pattern) else np.zeros(self.N_U)
                else:
                    u_t = None
                
                # Update network
                self.step(u_t)
                
                # Store in buffer
                buffer_x[buffer_idx] = self.x
                buffer_y[buffer_idx] = self.y
                buffer_act_E[buffer_idx] = np.mean(self.x)
                buffer_act_I[buffer_idx] = np.mean(self.y)
                buffer_idx += 1
                
                # Save buffer to disk when full
                if buffer_idx >= save_interval or t == steps - 1:
                    # Calculate indices for this batch
                    start_idx = t - buffer_idx + 1
                    end_idx = t + 1
                    
                    # Write to HDF5
                    dset_x[start_idx:end_idx] = buffer_x[:buffer_idx]
                    dset_y[start_idx:end_idx] = buffer_y[:buffer_idx]
                    dset_act_E[start_idx:end_idx] = buffer_act_E[:buffer_idx]
                    dset_act_I[start_idx:end_idx] = buffer_act_I[:buffer_idx]
                    
                    # Reset buffer
                    buffer_idx = 0
                
                # Save thresholds less frequently
                if t % threshold_save_interval == 0:
                    dset_T_E[threshold_idx] = self.T_E
                    threshold_idx += 1
                
                # Record network statistics
                if t % record_interval == 0:
                    conn_frac = self.get_connection_fraction()
                    mean_weight = self.get_mean_weight()
                    dset_conn_frac[stat_idx] = conn_frac
                    dset_mean_weight[stat_idx] = mean_weight
                    stat_idx += 1
                    
                    # Keep some statistics in memory for plotting
                    summary['connection_fraction_history'].append(conn_frac)
                
                # Keep downsampled activity in memory (every 1000 steps)
                if t % 1000 == 0:
                    summary['activity_E_history'].append(np.mean(self.x))
                
                # Progress report
                if t % 100000 == 0:
                    print(f"Step {t}/{steps}, Activity: {np.mean(self.x):.3f}, "
                          f"Conn. Frac: {self.get_connection_fraction():.3f}")
            
            # Update summary statistics
            summary['final_activity_E'] = np.mean(self.x)
            summary['final_activity_I'] = np.mean(self.y)
            summary['final_mean_threshold'] = self.T_E.mean()
            summary['final_connection_fraction'] = self.get_connection_fraction()
            summary['final_mean_weight'] = self.get_mean_weight()
            
            # Convert lists to arrays
            summary['activity_E_history'] = np.array(summary['activity_E_history'])
            summary['connection_fraction_history'] = np.array(summary['connection_fraction_history'])
        
        print(f"\nSpiking data saved incrementally to {filename}")
        return summary


# Batch simulation functions
def run_single_simulation(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single simulation with given parameters
    
    Parameters:
    -----------
    sim_params : dict
        Dictionary containing simulation parameters including:
        - run_id: Unique identifier for this run
        - seed: Random seed for reproducibility
        - sorn_params: Parameters for SORN initialization
        - sim_steps: Number of simulation steps
        - output_dir: Directory to save results
        - save_full_data: Whether to save full spike data
        
    Returns:
    --------
    results : dict
        Summary results from the simulation
    """
    # Extract parameters
    run_id = sim_params['run_id']
    seed = sim_params['seed']
    sorn_params = sim_params['sorn_params']
    sim_steps = sim_params.get('sim_steps', 300000)
    output_dir = sim_params.get('output_dir', 'batch_results')
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nStarting run {run_id} with seed {seed}")
    print(f"Parameters: h_ip={sorn_params.get('h_ip', 0.1)}, "
          f"noise_sig={sorn_params.get('noise_sig', np.sqrt(0.05))}")
    
    # Create SORN network
    sorn = SORN(**sorn_params)
    
    # Run simulation - save h5 file
    filename = os.path.join(run_dir, "spiking_data.h5")
    
    summary = sorn.simulate_incremental(
        sim_steps,
        save_interval=10000,
        record_interval=1000,
        filename=filename
    )
    
    print(f"\nSimulation complete!")
    print(f"Final mean excitatory activity: {summary['final_activity_E']:.3f}")
    print(f"Final mean inhibitory activity: {summary['final_activity_I']:.3f}")
    print(f"Final mean threshold: {summary['final_mean_threshold']:.3f}")
    print(f"Final connection fraction: {summary['final_connection_fraction']:.3f}")
    print(f"Final mean weight: {summary['final_mean_weight']:.3f}")
    
    # Create plots from HDF5 file
    print("\nCreating plots from HDF5 file...")
    
    # Create coarse-grained versions for easier viewing
    print("\nCreating coarse-grained raster plots...")
    
    # Coarse-grained full simulation
    plot_raster_coarsegrained_from_hdf5(
        filename,
        save_path=os.path.join(run_dir, "raster_coarse_3M.png"),
        time_bin=1000,  # Bin every 1000 timesteps
        neuron_bin=1,    # Don't bin neurons
        max_width_inches=20.0,
        dpi=150
    )
    
    # Coarse-grained with activity
    plot_raster_with_activity_coarsegrained(
        filename,
        save_path=os.path.join(run_dir, "raster_activity_coarse_3M.png"),
        time_bin=100,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    # Also create zoomed-in versions for easier viewing
    print("\nCreating zoomed raster plots...")
    
    # Last 100k timesteps - coarse-grained
    plot_raster_with_activity_coarsegrained(
        filename,
        save_path=os.path.join(run_dir, "raster_zoom_last100k_coarse.png"),
        start_time=sim_steps-100000,
        duration=100000,
        time_bin=100,  # Finer binning for zoomed view
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    # Network statistics
    plot_network_statistics_from_hdf5(
        filename, 
        save_path=os.path.join(run_dir, "network_stats.png")
    )
    
    # Create connection fraction plot (like Fig 1A)
    print("Creating connection fraction plot...")
    with h5py.File(filename, 'r') as f:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Colors from Figure 1
        c_stable = '#2E4172'
        c_notstable = '#7887AB'
        
        # Get connection fraction data
        conn_frac = f['connection_fraction'][:]
        record_interval = f.attrs.get('record_interval', 1000)
        x_vals = np.arange(len(conn_frac)) * record_interval
        
        # Identify phases (simplified - assuming first half is growth/decay)
        phase1_end = len(conn_frac) // 2
        
        # Plot connection fraction
        ax.plot(x_vals[:phase1_end], conn_frac[:phase1_end] * 100, 
                color=c_notstable, linewidth=1.5, label='Growth/Decay')
        ax.plot(x_vals[phase1_end:], conn_frac[phase1_end:] * 100, 
                color=c_stable, linewidth=1.5, label='Stable')
        
        # Annotations
        ax.text(x_vals[phase1_end//2], conn_frac[phase1_end//2] * 100 + 0.5, 
                'growth/decay', fontsize=12, color=c_notstable)
        ax.text(x_vals[phase1_end] + x_vals[-1] * 0.1, conn_frac[phase1_end] * 100 + 0.5, 
                'stable', fontsize=12, color=c_stable)
        
        # Formatting
        ax.set_xlim([0, x_vals[-1]])
        ax.set_xlabel(r'$10^6$ time steps', fontsize=12)
        ax.set_ylabel('Active Connections (%)', fontsize=12)
        ax.set_title(f'Run {run_id}: Connection Fraction Evolution (h_ip={sorn_params.get("h_ip", 0.1)})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis to show in millions
        n_millions = int(x_vals[-1] / 1e6)
        ax.set_xticks(np.arange(0, n_millions + 1) * 1e6)
        ax.set_xticklabels([str(i) for i in range(n_millions + 1)])
        
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'connection_fraction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to {run_dir}!")
    
    # Save summary results
    results = {
        'run_id': run_id,
        'seed': seed,
        'parameters': sorn_params,
        'sim_steps': sim_steps,
        'final_activity_E': float(summary['final_activity_E']),
        'final_activity_I': float(summary['final_activity_I']),
        'final_mean_threshold': float(summary['final_mean_threshold']),
        'final_connection_fraction': float(summary['final_connection_fraction']),
        'final_mean_weight': float(summary['final_mean_weight']),
        'mean_activity_E': float(np.mean(summary['activity_E_history'])),
        'std_activity_E': float(np.std(summary['activity_E_history'])),
        'mean_connection_fraction': float(np.mean(summary['connection_fraction_history'])),
        'std_connection_fraction': float(np.std(summary['connection_fraction_history']))
    }
    
    # Save results to JSON
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed run {run_id}")
    return results

def compute_eigenvalue_spectrum(W_EE: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalue spectrum of the weight matrix
    
    Parameters:
    -----------
    W_EE : np.ndarray
        Excitatory-to-excitatory weight matrix
        
    Returns:
    --------
    eigenvalues : np.ndarray
        Complex eigenvalues sorted by magnitude
    eigenvectors : np.ndarray
        Corresponding eigenvectors
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(W_EE)
    
    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, 
                           save_path: str = "eigenvalue_spectrum.png",
                           title_suffix: str = ""):
    """
    Create eigenvalue spectrum plots
    
    Parameters:
    -----------
    eigenvalues : np.ndarray
        Complex eigenvalues
    save_path : str
        Path to save the plot
    title_suffix : str
        Additional text for the title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Complex plane plot
    ax = axes[0, 0]
    ax.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6, s=30)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='Unit circle')
    
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('Eigenvalues in Complex Plane')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    # 2. Eigenvalue magnitudes (sorted)
    ax = axes[0, 1]
    magnitudes = np.abs(eigenvalues)
    ax.plot(magnitudes, 'b-', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='|λ| = 1')
    ax.set_xlabel('Index')
    ax.set_ylabel('|λ|')
    ax.set_title('Eigenvalue Magnitudes (sorted)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # 3. Histogram of real parts
    ax = axes[1, 0]
    ax.hist(eigenvalues.real, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Real part')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Real Parts')
    ax.grid(True, alpha=0.3)
    
    # 4. Spectral radius and statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    spectral_radius = np.max(magnitudes)
    n_unstable = np.sum(magnitudes > 1)
    max_real = np.max(eigenvalues.real)
    
    stats_text = f"""
    Eigenvalue Statistics{title_suffix}:
    
    Spectral radius: {spectral_radius:.4f}
    Largest real part: {max_real:.4f}
    Number of |λ| > 1: {n_unstable} / {len(eigenvalues)}
    
    Mean |λ|: {np.mean(magnitudes):.4f}
    Std |λ|: {np.std(magnitudes):.4f}
    
    Network stability:
    {'STABLE' if spectral_radius <= 1 else 'UNSTABLE'} (spectral radius {'≤' if spectral_radius <= 1 else '>'} 1)
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    plt.suptitle(f'Eigenvalue Spectrum Analysis{title_suffix}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Eigenvalue spectrum plot saved to {save_path}")
    return spectral_radius, n_unstable


def analyze_eigenvalues_at_checkpoint(sorn: SORN, 
                                    checkpoint_name: str = "2M_steps") -> Dict[str, Any]:
    """
    Analyze eigenvalue spectrum at a specific checkpoint
    
    Parameters:
    -----------
    sorn : SORN
        SORN network instance
    checkpoint_name : str
        Name for this checkpoint
        
    Returns:
    --------
    analysis : dict
        Dictionary containing eigenvalue analysis results
    """
    print(f"\nAnalyzing eigenvalue spectrum at {checkpoint_name}...")
    
    # Compute eigenvalues
    eigenvalues, eigenvectors = compute_eigenvalue_spectrum(sorn.W_EE)
    
    # Create plots
    spectral_radius, n_unstable = plot_eigenvalue_spectrum(
        eigenvalues, 
        save_path=f"eigenvalue_spectrum_{checkpoint_name}.png",
        title_suffix=f" at {checkpoint_name}"
    )
    
    # Additional analysis
    magnitudes = np.abs(eigenvalues)
    
    analysis = {
        'checkpoint': checkpoint_name,
        'spectral_radius': float(spectral_radius),
        'n_unstable': int(n_unstable),
        'max_real_part': float(np.max(eigenvalues.real)),
        'mean_magnitude': float(np.mean(magnitudes)),
        'std_magnitude': float(np.std(magnitudes)),
        'n_real': int(np.sum(np.abs(eigenvalues.imag) < 1e-10)),
        'n_complex': int(np.sum(np.abs(eigenvalues.imag) >= 1e-10))
    }
    
    return analysis


def simulate_with_eigenvalue_checkpoints(sorn: SORN,
                                       total_steps: int = 3000000,
                                       checkpoints: List[int] = [1000000, 2000000, 3000000],
                                       save_dir: str = "eigenvalue_analysis"):
    """
    Run simulation with eigenvalue analysis at specified checkpoints
    
    Parameters:
    -----------
    sorn : SORN
        SORN network instance
    total_steps : int
        Total simulation steps
    checkpoints : List[int]
        Steps at which to analyze eigenvalues
    save_dir : str
        Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    eigenvalue_results = []
    current_step = 0
    
    for checkpoint in sorted(checkpoints):
        if checkpoint > total_steps:
            break
            
        # Run simulation until checkpoint
        steps_to_run = checkpoint - current_step
        print(f"\nRunning simulation from step {current_step} to {checkpoint}...")
        
        for _ in range(steps_to_run):
            sorn.step()
            current_step += 1
            
            if current_step % 100000 == 0:
                print(f"  Step {current_step}: Activity = {np.mean(sorn.x):.3f}")
        
        # Analyze eigenvalues at checkpoint
        checkpoint_name = f"{checkpoint//1000000}M_steps"
        analysis = analyze_eigenvalues_at_checkpoint(sorn, checkpoint_name)
        
        # Save the analysis
        analysis['activity'] = float(np.mean(sorn.x))
        analysis['connection_fraction'] = float(sorn.get_connection_fraction())
        eigenvalue_results.append(analysis)
        
        # Move plots to save directory
        import shutil
        plot_path = f"eigenvalue_spectrum_{checkpoint_name}.png"
        if os.path.exists(plot_path):
            shutil.move(plot_path, os.path.join(save_dir, plot_path))
    
    # Save all results
    with open(os.path.join(save_dir, 'eigenvalue_evolution.json'), 'w') as f:
        json.dump(eigenvalue_results, f, indent=2)
    
    # Create evolution plot
    plot_eigenvalue_evolution(eigenvalue_results, save_dir)
    
    return eigenvalue_results


def plot_eigenvalue_evolution(results: List[Dict[str, Any]], save_dir: str):
    """
    Plot how eigenvalue properties evolve over time
    """
    steps = [r['checkpoint'].split('_')[0].rstrip('M') for r in results]
    steps = [float(s) for s in steps]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spectral radius evolution
    ax = axes[0, 0]
    spectral_radii = [r['spectral_radius'] for r in results]
    ax.plot(steps, spectral_radii, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Stability boundary')
    ax.set_xlabel('Time (M steps)')
    ax.set_ylabel('Spectral Radius')
    ax.set_title('Spectral Radius Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Number of unstable eigenvalues
    ax = axes[0, 1]
    n_unstable = [r['n_unstable'] for r in results]
    ax.plot(steps, n_unstable, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Time (M steps)')
    ax.set_ylabel('Number of |λ| > 1')
    ax.set_title('Unstable Eigenvalues')
    ax.grid(True, alpha=0.3)
    
    # Maximum real part
    ax = axes[1, 0]
    max_real = [r['max_real_part'] for r in results]
    ax.plot(steps, max_real, 'g-o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (M steps)')
    ax.set_ylabel('Max Real Part')
    ax.set_title('Maximum Real Part Evolution')
    ax.grid(True, alpha=0.3)
    
    # Activity vs spectral radius
    ax = axes[1, 1]
    activities = [r['activity'] for r in results]
    ax.scatter(activities, spectral_radii, s=100, alpha=0.7)
    for i, step in enumerate(steps):
        ax.annotate(f'{step}M', (activities[i], spectral_radii[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Mean Activity')
    ax.set_ylabel('Spectral Radius')
    ax.set_title('Activity vs Spectral Radius')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Eigenvalue Properties Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eigenvalue_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()


# Add this method to the SORN class
def add_eigenvalue_analysis_to_simulate_incremental(self):
    """
    Modified simulate_incremental method that includes eigenvalue analysis
    This should be added to the SORN class
    """
    # Add this inside the simulate_incremental method, after the main simulation loop
    # Around line 800-900 in your original code
    
    # Example modification to simulate_incremental:
    # After the simulation loop completes, add:
    
    # if steps >= 2000000:  # Only analyze if we ran at least 2M steps
    #     print("\nAnalyzing eigenvalue spectrum of stabilized network...")
    #     eigenvalues, _ = compute_eigenvalue_spectrum(self.W_EE)
    #     spectral_radius, n_unstable = plot_eigenvalue_spectrum(
    #         eigenvalues,
    #         save_path=os.path.join(os.path.dirname(filename), "eigenvalue_spectrum_final.png"),
    #         title_suffix=" (Final Network State)"
    #     )
    #     
    #     summary['eigenvalue_analysis'] = {
    #         'spectral_radius': float(spectral_radius),
    #         'n_unstable': int(n_unstable),
    #         'max_real_part': float(np.max(eigenvalues.real)),
    #         'analyzed_at_step': steps
    #     }


# Example usage:
if __name__ == "__main__":
    # Example 1: Analyze eigenvalues during simulation
    print("Running SORN with eigenvalue analysis...")
    
    # Create SORN network
    sorn = SORN(
        N_E=200,
        N_I=40,
        h_ip=0.1,
        eta_stdp=0.004,
        eta_istdp=0.001,
        eta_ip=0.01,
        lambda_=20,
        noise_sig=np.sqrt(0.05)
    )
    
    # Run simulation with eigenvalue checkpoints
    eigenvalue_results = simulate_with_eigenvalue_checkpoints(
        sorn,
        total_steps=3000000,
        checkpoints=[1000000, 2000000, 3000000],
        save_dir="eigenvalue_analysis"
    )
    
    # Print summary
    print("\nEigenvalue Analysis Summary:")
    for result in eigenvalue_results:
        print(f"\n{result['checkpoint']}:")
        print(f"  Spectral radius: {result['spectral_radius']:.4f}")
        print(f"  Unstable eigenvalues: {result['n_unstable']}")
        print(f"  Network activity: {result['activity']:.3f}")

def run_single_simulation_with_eigenvalues(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single simulation with eigenvalue analysis after stabilization
    
    Parameters:
    -----------
    sim_params : dict
        Dictionary containing simulation parameters including:
        - run_id: Unique identifier for this run
        - seed: Random seed for reproducibility
        - sorn_params: Parameters for SORN initialization
        - sim_steps: Number of simulation steps
        - output_dir: Directory to save results
        - save_full_data: Whether to save full spike data
        - eigenvalue_checkpoint: Step at which to analyze eigenvalues (default: 2000000)
        
    Returns:
    --------
    results : dict
        Summary results from the simulation including eigenvalue analysis
    """
    # Extract parameters
    run_id = sim_params['run_id']
    seed = sim_params['seed']
    sorn_params = sim_params['sorn_params']
    sim_steps = sim_params.get('sim_steps', 3000000)
    output_dir = sim_params.get('output_dir', 'batch_results')
    eigenvalue_checkpoint = sim_params.get('eigenvalue_checkpoint', 2000000)
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nStarting run {run_id} with seed {seed}")
    print(f"Parameters: h_ip={sorn_params.get('h_ip', 0.1)}, "
          f"noise_sig={sorn_params.get('noise_sig', np.sqrt(0.05))}")
    
    # Create SORN network
    sorn = SORN(**sorn_params)
    
    # Run simulation - save h5 file
    filename = os.path.join(run_dir, "spiking_data.h5")
    
    # Modified simulation to save W_EE at checkpoint
    eigenvalue_analysis = None
    
    # Initialize HDF5 file with additional dataset for weight matrix
    with h5py.File(filename, 'w') as f:
        # Create all the standard datasets (as in original code)
        chunk_size = min(10000, sim_steps)
        
        dset_x = f.create_dataset('spikes_E', 
                                shape=(sim_steps, sorn.N_E),
                                dtype='bool',
                                chunks=(chunk_size, sorn.N_E),
                                compression='gzip')
        
        dset_y = f.create_dataset('spikes_I',
                                shape=(sim_steps, sorn.N_I),
                                dtype='bool', 
                                chunks=(chunk_size, sorn.N_I),
                                compression='gzip')
        
        threshold_save_interval = 10000
        n_threshold_saves = sim_steps // threshold_save_interval + 1
        dset_T_E = f.create_dataset('thresholds_E',
                                  shape=(n_threshold_saves, sorn.N_E),
                                  dtype='float32',
                                  compression='gzip')
        
        dset_act_E = f.create_dataset('activity_E',
                                    shape=(sim_steps,),
                                    dtype='float32',
                                    chunks=(chunk_size,),
                                    compression='gzip')
        
        dset_act_I = f.create_dataset('activity_I',
                                    shape=(sim_steps,),
                                    dtype='float32',
                                    chunks=(chunk_size,),
                                    compression='gzip')
        
        record_interval = 1000
        n_stat_records = sim_steps // record_interval + 1
        dset_conn_frac = f.create_dataset('connection_fraction',
                                        shape=(n_stat_records,),
                                        dtype='float32')
        
        dset_mean_weight = f.create_dataset('mean_weight',
                                          shape=(n_stat_records,),
                                          dtype='float32')
        
        # Create dataset for weight matrix at checkpoint
        dset_W_EE_checkpoint = f.create_dataset('W_EE_checkpoint',
                                              shape=(sorn.N_E, sorn.N_E),
                                              dtype='float32')
        
        # Save metadata
        f.attrs['N_E'] = sorn.N_E
        f.attrs['N_I'] = sorn.N_I
        f.attrs['n_timesteps'] = sim_steps
        f.attrs['save_interval'] = 10000
        f.attrs['record_interval'] = record_interval
        f.attrs['eigenvalue_checkpoint'] = eigenvalue_checkpoint
        
        # Temporary buffers
        save_interval = 10000
        buffer_x = np.zeros((save_interval, sorn.N_E), dtype=bool)
        buffer_y = np.zeros((save_interval, sorn.N_I), dtype=bool)
        buffer_act_E = np.zeros(save_interval, dtype=np.float32)
        buffer_act_I = np.zeros(save_interval, dtype=np.float32)
        
        # Run simulation
        buffer_idx = 0
        stat_idx = 0
        threshold_idx = 0
        
        for t in range(sim_steps):
            # Update network
            sorn.step()
            
            # Store in buffer
            buffer_x[buffer_idx] = sorn.x
            buffer_y[buffer_idx] = sorn.y
            buffer_act_E[buffer_idx] = np.mean(sorn.x)
            buffer_act_I[buffer_idx] = np.mean(sorn.y)
            buffer_idx += 1
            
            # Save buffer to disk when full
            if buffer_idx >= save_interval or t == sim_steps - 1:
                start_idx = t - buffer_idx + 1
                end_idx = t + 1
                
                dset_x[start_idx:end_idx] = buffer_x[:buffer_idx]
                dset_y[start_idx:end_idx] = buffer_y[:buffer_idx]
                dset_act_E[start_idx:end_idx] = buffer_act_E[:buffer_idx]
                dset_act_I[start_idx:end_idx] = buffer_act_I[:buffer_idx]
                
                buffer_idx = 0
            
            # Save thresholds
            if t % threshold_save_interval == 0:
                dset_T_E[threshold_idx] = sorn.T_E
                threshold_idx += 1
            
            # Record network statistics
            if t % record_interval == 0:
                dset_conn_frac[stat_idx] = sorn.get_connection_fraction()
                dset_mean_weight[stat_idx] = sorn.get_mean_weight()
                stat_idx += 1
            
            # Eigenvalue analysis at checkpoint
            if t == eigenvalue_checkpoint - 1:
                print(f"\nPerforming eigenvalue analysis at step {t+1}...")
                
                # Save weight matrix
                dset_W_EE_checkpoint[:] = sorn.W_EE
                
                # Compute eigenvalues
                eigenvalues, _ = compute_eigenvalue_spectrum(sorn.W_EE)
                
                # Create eigenvalue plot
                spectral_radius, n_unstable = plot_eigenvalue_spectrum(
                    eigenvalues,
                    save_path=os.path.join(run_dir, f"eigenvalue_spectrum_{eigenvalue_checkpoint//1000000}M.png"),
                    title_suffix=f" at {eigenvalue_checkpoint//1000000}M steps"
                )
                
                # Store analysis results
                eigenvalue_analysis = {
                    'checkpoint_step': eigenvalue_checkpoint,
                    'spectral_radius': float(spectral_radius),
                    'n_unstable': int(n_unstable),
                    'max_real_part': float(np.max(eigenvalues.real)),
                    'mean_magnitude': float(np.mean(np.abs(eigenvalues))),
                    'std_magnitude': float(np.std(np.abs(eigenvalues))),
                    'activity_at_checkpoint': float(np.mean(sorn.x)),
                    'connection_fraction_at_checkpoint': float(sorn.get_connection_fraction())
                }
                
                print(f"Eigenvalue analysis complete:")
                print(f"  Spectral radius: {spectral_radius:.4f}")
                print(f"  Unstable eigenvalues: {n_unstable}/{sorn.N_E}")
                print(f"  Network stability: {'STABLE' if spectral_radius <= 1 else 'UNSTABLE'}")
            
            # Progress report
            if t % 100000 == 0:
                print(f"Step {t}/{sim_steps}, Activity: {np.mean(sorn.x):.3f}, "
                      f"Conn. Frac: {sorn.get_connection_fraction():.3f}")
    
    print(f"\nSimulation complete!")
    
    # Create standard plots
    print("\nCreating plots...")
    
    # Create all the standard plots (as in original)
    plot_raster_coarsegrained_from_hdf5(
        filename,
        save_path=os.path.join(run_dir, "raster_coarse_3M.png"),
        time_bin=1000,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    plot_raster_with_activity_coarsegrained(
        filename,
        save_path=os.path.join(run_dir, "raster_activity_coarse_3M.png"),
        time_bin=100,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    plot_network_statistics_from_hdf5(
        filename, 
        save_path=os.path.join(run_dir, "network_stats.png")
    )
    
    # Create connection fraction plot
    print("Creating connection fraction plot...")
    with h5py.File(filename, 'r') as f:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        c_stable = '#2E4172'
        c_notstable = '#7887AB'
        
        conn_frac = f['connection_fraction'][:]
        record_interval = f.attrs.get('record_interval', 1000)
        x_vals = np.arange(len(conn_frac)) * record_interval
        
        phase1_end = len(conn_frac) // 2
        
        ax.plot(x_vals[:phase1_end], conn_frac[:phase1_end] * 100, 
                color=c_notstable, linewidth=1.5, label='Growth/Decay')
        ax.plot(x_vals[phase1_end:], conn_frac[phase1_end:] * 100, 
                color=c_stable, linewidth=1.5, label='Stable')
        
        # Mark eigenvalue checkpoint
        if eigenvalue_checkpoint <= x_vals[-1]:
            ax.axvline(x=eigenvalue_checkpoint, color='red', linestyle='--', 
                      alpha=0.7, label=f'Eigenvalue analysis ({eigenvalue_checkpoint//1000000}M)')
        
        ax.text(x_vals[phase1_end//2], conn_frac[phase1_end//2] * 100 + 0.5, 
                'growth/decay', fontsize=12, color=c_notstable)
        ax.text(x_vals[phase1_end] + x_vals[-1] * 0.1, conn_frac[phase1_end] * 100 + 0.5, 
                'stable', fontsize=12, color=c_stable)
        
        ax.set_xlim([0, x_vals[-1]])
        ax.set_xlabel(r'$10^6$ time steps', fontsize=12)
        ax.set_ylabel('Active Connections (%)', fontsize=12)
        ax.set_title(f'Run {run_id}: Connection Fraction Evolution (h_ip={sorn_params.get("h_ip", 0.1)})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        n_millions = int(x_vals[-1] / 1e6)
        ax.set_xticks(np.arange(0, n_millions + 1) * 1e6)
        ax.set_xticklabels([str(i) for i in range(n_millions + 1)])
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'connection_fraction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to {run_dir}!")
    
    # Prepare results
    results = {
        'run_id': run_id,
        'seed': seed,
        'parameters': sorn_params,
        'sim_steps': sim_steps,
        'final_activity_E': float(np.mean(sorn.x)),
        'final_activity_I': float(np.mean(sorn.y)),
        'final_mean_threshold': float(sorn.T_E.mean()),
        'final_connection_fraction': float(sorn.get_connection_fraction()),
        'final_mean_weight': float(sorn.get_mean_weight())
    }
    
    # Add eigenvalue analysis if performed
    if eigenvalue_analysis is not None:
        results['eigenvalue_analysis'] = eigenvalue_analysis
    
    # Save results to JSON
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed run {run_id}")
    return results


# Modified parameter sweep function that uses the new simulation function
def run_parameter_sweep_with_eigenvalues(param_ranges: Dict[str, List[Any]],
                                        base_params: Optional[Dict[str, Any]] = None,
                                        sim_steps: int = 3000000,
                                        eigenvalue_checkpoint: int = 2000000,
                                        output_dir: str = "sweep_results",
                                        n_processes: Optional[int] = None,
                                        n_seeds: int = 3,
                                        base_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Run parameter sweep with eigenvalue analysis
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"sweep_eigenvalues_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Default base parameters
    if base_params is None:
        base_params = {
            'N_E': 200,
            'N_I': 40,
            'h_ip': 0.1,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        }
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    import itertools
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Parameter sweep with eigenvalue analysis:")
    for name, values in param_ranges.items():
        print(f"  {name}: {values}")
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Seeds per combination: {n_seeds}")
    print(f"Total runs: {len(all_combinations) * n_seeds}")
    print(f"Eigenvalue checkpoint: {eigenvalue_checkpoint} steps")
    
    # Prepare simulation parameters
    sim_params_list = []
    run_id = 0
    
    for combo in all_combinations:
        combo_params = base_params.copy()
        for name, value in zip(param_names, combo):
            combo_params[name] = value
        
        param_folder = '_'.join([f'{name}_{value}' for name, value in zip(param_names, combo)])
        param_dir = os.path.join(full_output_dir, param_folder)
        
        for seed_idx in range(n_seeds):
            run_name = f"seed_{seed_idx}"
            
            sim_params = {
                'run_id': run_name,
                'seed': base_seed + run_id,
                'sorn_params': combo_params.copy(),
                'sim_steps': sim_steps,
                'output_dir': param_dir,
                'eigenvalue_checkpoint': eigenvalue_checkpoint
            }
            sim_params_list.append(sim_params)
            run_id += 1
    
    # Save configuration
    sweep_config = {
        'param_ranges': param_ranges,
        'base_params': base_params,
        'sim_steps': sim_steps,
        'eigenvalue_checkpoint': eigenvalue_checkpoint,
        'n_seeds': n_seeds,
        'total_runs': len(sim_params_list),
        'timestamp': timestamp,
        'base_seed': base_seed
    }
    with open(os.path.join(full_output_dir, 'sweep_config.json'), 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"\nRunning parameter sweep using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation_with_eigenvalues, sim_params_list)
    
    # Save results
    with open(os.path.join(full_output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create eigenvalue comparison plots
    plot_eigenvalue_comparison(results, param_ranges, full_output_dir, n_seeds)
    
    print(f"\nParameter sweep complete. Results saved to {full_output_dir}")
    return results


def plot_eigenvalue_comparison(results: List[Dict[str, Any]], 
                             param_ranges: Dict[str, List[Any]],
                             output_dir: str,
                             n_seeds: int):
    """
    Create comparison plots for eigenvalue analysis across parameter values
    """
    param_names = list(param_ranges.keys())
    
    if len(param_names) == 1:
        # 1D parameter sweep
        param_name = param_names[0]
        param_values = param_ranges[param_name]
        
        # Organize results by parameter value
        results_by_value = {v: [] for v in param_values}
        for r in results:
            if 'eigenvalue_analysis' in r:
                value = r['parameters'][param_name]
                results_by_value[value].append(r)
        
        # Calculate statistics
        spectral_radii_mean = []
        spectral_radii_std = []
        n_unstable_mean = []
        activities_mean = []
        
        for v in param_values:
            if results_by_value[v]:
                spectral_radii = [r['eigenvalue_analysis']['spectral_radius'] 
                                 for r in results_by_value[v]]
                n_unstable = [r['eigenvalue_analysis']['n_unstable'] 
                             for r in results_by_value[v]]
                activities = [r['eigenvalue_analysis']['activity_at_checkpoint'] 
                            for r in results_by_value[v]]
                
                spectral_radii_mean.append(np.mean(spectral_radii))
                spectral_radii_std.append(np.std(spectral_radii))
                n_unstable_mean.append(np.mean(n_unstable))
                activities_mean.append(np.mean(activities))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Spectral radius vs parameter
        ax1.errorbar(param_values[:len(spectral_radii_mean)], spectral_radii_mean, 
                    yerr=spectral_radii_std, marker='o', capsize=5, 
                    capthick=2, markersize=8, linewidth=2)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, 
                   label='Stability boundary')
        ax1.set_xlabel(param_name, fontsize=12)
        ax1.set_ylabel('Spectral Radius', fontsize=12)
        ax1.set_title(f'Spectral Radius vs {param_name} (n={n_seeds} per point)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Activity vs spectral radius
        ax2.scatter(activities_mean, spectral_radii_mean, s=100, alpha=0.7)
        for i, v in enumerate(param_values[:len(activities_mean)]):
            ax2.annotate(f'{param_name}={v}', 
                        (activities_mean[i], spectral_radii_mean[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Mean Activity at Checkpoint', fontsize=12)
        ax2.set_ylabel('Spectral Radius', fontsize=12)
        ax2.set_title('Activity vs Spectral Radius', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eigenvalue_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def run_batch_simulations(n_runs: int = 3,
                         base_params: Optional[Dict[str, Any]] = None,
                         sim_steps: int = 300000,
                         output_dir: str = "batch_results",
                         n_processes: Optional[int] = None,
                         save_full_data: bool = True,
                         base_seed: int = 69) -> List[Dict[str, Any]]:
    """
    Run multiple identical simulations with different random seeds
    
    Parameters:
    -----------
    n_runs : int
        Number of simulations to run
    base_params : dict, optional
        Base parameters for SORN (if None, uses defaults)
    sim_steps : int
        Number of simulation steps
    output_dir : str
        Directory to save results
    n_processes : int, optional
        Number of parallel processes (if None, uses CPU count)
    save_full_data : bool
        Whether to save full spike data for each run
    base_seed : int
        Base random seed (each run gets base_seed + run_id)
    
    Returns:
    --------
    results : list
        List of summary dictionaries from all runs
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Default parameters
    if base_params is None:
        base_params = {
            'N_E': 200,
            'N_I': 40,
            'h_ip': 0.1,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        }
    
    # Prepare simulation parameters for each run
    sim_params_list = []
    for i in range(n_runs):
        sim_params = {
            'run_id': i,
            'seed': base_seed + i,
            'sorn_params': base_params.copy(),
            'sim_steps': sim_steps,
            'output_dir': full_output_dir,
            'save_full_data': save_full_data
        }
        sim_params_list.append(sim_params)
    
    # Save batch configuration
    batch_config = {
        'n_runs': n_runs,
        'base_params': base_params,
        'sim_steps': sim_steps,
        'timestamp': timestamp,
        'base_seed': base_seed
    }
    with open(os.path.join(full_output_dir, 'batch_config.json'), 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    # Run simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"Running {n_runs} simulations using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, sim_params_list)
    
    # Save combined results
    with open(os.path.join(full_output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plots
    plot_batch_summary(results, full_output_dir)
    
    print(f"\nBatch simulation complete. Results saved to {full_output_dir}")
    return results


def run_parameter_sweep(param_ranges: Dict[str, List[Any]],
                       base_params: Optional[Dict[str, Any]] = None,
                       sim_steps: int = 300000,
                       output_dir: str = "sweep_results",
                       n_processes: Optional[int] = None,
                       n_seeds: int = 3,
                       save_full_data: bool = True,
                       base_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Run parameter sweep over specified parameter ranges
    
    Parameters:
    -----------
    param_ranges : dict
        Dictionary mapping parameter names to lists of values to test
        Example: {'h_ip': [0.05, 0.1, 0.15], 'noise_sig': [0.01, 0.1, 1.0]}
    base_params : dict, optional
        Base parameters for SORN (swept parameters will override these)
    sim_steps : int
        Number of simulation steps
    output_dir : str
        Directory to save results
    n_processes : int, optional
        Number of parallel processes
    n_seeds : int
        Number of random seeds to test for each parameter combination
    save_full_data : bool
        Whether to save full spike data
    base_seed : int
        Base random seed
    
    Returns:
    --------
    results : list
        List of summary dictionaries from all runs
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"sweep_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Default base parameters
    if base_params is None:
        base_params = {
            'N_E': 200,
            'N_I': 40,
            'h_ip': 0.1,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        }
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Create all combinations
    import itertools
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Parameter sweep:")
    for name, values in param_ranges.items():
        print(f"  {name}: {values}")
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Seeds per combination: {n_seeds}")
    print(f"Total runs: {len(all_combinations) * n_seeds}")
    
    # Prepare simulation parameters
    sim_params_list = []
    run_id = 0
    
    for combo in all_combinations:
        # Create parameter dict for this combination
        combo_params = base_params.copy()
        for name, value in zip(param_names, combo):
            combo_params[name] = value
        
        # Create parameter folder name
        param_folder = '_'.join([f'{name}_{value}' for name, value in zip(param_names, combo)])
        param_dir = os.path.join(full_output_dir, param_folder)
        
        # Run multiple seeds for this combination
        for seed_idx in range(n_seeds):
            # Each seed gets its own subfolder within the parameter folder
            run_name = f"seed_{seed_idx}"
            
            sim_params = {
                'run_id': run_name,
                'seed': base_seed + run_id,
                'sorn_params': combo_params.copy(),
                'sim_steps': sim_steps,
                'output_dir': param_dir,  # Use parameter-specific directory
                'save_full_data': save_full_data
            }
            sim_params_list.append(sim_params)
            run_id += 1
    
    # Save sweep configuration
    sweep_config = {
        'param_ranges': param_ranges,
        'base_params': base_params,
        'sim_steps': sim_steps,
        'n_seeds': n_seeds,
        'total_runs': len(sim_params_list),
        'timestamp': timestamp,
        'base_seed': base_seed
    }
    with open(os.path.join(full_output_dir, 'sweep_config.json'), 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"\nRunning parameter sweep using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, sim_params_list)
    
    # Save combined results
    with open(os.path.join(full_output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create parameter sweep plots
    plot_parameter_sweep_results(results, param_ranges, full_output_dir, n_seeds)
    
    print(f"\nParameter sweep complete. Results saved to {full_output_dir}")
    return results



def plot_batch_summary(results: List[Dict[str, Any]], output_dir: str):
    """Create summary plots for batch simulations"""
    
    # Extract data
    final_activities = [r['final_activity_E'] for r in results]
    final_conn_fracs = [r['final_connection_fraction'] for r in results]
    mean_activities = [r['mean_activity_E'] for r in results]
    mean_conn_fracs = [r['mean_connection_fraction'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Final activity distribution
    ax = axes[0, 0]
    ax.hist(final_activities, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_activities), color='red', linestyle='--', 
              label=f'Mean: {np.mean(final_activities):.3f}')
    ax.set_xlabel('Final Activity (E)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Activities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final connection fraction distribution
    ax = axes[0, 1]
    ax.hist(final_conn_fracs, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_conn_fracs), color='red', linestyle='--',
              label=f'Mean: {np.mean(final_conn_fracs):.3f}')
    ax.set_xlabel('Final Connection Fraction')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Connection Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean activity vs connection fraction
    ax = axes[1, 0]
    ax.scatter(mean_activities, mean_conn_fracs, alpha=0.6)
    ax.set_xlabel('Mean Activity (E)')
    ax.set_ylabel('Mean Connection Fraction')
    ax.set_title('Activity vs Connection Fraction')
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Batch Summary (n={len(results)}):
    
    Final Activity:
      Mean: {np.mean(final_activities):.4f} ± {np.std(final_activities):.4f}
      Range: [{np.min(final_activities):.4f}, {np.max(final_activities):.4f}]
    
    Final Connection Fraction:
      Mean: {np.mean(final_conn_fracs):.4f} ± {np.std(final_conn_fracs):.4f}
      Range: [{np.min(final_conn_fracs):.4f}, {np.max(final_conn_fracs):.4f}]
    
    Mean Activity (over time):
      Mean: {np.mean(mean_activities):.4f} ± {np.std(mean_activities):.4f}
    
    Mean Connection Fraction (over time):
      Mean: {np.mean(mean_conn_fracs):.4f} ± {np.std(mean_conn_fracs):.4f}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_sweep_results(results: List[Dict[str, Any]], 
                                param_ranges: Dict[str, List[Any]],
                                output_dir: str,
                                n_seeds: int):
    """Create plots for parameter sweep results"""
    
    # Organize results by parameter values
    param_names = list(param_ranges.keys())
    
    if len(param_names) == 1:
        # 1D parameter sweep
        plot_1d_parameter_sweep(results, param_names[0], param_ranges[param_names[0]], 
                               output_dir, n_seeds)
    elif len(param_names) == 2:
        # 2D parameter sweep
        plot_2d_parameter_sweep(results, param_names, param_ranges, output_dir, n_seeds)
    else:
        print(f"Plotting for {len(param_names)}D parameter sweeps not implemented. "
              f"Results saved to JSON.")


def plot_1d_parameter_sweep(results: List[Dict[str, Any]], 
                           param_name: str,
                           param_values: List[Any],
                           output_dir: str,
                           n_seeds: int):
    """Plot results for 1D parameter sweep"""
    
    # Organize results by parameter value
    results_by_value = {v: [] for v in param_values}
    for r in results:
        value = r['parameters'][param_name]
        results_by_value[value].append(r)
    
    # Calculate statistics
    means_activity = []
    stds_activity = []
    means_conn_frac = []
    stds_conn_frac = []
    
    for v in param_values:
        activities = [r['final_activity_E'] for r in results_by_value[v]]
        conn_fracs = [r['final_connection_fraction'] for r in results_by_value[v]]
        
        means_activity.append(np.mean(activities))
        stds_activity.append(np.std(activities))
        means_conn_frac.append(np.mean(conn_fracs))
        stds_conn_frac.append(np.std(conn_fracs))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Activity plot
    ax1.errorbar(param_values, means_activity, yerr=stds_activity, 
                marker='o', capsize=5, capthick=2, markersize=8)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Final Activity (E)')
    ax1.set_title(f'Activity vs {param_name} (n={n_seeds} per point)')
    ax1.grid(True, alpha=0.3)
    if param_name == 'noise_sig' and min(param_values) > 0:
        ax1.set_xscale('log')
    
    # Connection fraction plot
    ax2.errorbar(param_values, means_conn_frac, yerr=stds_conn_frac,
                marker='o', capsize=5, capthick=2, markersize=8)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Final Connection Fraction')
    ax2.set_title(f'Connection Fraction vs {param_name} (n={n_seeds} per point)')
    ax2.grid(True, alpha=0.3)
    if param_name == 'noise_sig' and min(param_values) > 0:
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_1d_{param_name}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_2d_parameter_sweep(results: List[Dict[str, Any]],
                           param_names: List[str],
                           param_ranges: Dict[str, List[Any]],
                           output_dir: str,
                           n_seeds: int):
    """Plot results for 2D parameter sweep"""
    
    param1, param2 = param_names
    values1 = param_ranges[param1]
    values2 = param_ranges[param2]
    
    # Create grids for results
    activity_grid = np.zeros((len(values2), len(values1)))
    conn_frac_grid = np.zeros((len(values2), len(values1)))
    
    # Fill grids
    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            # Find matching results
            matching = [r for r in results 
                       if r['parameters'][param1] == v1 and r['parameters'][param2] == v2]
            
            if matching:
                activity_grid[j, i] = np.mean([r['final_activity_E'] for r in matching])
                conn_frac_grid[j, i] = np.mean([r['final_connection_fraction'] for r in matching])
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Activity heatmap
    im1 = ax1.imshow(activity_grid, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xticks(range(len(values1)))
    ax1.set_xticklabels([f'{v:.3g}' for v in values1])
    ax1.set_yticks(range(len(values2)))
    ax1.set_yticklabels([f'{v:.3g}' for v in values2])
    ax1.set_xlabel(param1)
    ax1.set_ylabel(param2)
    ax1.set_title(f'Final Activity (n={n_seeds} per cell)')
    plt.colorbar(im1, ax=ax1)
    
    # Connection fraction heatmap
    im2 = ax2.imshow(conn_frac_grid, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xticks(range(len(values1)))
    ax2.set_xticklabels([f'{v:.3g}' for v in values1])
    ax2.set_yticks(range(len(values2)))
    ax2.set_yticklabels([f'{v:.3g}' for v in values2])
    ax2.set_xlabel(param1)
    ax2.set_ylabel(param2)
    ax2.set_title(f'Final Connection Fraction (n={n_seeds} per cell)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_2d_{param1}_{param2}.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from typing import Dict, Tuple, Optional, List, Any
import os
import h5py
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time  

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
        
    def simulate_incremental(self, steps: int, 
                           input_pattern: Optional[np.ndarray] = None,
                           save_interval: int = 10000,
                           record_interval: int = 1000,
                           filename: str = "spiking_data.h5") -> Dict[str, List]:
        """
        Run simulation with incremental saving to HDF5
        
        Parameters:
        -----------
        steps : int
            Number of simulation steps
        input_pattern : np.ndarray, optional
            Input pattern (steps × N_U). If None, zero input is used.
        save_interval : int
            Interval for saving data to disk (default: 10000)
        record_interval : int
            Interval for recording connection fraction
        filename : str
            HDF5 filename for saving data
            
        Returns:
        --------
        summary : dict
            Dictionary containing summary statistics (not full history)
        """
        
        # Initialize input
        if input_pattern is None or self.N_U == 0:
            input_pattern = np.zeros((steps, self.N_U)) if self.N_U > 0 else None
        
        # Create/open HDF5 file with appropriate datasets
        with h5py.File(filename, 'w') as f:
            # Create datasets with appropriate chunk sizes for efficient writing
            chunk_size = min(save_interval, 10000)
            
            # Main spike data
            dset_x = f.create_dataset('spikes_E', 
                                    shape=(steps, self.N_E),
                                    dtype='bool',
                                    chunks=(chunk_size, self.N_E),
                                    compression='gzip')
            
            dset_y = f.create_dataset('spikes_I',
                                    shape=(steps, self.N_I),
                                    dtype='bool', 
                                    chunks=(chunk_size, self.N_I),
                                    compression='gzip')
            
            # Thresholds (save less frequently to save space)
            threshold_save_interval = save_interval * 10
            n_threshold_saves = steps // threshold_save_interval + 1
            dset_T_E = f.create_dataset('thresholds_E',
                                      shape=(n_threshold_saves, self.N_E),
                                      dtype='float32',
                                      compression='gzip')
            
            # Activity (1D arrays)
            dset_act_E = f.create_dataset('activity_E',
                                        shape=(steps,),
                                        dtype='float32',
                                        chunks=(chunk_size,),
                                        compression='gzip')
            
            dset_act_I = f.create_dataset('activity_I',
                                        shape=(steps,),
                                        dtype='float32',
                                        chunks=(chunk_size,),
                                        compression='gzip')
            
            # Network statistics (recorded less frequently)
            n_stat_records = steps // record_interval + 1
            dset_conn_frac = f.create_dataset('connection_fraction',
                                            shape=(n_stat_records,),
                                            dtype='float32')
            
            dset_mean_weight = f.create_dataset('mean_weight',
                                              shape=(n_stat_records,),
                                              dtype='float32')
            
            # Save metadata
            f.attrs['N_E'] = self.N_E
            f.attrs['N_I'] = self.N_I
            f.attrs['n_timesteps'] = steps
            f.attrs['save_interval'] = save_interval
            f.attrs['record_interval'] = record_interval
            
            # Temporary buffers for batch writing
            buffer_x = np.zeros((save_interval, self.N_E), dtype=bool)
            buffer_y = np.zeros((save_interval, self.N_I), dtype=bool)
            buffer_act_E = np.zeros(save_interval, dtype=np.float32)
            buffer_act_I = np.zeros(save_interval, dtype=np.float32)
            
            # Summary statistics to return
            summary = {
                'final_activity_E': 0,
                'final_activity_I': 0,
                'final_mean_threshold': 0,
                'final_connection_fraction': 0,
                'final_mean_weight': 0,
                'activity_E_history': [],  # Keep a downsampled version in memory
                'connection_fraction_history': []
            }
            
            # Run simulation
            buffer_idx = 0
            stat_idx = 0
            threshold_idx = 0
            
            for t in range(steps):
                # Get input for this timestep
                if input_pattern is not None and self.N_U > 0:
                    u_t = input_pattern[t] if t < len(input_pattern) else np.zeros(self.N_U)
                else:
                    u_t = None
                
                # Update network
                self.step(u_t)
                
                # Store in buffer
                buffer_x[buffer_idx] = self.x
                buffer_y[buffer_idx] = self.y
                buffer_act_E[buffer_idx] = np.mean(self.x)
                buffer_act_I[buffer_idx] = np.mean(self.y)
                buffer_idx += 1
                
                # Save buffer to disk when full
                if buffer_idx >= save_interval or t == steps - 1:
                    # Calculate indices for this batch
                    start_idx = t - buffer_idx + 1
                    end_idx = t + 1
                    
                    # Write to HDF5
                    dset_x[start_idx:end_idx] = buffer_x[:buffer_idx]
                    dset_y[start_idx:end_idx] = buffer_y[:buffer_idx]
                    dset_act_E[start_idx:end_idx] = buffer_act_E[:buffer_idx]
                    dset_act_I[start_idx:end_idx] = buffer_act_I[:buffer_idx]
                    
                    # Reset buffer
                    buffer_idx = 0
                
                # Save thresholds less frequently
                if t % threshold_save_interval == 0:
                    dset_T_E[threshold_idx] = self.T_E
                    threshold_idx += 1
                
                # Record network statistics
                if t % record_interval == 0:
                    conn_frac = self.get_connection_fraction()
                    mean_weight = self.get_mean_weight()
                    dset_conn_frac[stat_idx] = conn_frac
                    dset_mean_weight[stat_idx] = mean_weight
                    stat_idx += 1
                    
                    # Keep some statistics in memory for plotting
                    summary['connection_fraction_history'].append(conn_frac)
                
                # Keep downsampled activity in memory (every 1000 steps)
                if t % 1000 == 0:
                    summary['activity_E_history'].append(np.mean(self.x))
                
                # Progress report
                if t % 100000 == 0:
                    print(f"Step {t}/{steps}, Activity: {np.mean(self.x):.3f}, "
                          f"Conn. Frac: {self.get_connection_fraction():.3f}")
            
            # Update summary statistics
            summary['final_activity_E'] = np.mean(self.x)
            summary['final_activity_I'] = np.mean(self.y)
            summary['final_mean_threshold'] = self.T_E.mean()
            summary['final_connection_fraction'] = self.get_connection_fraction()
            summary['final_mean_weight'] = self.get_mean_weight()
            
            # Convert lists to arrays
            summary['activity_E_history'] = np.array(summary['activity_E_history'])
            summary['connection_fraction_history'] = np.array(summary['connection_fraction_history'])
        
        print(f"\nSpiking data saved incrementally to {filename}")
        return summary


# Batch simulation functions
def run_single_simulation(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single simulation with given parameters
    
    Parameters:
    -----------
    sim_params : dict
        Dictionary containing simulation parameters including:
        - run_id: Unique identifier for this run
        - seed: Random seed for reproducibility
        - sorn_params: Parameters for SORN initialization
        - sim_steps: Number of simulation steps
        - output_dir: Directory to save results
        - save_full_data: Whether to save full spike data
        
    Returns:
    --------
    results : dict
        Summary results from the simulation
    """
    # Extract parameters
    run_id = sim_params['run_id']
    seed = sim_params['seed']
    sorn_params = sim_params['sorn_params']
    sim_steps = sim_params.get('sim_steps', 300000)
    output_dir = sim_params.get('output_dir', 'batch_results')
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    run_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\nStarting run {run_id} with seed {seed}")
    print(f"Parameters: h_ip={sorn_params.get('h_ip', 0.1)}, "
          f"noise_sig={sorn_params.get('noise_sig', np.sqrt(0.05))}")
    
    # Create SORN network
    sorn = SORN(**sorn_params)
    
    # Run simulation - save h5 file
    filename = os.path.join(run_dir, "spiking_data.h5")
    
    summary = sorn.simulate_incremental(
        sim_steps,
        save_interval=10000,
        record_interval=1000,
        filename=filename
    )
    
    print(f"\nSimulation complete!")
    print(f"Final mean excitatory activity: {summary['final_activity_E']:.3f}")
    print(f"Final mean inhibitory activity: {summary['final_activity_I']:.3f}")
    print(f"Final mean threshold: {summary['final_mean_threshold']:.3f}")
    print(f"Final connection fraction: {summary['final_connection_fraction']:.3f}")
    print(f"Final mean weight: {summary['final_mean_weight']:.3f}")
    
    # Create plots from HDF5 file
    print("\nCreating plots from HDF5 file...")
    
    # Create coarse-grained versions for easier viewing
    print("\nCreating coarse-grained raster plots...")
    
    # Coarse-grained full simulation
    plot_raster_coarsegrained_from_hdf5(
        filename,
        save_path=os.path.join(run_dir, "raster_coarse_3M.png"),
        time_bin=1000,  # Bin every 1000 timesteps
        neuron_bin=1,    # Don't bin neurons
        max_width_inches=20.0,
        dpi=150
    )
    
    # Coarse-grained with activity
    plot_raster_with_activity_coarsegrained(
        filename,
        save_path=os.path.join(run_dir, "raster_activity_coarse_3M.png"),
        time_bin=100,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    # Also create zoomed-in versions for easier viewing
    print("\nCreating zoomed raster plots...")
    
    # Last 100k timesteps - coarse-grained
    plot_raster_with_activity_coarsegrained(
        filename,
        save_path=os.path.join(run_dir, "raster_zoom_last100k_coarse.png"),
        start_time=sim_steps-100000,
        duration=100000,
        time_bin=100,  # Finer binning for zoomed view
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    # Network statistics
    plot_network_statistics_from_hdf5(
        filename, 
        save_path=os.path.join(run_dir, "network_stats.png")
    )
    
    # Create connection fraction plot (like Fig 1A)
    print("Creating connection fraction plot...")
    with h5py.File(filename, 'r') as f:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Colors from Figure 1
        c_stable = '#2E4172'
        c_notstable = '#7887AB'
        
        # Get connection fraction data
        conn_frac = f['connection_fraction'][:]
        record_interval = f.attrs.get('record_interval', 1000)
        x_vals = np.arange(len(conn_frac)) * record_interval
        
        # Identify phases (simplified - assuming first half is growth/decay)
        phase1_end = len(conn_frac) // 2
        
        # Plot connection fraction
        ax.plot(x_vals[:phase1_end], conn_frac[:phase1_end] * 100, 
                color=c_notstable, linewidth=1.5, label='Growth/Decay')
        ax.plot(x_vals[phase1_end:], conn_frac[phase1_end:] * 100, 
                color=c_stable, linewidth=1.5, label='Stable')
        
        # Annotations
        ax.text(x_vals[phase1_end//2], conn_frac[phase1_end//2] * 100 + 0.5, 
                'growth/decay', fontsize=12, color=c_notstable)
        ax.text(x_vals[phase1_end] + x_vals[-1] * 0.1, conn_frac[phase1_end] * 100 + 0.5, 
                'stable', fontsize=12, color=c_stable)
        
        # Formatting
        ax.set_xlim([0, x_vals[-1]])
        ax.set_xlabel(r'$10^6$ time steps', fontsize=12)
        ax.set_ylabel('Active Connections (%)', fontsize=12)
        ax.set_title(f'Run {run_id}: Connection Fraction Evolution (h_ip={sorn_params.get("h_ip", 0.1)})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis to show in millions
        n_millions = int(x_vals[-1] / 1e6)
        ax.set_xticks(np.arange(0, n_millions + 1) * 1e6)
        ax.set_xticklabels([str(i) for i in range(n_millions + 1)])
        
        # Remove top and right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'connection_fraction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to {run_dir}!")
    
    # Save summary results
    results = {
        'run_id': run_id,
        'seed': seed,
        'parameters': sorn_params,
        'sim_steps': sim_steps,
        'final_activity_E': float(summary['final_activity_E']),
        'final_activity_I': float(summary['final_activity_I']),
        'final_mean_threshold': float(summary['final_mean_threshold']),
        'final_connection_fraction': float(summary['final_connection_fraction']),
        'final_mean_weight': float(summary['final_mean_weight']),
        'mean_activity_E': float(np.mean(summary['activity_E_history'])),
        'std_activity_E': float(np.std(summary['activity_E_history'])),
        'mean_connection_fraction': float(np.mean(summary['connection_fraction_history'])),
        'std_connection_fraction': float(np.std(summary['connection_fraction_history']))
    }
    
    # Save results to JSON
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Completed run {run_id}")
    return results


def run_batch_simulations(n_runs: int = 3,
                         base_params: Optional[Dict[str, Any]] = None,
                         sim_steps: int = 300000,
                         output_dir: str = "batch_results",
                         n_processes: Optional[int] = None,
                         save_full_data: bool = True,
                         base_seed: int = 69) -> List[Dict[str, Any]]:
    """
    Run multiple identical simulations with different random seeds
    
    Parameters:
    -----------
    n_runs : int
        Number of simulations to run
    base_params : dict, optional
        Base parameters for SORN (if None, uses defaults)
    sim_steps : int
        Number of simulation steps
    output_dir : str
        Directory to save results
    n_processes : int, optional
        Number of parallel processes (if None, uses CPU count)
    save_full_data : bool
        Whether to save full spike data for each run
    base_seed : int
        Base random seed (each run gets base_seed + run_id)
    
    Returns:
    --------
    results : list
        List of summary dictionaries from all runs
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Default parameters
    if base_params is None:
        base_params = {
            'N_E': 200,
            'N_I': 40,
            'h_ip': 0.1,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        }
    
    # Prepare simulation parameters for each run
    sim_params_list = []
    for i in range(n_runs):
        sim_params = {
            'run_id': i,
            'seed': base_seed + i,
            'sorn_params': base_params.copy(),
            'sim_steps': sim_steps,
            'output_dir': full_output_dir,
            'save_full_data': save_full_data
        }
        sim_params_list.append(sim_params)
    
    # Save batch configuration
    batch_config = {
        'n_runs': n_runs,
        'base_params': base_params,
        'sim_steps': sim_steps,
        'timestamp': timestamp,
        'base_seed': base_seed
    }
    with open(os.path.join(full_output_dir, 'batch_config.json'), 'w') as f:
        json.dump(batch_config, f, indent=2)
    
    # Run simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"Running {n_runs} simulations using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, sim_params_list)
    
    # Save combined results
    with open(os.path.join(full_output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plots
    plot_batch_summary(results, full_output_dir)
    
    print(f"\nBatch simulation complete. Results saved to {full_output_dir}")
    return results


def run_parameter_sweep(param_ranges: Dict[str, List[Any]],
                       base_params: Optional[Dict[str, Any]] = None,
                       sim_steps: int = 300000,
                       output_dir: str = "sweep_results",
                       n_processes: Optional[int] = None,
                       n_seeds: int = 3,
                       save_full_data: bool = True,
                       base_seed: int = 42) -> List[Dict[str, Any]]:
    """
    Run parameter sweep over specified parameter ranges
    
    Parameters:
    -----------
    param_ranges : dict
        Dictionary mapping parameter names to lists of values to test
        Example: {'h_ip': [0.05, 0.1, 0.15], 'noise_sig': [0.01, 0.1, 1.0]}
    base_params : dict, optional
        Base parameters for SORN (swept parameters will override these)
    sim_steps : int
        Number of simulation steps
    output_dir : str
        Directory to save results
    n_processes : int, optional
        Number of parallel processes
    n_seeds : int
        Number of random seeds to test for each parameter combination
    save_full_data : bool
        Whether to save full spike data
    base_seed : int
        Base random seed
    
    Returns:
    --------
    results : list
        List of summary dictionaries from all runs
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"sweep_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Default base parameters
    if base_params is None:
        base_params = {
            'N_E': 200,
            'N_I': 40,
            'h_ip': 0.1,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        }
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Create all combinations
    import itertools
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Parameter sweep:")
    for name, values in param_ranges.items():
        print(f"  {name}: {values}")
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Seeds per combination: {n_seeds}")
    print(f"Total runs: {len(all_combinations) * n_seeds}")
    
    # Prepare simulation parameters
    sim_params_list = []
    run_id = 0
    
    for combo in all_combinations:
        # Create parameter dict for this combination
        combo_params = base_params.copy()
        for name, value in zip(param_names, combo):
            combo_params[name] = value
        
        # Create parameter folder name
        param_folder = '_'.join([f'{name}_{value}' for name, value in zip(param_names, combo)])
        param_dir = os.path.join(full_output_dir, param_folder)
        
        # Run multiple seeds for this combination
        for seed_idx in range(n_seeds):
            # Each seed gets its own subfolder within the parameter folder
            run_name = f"seed_{seed_idx}"
            
            sim_params = {
                'run_id': run_name,
                'seed': base_seed + run_id,
                'sorn_params': combo_params.copy(),
                'sim_steps': sim_steps,
                'output_dir': param_dir,  # Use parameter-specific directory
                'save_full_data': save_full_data
            }
            sim_params_list.append(sim_params)
            run_id += 1
    
    # Save sweep configuration
    sweep_config = {
        'param_ranges': param_ranges,
        'base_params': base_params,
        'sim_steps': sim_steps,
        'n_seeds': n_seeds,
        'total_runs': len(sim_params_list),
        'timestamp': timestamp,
        'base_seed': base_seed
    }
    with open(os.path.join(full_output_dir, 'sweep_config.json'), 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run simulations
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"\nRunning parameter sweep using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(run_single_simulation, sim_params_list)
    
    # Save combined results
    with open(os.path.join(full_output_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create parameter sweep plots
    plot_parameter_sweep_results(results, param_ranges, full_output_dir, n_seeds)
    
    print(f"\nParameter sweep complete. Results saved to {full_output_dir}")
    return results



def plot_batch_summary(results: List[Dict[str, Any]], output_dir: str):
    """Create summary plots for batch simulations"""
    
    # Extract data
    final_activities = [r['final_activity_E'] for r in results]
    final_conn_fracs = [r['final_connection_fraction'] for r in results]
    mean_activities = [r['mean_activity_E'] for r in results]
    mean_conn_fracs = [r['mean_connection_fraction'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Final activity distribution
    ax = axes[0, 0]
    ax.hist(final_activities, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_activities), color='red', linestyle='--', 
              label=f'Mean: {np.mean(final_activities):.3f}')
    ax.set_xlabel('Final Activity (E)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Activities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final connection fraction distribution
    ax = axes[0, 1]
    ax.hist(final_conn_fracs, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_conn_fracs), color='red', linestyle='--',
              label=f'Mean: {np.mean(final_conn_fracs):.3f}')
    ax.set_xlabel('Final Connection Fraction')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Connection Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean activity vs connection fraction
    ax = axes[1, 0]
    ax.scatter(mean_activities, mean_conn_fracs, alpha=0.6)
    ax.set_xlabel('Mean Activity (E)')
    ax.set_ylabel('Mean Connection Fraction')
    ax.set_title('Activity vs Connection Fraction')
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Batch Summary (n={len(results)}):
    
    Final Activity:
      Mean: {np.mean(final_activities):.4f} ± {np.std(final_activities):.4f}
      Range: [{np.min(final_activities):.4f}, {np.max(final_activities):.4f}]
    
    Final Connection Fraction:
      Mean: {np.mean(final_conn_fracs):.4f} ± {np.std(final_conn_fracs):.4f}
      Range: [{np.min(final_conn_fracs):.4f}, {np.max(final_conn_fracs):.4f}]
    
    Mean Activity (over time):
      Mean: {np.mean(mean_activities):.4f} ± {np.std(mean_activities):.4f}
    
    Mean Connection Fraction (over time):
      Mean: {np.mean(mean_conn_fracs):.4f} ± {np.std(mean_conn_fracs):.4f}
    """
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_sweep_results(results: List[Dict[str, Any]], 
                                param_ranges: Dict[str, List[Any]],
                                output_dir: str,
                                n_seeds: int):
    """Create plots for parameter sweep results"""
    
    # Organize results by parameter values
    param_names = list(param_ranges.keys())
    
    if len(param_names) == 1:
        # 1D parameter sweep
        plot_1d_parameter_sweep(results, param_names[0], param_ranges[param_names[0]], 
                               output_dir, n_seeds)
    elif len(param_names) == 2:
        # 2D parameter sweep
        plot_2d_parameter_sweep(results, param_names, param_ranges, output_dir, n_seeds)
    else:
        print(f"Plotting for {len(param_names)}D parameter sweeps not implemented. "
              f"Results saved to JSON.")


def plot_1d_parameter_sweep(results: List[Dict[str, Any]], 
                           param_name: str,
                           param_values: List[Any],
                           output_dir: str,
                           n_seeds: int):
    """Plot results for 1D parameter sweep"""
    
    # Organize results by parameter value
    results_by_value = {v: [] for v in param_values}
    for r in results:
        value = r['parameters'][param_name]
        results_by_value[value].append(r)
    
    # Calculate statistics
    means_activity = []
    stds_activity = []
    means_conn_frac = []
    stds_conn_frac = []
    
    for v in param_values:
        activities = [r['final_activity_E'] for r in results_by_value[v]]
        conn_fracs = [r['final_connection_fraction'] for r in results_by_value[v]]
        
        means_activity.append(np.mean(activities))
        stds_activity.append(np.std(activities))
        means_conn_frac.append(np.mean(conn_fracs))
        stds_conn_frac.append(np.std(conn_fracs))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Activity plot
    ax1.errorbar(param_values, means_activity, yerr=stds_activity, 
                marker='o', capsize=5, capthick=2, markersize=8)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Final Activity (E)')
    ax1.set_title(f'Activity vs {param_name} (n={n_seeds} per point)')
    ax1.grid(True, alpha=0.3)
    if param_name == 'noise_sig' and min(param_values) > 0:
        ax1.set_xscale('log')
    
    # Connection fraction plot
    ax2.errorbar(param_values, means_conn_frac, yerr=stds_conn_frac,
                marker='o', capsize=5, capthick=2, markersize=8)
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Final Connection Fraction')
    ax2.set_title(f'Connection Fraction vs {param_name} (n={n_seeds} per point)')
    ax2.grid(True, alpha=0.3)
    if param_name == 'noise_sig' and min(param_values) > 0:
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_1d_{param_name}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_2d_parameter_sweep(results: List[Dict[str, Any]],
                           param_names: List[str],
                           param_ranges: Dict[str, List[Any]],
                           output_dir: str,
                           n_seeds: int):
    """Plot results for 2D parameter sweep"""
    
    param1, param2 = param_names
    values1 = param_ranges[param1]
    values2 = param_ranges[param2]
    
    # Create grids for results
    activity_grid = np.zeros((len(values2), len(values1)))
    conn_frac_grid = np.zeros((len(values2), len(values1)))
    
    # Fill grids
    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            # Find matching results
            matching = [r for r in results 
                       if r['parameters'][param1] == v1 and r['parameters'][param2] == v2]
            
            if matching:
                activity_grid[j, i] = np.mean([r['final_activity_E'] for r in matching])
                conn_frac_grid[j, i] = np.mean([r['final_connection_fraction'] for r in matching])
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Activity heatmap
    im1 = ax1.imshow(activity_grid, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xticks(range(len(values1)))
    ax1.set_xticklabels([f'{v:.3g}' for v in values1])
    ax1.set_yticks(range(len(values2)))
    ax1.set_yticklabels([f'{v:.3g}' for v in values2])
    ax1.set_xlabel(param1)
    ax1.set_ylabel(param2)
    ax1.set_title(f'Final Activity (n={n_seeds} per cell)')
    plt.colorbar(im1, ax=ax1)
    
    # Connection fraction heatmap
    im2 = ax2.imshow(conn_frac_grid, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xticks(range(len(values1)))
    ax2.set_xticklabels([f'{v:.3g}' for v in values1])
    ax2.set_yticks(range(len(values2)))
    ax2.set_yticklabels([f'{v:.3g}' for v in values2])
    ax2.set_xlabel(param1)
    ax2.set_ylabel(param2)
    ax2.set_title(f'Final Connection Fraction (n={n_seeds} per cell)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_2d_{param1}_{param2}.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_raster_with_activity_from_hdf5(filename: str,
                                       save_path: str = "raster_with_activity.pdf",
                                       start_time: Optional[int] = None,
                                       duration: Optional[int] = None,
                                       activity_height: float = 2.0):
    """
    Create a full-resolution raster plot with activity trace from HDF5
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the raster plot
    start_time : int, optional
        Starting timestep
    duration : int, optional
        Number of timesteps to plot
    activity_height : float
        Height of activity plot in inches
    """
    
    print("Creating full-resolution raster plot with activity from HDF5...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate figure dimensions
        width_inches = duration / 72.0
        raster_height = n_neurons / 72.0
        total_height = raster_height + activity_height + 0.5
        
        print(f"Creating figure: {width_inches:.1f} x {total_height:.1f} inches")
        
        # Create figure
        fig = plt.figure(figsize=(width_inches, total_height), dpi=72)
        
        # Calculate height ratios
        height_ratio_raster = raster_height / total_height
        height_ratio_activity = activity_height / total_height
        height_ratio_gap = 0.5 / total_height
        
        # Create axes
        ax_raster = plt.axes([0, height_ratio_activity + height_ratio_gap, 
                             1, height_ratio_raster])
        ax_activity = plt.axes([0, 0, 1, height_ratio_activity])
        
        # Plot raster
        print("Plotting raster...")
        chunk_size = 100000
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                ax_raster.vlines(spike_times + chunk_start, 
                               neuron_ids - 0.5, 
                               neuron_ids + 0.5,
                               colors='black',
                               linewidth=1.0,
                               antialiased=False)
            
            if chunk_start % 500000 == 0:
                print(f"  Processed {chunk_start:,} / {duration:,} timesteps")
        
        # Set raster limits
        ax_raster.set_xlim(0, duration)
        ax_raster.set_ylim(-0.5, n_neurons - 0.5)
        ax_raster.axis('off')
        
        # Plot activity
        print("Plotting activity...")
        activity_data = f['activity_E'][start_time:start_time + duration]
        ax_activity.fill_between(np.arange(duration), 
                               0, 
                               activity_data,
                               color='black',
                               alpha=0.7,
                               linewidth=0)
        
        # Add target activity line
        ax_activity.axhline(y=0.1, color='red', linewidth=1, alpha=0.8)
        
        # Set activity limits
        ax_activity.set_xlim(0, duration)
        ax_activity.set_ylim(0, max(0.2, activity_data.max() * 1.1))
        
        # Style activity plot
        ax_activity.spines['top'].set_visible(False)
        ax_activity.spines['right'].set_visible(False)
        ax_activity.set_xlabel('Time Steps', fontsize=10)
        ax_activity.set_ylabel('Mean Activity', fontsize=10)
        
        # Add tick marks
        if duration > 1000000:
            tick_positions = np.arange(0, duration + 1, 1000000)
            ax_activity.set_xticks(tick_positions)
            ax_activity.set_xticklabels([f'{t/1e6:.0f}M' for t in tick_positions], fontsize=8)
        
       # Save as PDF
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=72)
        plt.close()
        
        print(f"Full-resolution raster plot with activity saved to {save_path}")

def plot_raster_with_activity_coarsegrained(filename: str,
                                          save_path: str = "raster_activity_coarse.png",
                                          start_time: Optional[int] = None,
                                          duration: Optional[int] = None,
                                          time_bin: int = 100,
                                          neuron_bin: int = 1,
                                          max_width_inches: float = 20.0,
                                          dpi: int = 150):
    """
    Create a coarse-grained raster plot with activity trace
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the plot
    start_time : int, optional
        Starting timestep
    duration : int, optional
        Number of timesteps to plot
    time_bin : int
        Number of timesteps to bin together
    neuron_bin : int
        Number of neurons to bin together
    max_width_inches : float
        Maximum figure width in inches
    dpi : int
        DPI for the output image
    """
    
    print(f"Creating coarse-grained raster with activity plot...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins = n_neurons // neuron_bin
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_neuron_bins, n_time_bins), dtype=bool)
        
        # Process raster data
        chunk_size = 100000
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                spike_times += chunk_start
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
        
        # Load and bin activity data
        activity_data = f['activity_E'][start_time:start_time + duration]
        # Reshape and average over time bins
        n_complete_bins = (len(activity_data) // time_bin) * time_bin
        activity_binned = activity_data[:n_complete_bins].reshape(-1, time_bin).mean(axis=1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(max_width_inches, 10), dpi=dpi)
        
        # Create grid spec for better control
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
        
        ax_raster = fig.add_subplot(gs[0])
        ax_activity = fig.add_subplot(gs[1], sharex=ax_raster)
        
        # Plot raster
        im = ax_raster.imshow(raster_matrix, 
                             aspect='auto', 
                             cmap='binary_r',
                             interpolation='nearest',
                             origin='lower')
        
        ax_raster.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + ")" if neuron_bin > 1 else ""}', 
                            fontsize=12)
        ax_raster.set_title(f'Raster Plot and Activity ({n_time_bins} time bins)', fontsize=14)
        
        # Plot activity
        time_axis = np.arange(len(activity_binned))
        ax_activity.fill_between(time_axis, 0, activity_binned, 
                               color='darkblue', alpha=0.7, linewidth=0)
        ax_activity.axhline(y=0.1, color='red', linewidth=2, alpha=0.8, 
                           linestyle='--', label='Target (h_ip=0.1)')
        
        ax_activity.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax_activity.set_ylabel('Mean Activity', fontsize=12)
        ax_activity.set_ylim(0, max(0.2, activity_binned.max() * 1.1))
        ax_activity.legend(loc='upper right')
        ax_activity.grid(True, alpha=0.3)
        
        # Set x-axis labels
        n_time_ticks = min(10, n_time_bins)
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        ax_activity.set_xticks(time_tick_positions)
        ax_activity.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                                   for t in time_tick_positions], fontsize=10)
        
        # Style
        ax_raster.grid(True, alpha=0.2, linewidth=0.5)
        
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Coarse-grained raster with activity saved to {save_path}")


def plot_network_statistics_from_hdf5(filename: str, save_path: str = "network_stats.png"):
    """
    Plot network statistics from HDF5 file
    """
    with h5py.File(filename, 'r') as f:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Connection fraction
        ax = axes[0, 0]
        conn_frac = f['connection_fraction'][:]
        record_interval = f.attrs.get('record_interval', 1000)
        steps = np.arange(len(conn_frac)) * record_interval
        ax.plot(steps, conn_frac * 100, 'b-', linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Active Connections (%)')
        ax.set_title('Connection Fraction Evolution')
        ax.grid(True, alpha=0.3)
        
        # Mean weight
        ax = axes[0, 1]
        mean_weight = f['mean_weight'][:]
        ax.plot(steps, mean_weight, 'g-', linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Mean Weight Evolution')
        ax.grid(True, alpha=0.3)
        
        # Activity over time (smoothed)
        ax = axes[1, 0]
        n_timesteps = f.attrs['n_timesteps']
        window = 1000
        chunk_size = 100000
        smoothed_activity = []
        
        for i in range(0, n_timesteps - window, chunk_size):
            end_idx = min(i + chunk_size + window, n_timesteps)
            chunk = f['activity_E'][i:end_idx]
            smoothed = np.convolve(chunk, np.ones(window)/window, mode='valid')
            smoothed_activity.extend(smoothed[:chunk_size] if i + chunk_size < n_timesteps 
                                   else smoothed)
        
        smoothed_activity = np.array(smoothed_activity)
        ax.plot(smoothed_activity, 'r-', alpha=0.7, linewidth=1)
        ax.axhline(y=0.1, color='k', linestyle='--', label='Target (h_ip=0.1)')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Mean Activity')
        ax.set_title('Excitatory Activity (smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Threshold distribution
        ax = axes[1, 1]
        T_E_first = f['thresholds_E'][0]
        T_E_last = f['thresholds_E'][-1]
        ax.hist(T_E_first, bins=30, alpha=0.5, label='Initial', color='blue')
        ax.hist(T_E_last, bins=30, alpha=0.5, label='Final', color='red')
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Count')
        ax.set_title('Threshold Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Network statistics saved to {save_path}")

#!/usr/bin/env python3
"""
Example: Parameter sweep for h_ip values [0.01, 0.1] with eigenvalue analysis
Runs 3 seeds per parameter value and analyzes eigenvalue spectrum at 2M timesteps
"""

import numpy as np
import time
import os
import json
from datetime import datetime

# Import all the necessary functions from your main code
# (In practice, these would be imported from your main module)

if __name__ == "__main__":
    print("="*80)
    print("SORN Parameter Sweep: h_ip = [0.01, 0.1] with Eigenvalue Analysis")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define parameter sweep configuration
    param_ranges = {
        'h_ip': [0.01, 0.1]  # Target firing rates to compare
    }
    
    # Base parameters (all other parameters stay constant)
    base_params = {
        'N_E': 200,
        'N_I': 40,
        'eta_stdp': 0.004,
        'eta_istdp': 0.001,
        'eta_ip': 0.01,
        'lambda_': 20,
        'noise_sig': np.sqrt(0.05),
        'T_e_min': 0.5,
        'T_e_max': 1.0,
        'T_i_min': 0.5,
        'T_i_max': 1.0,
        'W_ee_initial': 0.001,
        'W_min': 0.0,
        'W_max': 1.0,
        'W_ei_min': 0.001
    }
    
    # Simulation configuration
    sim_steps = 3000000  # 3M timesteps total
    eigenvalue_checkpoint = 2000000  # Analyze eigenvalues at 2M timesteps
    n_seeds = 3  # 3 different random seeds per parameter value
    n_processes = None  # Use all available CPU cores
    
    # Use timestamp-based seed for reproducibility
    base_seed = int(time.time()) % 1000000
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"h_ip_sweep_eigenvalues_{timestamp}"
    
    print(f"Configuration:")
    print(f"  - h_ip values: {param_ranges['h_ip']}")
    print(f"  - Seeds per value: {n_seeds}")
    print(f"  - Total runs: {len(param_ranges['h_ip']) * n_seeds}")
    print(f"  - Simulation steps: {sim_steps:,} ({sim_steps/1e6:.1f}M)")
    print(f"  - Eigenvalue checkpoint: {eigenvalue_checkpoint:,} ({eigenvalue_checkpoint/1e6:.1f}M)")
    print(f"  - Base random seed: {base_seed}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Parallel processes: {'all available' if n_processes is None else n_processes}")
    print()
    
    # Run the parameter sweep with eigenvalue analysis
    print("Starting parameter sweep...")
    print("-"*60)
    
    try:
        sweep_results = run_parameter_sweep_with_eigenvalues(
            param_ranges=param_ranges,
            base_params=base_params,
            sim_steps=sim_steps,
            eigenvalue_checkpoint=eigenvalue_checkpoint,
            output_dir=output_dir,
            n_processes=n_processes,
            n_seeds=n_seeds,
            base_seed=base_seed
        )
        
        print("\n" + "="*80)
        print("SWEEP COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Analyze and summarize results
        print("\nRESULTS SUMMARY:")
        print("-"*60)
        
        # Group results by h_ip value
        results_by_hip = {0.01: [], 0.1: []}
        eigenvalue_results_by_hip = {0.01: [], 0.1: []}
        
        for result in sweep_results:
            h_ip_value = result['parameters']['h_ip']
            results_by_hip[h_ip_value].append(result)
            
            if 'eigenvalue_analysis' in result:
                eigenvalue_results_by_hip[h_ip_value].append(result['eigenvalue_analysis'])
        
        # Print detailed summary for each h_ip value
        for h_ip in [0.01, 0.1]:
            print(f"\n>>> h_ip = {h_ip} <<<")
            print("-"*40)
            
            runs = results_by_hip[h_ip]
            eigen_analyses = eigenvalue_results_by_hip[h_ip]
            
            if runs:
                # Activity statistics
                final_activities = [r['final_activity_E'] for r in runs]
                print(f"\nFinal Activity (at 3M steps):")
                print(f"  Mean: {np.mean(final_activities):.4f} ± {np.std(final_activities):.4f}")
                print(f"  Range: [{np.min(final_activities):.4f}, {np.max(final_activities):.4f}]")
                
                # Connection fraction statistics
                final_conn_fracs = [r['final_connection_fraction'] for r in runs]
                print(f"\nFinal Connection Fraction:")
                print(f"  Mean: {np.mean(final_conn_fracs):.4f} ± {np.std(final_conn_fracs):.4f}")
                print(f"  Range: [{np.min(final_conn_fracs):.4f}, {np.max(final_conn_fracs):.4f}]")
                
                # Eigenvalue analysis (at 2M steps)
                if eigen_analyses:
                    print(f"\nEigenvalue Analysis (at 2M steps):")
                    
                    spectral_radii = [ea['spectral_radius'] for ea in eigen_analyses]
                    print(f"  Spectral Radius:")
                    print(f"    Mean: {np.mean(spectral_radii):.4f} ± {np.std(spectral_radii):.4f}")
                    print(f"    Range: [{np.min(spectral_radii):.4f}, {np.max(spectral_radii):.4f}]")
                    
                    n_unstable_list = [ea['n_unstable'] for ea in eigen_analyses]
                    print(f"  Unstable Eigenvalues (|λ| > 1):")
                    print(f"    Mean: {np.mean(n_unstable_list):.1f} ± {np.std(n_unstable_list):.1f}")
                    print(f"    Range: [{np.min(n_unstable_list)}, {np.max(n_unstable_list)}]")
                    
                    activities_at_checkpoint = [ea['activity_at_checkpoint'] for ea in eigen_analyses]
                    print(f"  Activity at 2M steps:")
                    print(f"    Mean: {np.mean(activities_at_checkpoint):.4f} ± {np.std(activities_at_checkpoint):.4f}")
                    
                    # Stability assessment
                    all_stable = all(sr <= 1.0 for sr in spectral_radii)
                    print(f"  Network Stability: {'ALL STABLE' if all_stable else 'SOME UNSTABLE'}")
                    
                    # Individual run details
                    print(f"\n  Individual Runs:")
                    for i, (ea, r) in enumerate(zip(eigen_analyses, runs)):
                        stability = 'STABLE' if ea['spectral_radius'] <= 1.0 else 'UNSTABLE'
                        print(f"    Seed {i}: ρ={ea['spectral_radius']:.4f}, "
                              f"Activity={ea['activity_at_checkpoint']:.3f}, "
                              f"{stability}")
        
        # Comparison between h_ip values
        print("\n" + "="*60)
        print("COMPARISON: h_ip = 0.01 vs h_ip = 0.1")
        print("="*60)
        
        if eigenvalue_results_by_hip[0.01] and eigenvalue_results_by_hip[0.1]:
            # Compare spectral radii
            sr_001 = [ea['spectral_radius'] for ea in eigenvalue_results_by_hip[0.01]]
            sr_01 = [ea['spectral_radius'] for ea in eigenvalue_results_by_hip[0.1]]
            
            print(f"\nSpectral Radius:")
            print(f"  h_ip=0.01: {np.mean(sr_001):.4f} ± {np.std(sr_001):.4f}")
            print(f"  h_ip=0.1:  {np.mean(sr_01):.4f} ± {np.std(sr_01):.4f}")
            print(f"  Difference: {np.mean(sr_01) - np.mean(sr_001):+.4f}")
            
            # Compare activities
            act_001 = [ea['activity_at_checkpoint'] for ea in eigenvalue_results_by_hip[0.01]]
            act_01 = [ea['activity_at_checkpoint'] for ea in eigenvalue_results_by_hip[0.1]]
            
            print(f"\nActivity at 2M steps:")
            print(f"  h_ip=0.01: {np.mean(act_001):.4f} ± {np.std(act_001):.4f}")
            print(f"  h_ip=0.1:  {np.mean(act_01):.4f} ± {np.std(act_01):.4f}")
            print(f"  Difference: {np.mean(act_01) - np.mean(act_001):+.4f}")
            
            # Stability comparison
            stable_001 = sum(1 for sr in sr_001 if sr <= 1.0)
            stable_01 = sum(1 for sr in sr_01 if sr <= 1.0)
            
            print(f"\nNetwork Stability:")
            print(f"  h_ip=0.01: {stable_001}/{len(sr_001)} networks stable")
            print(f"  h_ip=0.1:  {stable_01}/{len(sr_01)} networks stable")
        
        # Output locations
        print("\n" + "="*60)
        print("OUTPUT FILES:")
        print("="*60)
        print(f"\nMain output directory: {output_dir}/")
        print("\nKey files:")
        print(f"  - Sweep configuration: {output_dir}/sweep_config.json")
        print(f"  - All results: {output_dir}/all_results.json")
        print(f"  - Eigenvalue comparison plot: {output_dir}/eigenvalue_comparison.png")
        
        print("\nPer-run directories:")
        for h_ip in [0.01, 0.1]:
            print(f"\n  h_ip_{h_ip}/")
            for i in range(n_seeds):
                print(f"    └── seed_{i}/")
                print(f"        ├── spiking_data.h5")
                print(f"        ├── eigenvalue_spectrum_2M.png")
                print(f"        ├── raster_coarse_3M.png")
                print(f"        ├── raster_activity_coarse_3M.png")
                print(f"        ├── network_stats.png")
                print(f"        ├── connection_fraction.png")
                print(f"        └── summary.json")
        
        # Final notes
        print("\n" + "="*60)
        print("NOTES:")
        print("="*60)
        print("1. Eigenvalue analysis was performed at 2M timesteps (stabilized network)")
        print("2. Spectral radius > 1 indicates potential instability")
        print("3. Each h_ip value was tested with 3 different random seeds")
        print("4. Full spike data (3M timesteps) saved in HDF5 format")
        print("5. All plots include eigenvalue checkpoint markers where applicable")
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nParameter sweep completed successfully!")
        
    except Exception as e:
        print(f"\nERROR: Parameter sweep failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*80)



def plot_raster_coarsegrained_from_hdf5(filename: str, 
                                       save_path: str = "raster_coarse.png",
                                       start_time: Optional[int] = None,
                                       duration: Optional[int] = None,
                                       time_bin: int = 100,
                                       neuron_bin: int = 1,
                                       max_width_inches: float = 20.0,
                                       max_height_inches: float = 10.0,
                                       dpi: int = 150):
    """
    Create a coarse-grained raster plot from HDF5 file
    Each spike is represented as a single pixel, no overlaps
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the raster plot
    start_time : int, optional
        Starting timestep (if None, start from beginning)
    duration : int, optional
        Number of timesteps to plot (if None, plot all)
    time_bin : int
        Number of timesteps to bin together (default: 100)
    neuron_bin : int
        Number of neurons to bin together (default: 1)
    max_width_inches : float
        Maximum figure width in inches
    max_height_inches : float
        Maximum figure height in inches
    dpi : int
        DPI for the output image
    """
    
    print(f"Creating coarse-grained raster plot (time_bin={time_bin}, neuron_bin={neuron_bin})...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins = n_neurons // neuron_bin
        
        print(f"Original: {duration} timesteps x {n_neurons} neurons")
        print(f"Binned: {n_time_bins} time bins x {n_neuron_bins} neuron bins")
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_neuron_bins, n_time_bins), dtype=bool)
        
        # Process data in chunks for memory efficiency
        chunk_size = 100000
        print("Processing spike data...")
        
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            
            # Load chunk from HDF5
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            
            # Find spikes in this chunk
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                # Adjust spike times to be relative to chunk
                spike_times += chunk_start
                
                # Bin the spikes
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                
                # Keep only spikes within our binned dimensions
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                
                # Mark bins with spikes (using OR to avoid overlaps)
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
            
            if chunk_start % 500000 == 0:
                print(f"  Processed {chunk_start:,} / {duration:,} timesteps")
        
        # Calculate figure size based on matrix dimensions and constraints
        aspect_ratio = n_time_bins / n_neuron_bins
        
        if aspect_ratio > max_width_inches / max_height_inches:
            # Width-limited
            width = max_width_inches
            height = width / aspect_ratio
        else:
            # Height-limited
            height = max_height_inches
            width = height * aspect_ratio
        
        print(f"Creating figure: {width:.1f} x {height:.1f} inches at {dpi} DPI")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        
        # Plot raster using imshow for efficiency
        # Each spike is exactly one pixel
        ax.imshow(raster_matrix, 
                  aspect='auto', 
                  cmap='binary_r',  # White background, black spikes
                  interpolation='nearest',
                  origin='lower')
        
        # Set proper labels
        ax.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + " neurons)" if neuron_bin > 1 else ""}', 
                     fontsize=12)
        ax.set_title(f'Raster Plot ({n_time_bins} x {n_neuron_bins} bins)', fontsize=14)
        
        # Add tick labels showing actual time/neuron values
        n_time_ticks = min(10, n_time_bins)
        n_neuron_ticks = min(10, n_neuron_bins)
        
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        neuron_tick_positions = np.linspace(0, n_neuron_bins-1, n_neuron_ticks, dtype=int)
        
        ax.set_xticks(time_tick_positions)
        ax.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                           for t in time_tick_positions], fontsize=10)
        
        ax.set_yticks(neuron_tick_positions)
        ax.set_yticklabels([f'{t*neuron_bin}' for t in neuron_tick_positions], fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Save
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        spike_density = np.sum(raster_matrix) / (n_time_bins * n_neuron_bins) * 100
        print(f"Spike density in plot: {spike_density:.2f}%")
        print(f"Coarse-grained raster plot saved to {save_path}")


def plot_raster_with_activity_from_hdf5(filename: str,
                                       save_path: str = "raster_with_activity.pdf",
                                       start_time: Optional[int] = None,
                                       duration: Optional[int] = None,
                                       activity_height: float = 2.0):
    """
    Create a full-resolution raster plot with activity trace from HDF5
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the raster plot
    start_time : int, optional
        Starting timestep
    duration : int, optional
        Number of timesteps to plot
    activity_height : float
        Height of activity plot in inches
    """
    
    print("Creating full-resolution raster plot with activity from HDF5...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate figure dimensions
        width_inches = duration / 72.0
        raster_height = n_neurons / 72.0
        total_height = raster_height + activity_height + 0.5
        
        print(f"Creating figure: {width_inches:.1f} x {total_height:.1f} inches")
        
        # Create figure
        fig = plt.figure(figsize=(width_inches, total_height), dpi=72)
        
        # Calculate height ratios
        height_ratio_raster = raster_height / total_height
        height_ratio_activity = activity_height / total_height
        height_ratio_gap = 0.5 / total_height
        
        # Create axes
        ax_raster = plt.axes([0, height_ratio_activity + height_ratio_gap, 
                             1, height_ratio_raster])
        ax_activity = plt.axes([0, 0, 1, height_ratio_activity])
        
        # Plot raster
        print("Plotting raster...")
        chunk_size = 100000
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                ax_raster.vlines(spike_times + chunk_start, 
                               neuron_ids - 0.5, 
                               neuron_ids + 0.5,
                               colors='black',
                               linewidth=1.0,
                               antialiased=False)
            
            if chunk_start % 500000 == 0:
                print(f"  Processed {chunk_start:,} / {duration:,} timesteps")
        
        # Set raster limits
        ax_raster.set_xlim(0, duration)
        ax_raster.set_ylim(-0.5, n_neurons - 0.5)
        ax_raster.axis('off')
        
        # Plot activity
        print("Plotting activity...")
        activity_data = f['activity_E'][start_time:start_time + duration]
        ax_activity.fill_between(np.arange(duration), 
                               0, 
                               activity_data,
                               color='black',
                               alpha=0.7,
                               linewidth=0)
        
        # Add target activity line
        ax_activity.axhline(y=0.1, color='red', linewidth=1, alpha=0.8)
        
        # Set activity limits
        ax_activity.set_xlim(0, duration)
        ax_activity.set_ylim(0, max(0.2, activity_data.max() * 1.1))
        
        # Style activity plot
        ax_activity.spines['top'].set_visible(False)
        ax_activity.spines['right'].set_visible(False)
        ax_activity.set_xlabel('Time Steps', fontsize=10)
        ax_activity.set_ylabel('Mean Activity', fontsize=10)
        
        # Add tick marks
        if duration > 1000000:
            tick_positions = np.arange(0, duration + 1, 1000000)
            ax_activity.set_xticks(tick_positions)
            ax_activity.set_xticklabels([f'{t/1e6:.0f}M' for t in tick_positions], fontsize=8)
        
       # Save as PDF
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=72)
        plt.close()
        
        print(f"Full-resolution raster plot with activity saved to {save_path}")

def plot_raster_with_activity_coarsegrained(filename: str,
                                          save_path: str = "raster_activity_coarse.png",
                                          start_time: Optional[int] = None,
                                          duration: Optional[int] = None,
                                          time_bin: int = 100,
                                          neuron_bin: int = 1,
                                          max_width_inches: float = 20.0,
                                          dpi: int = 150):
    """
    Create a coarse-grained raster plot with activity trace
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the plot
    start_time : int, optional
        Starting timestep
    duration : int, optional
        Number of timesteps to plot
    time_bin : int
        Number of timesteps to bin together
    neuron_bin : int
        Number of neurons to bin together
    max_width_inches : float
        Maximum figure width in inches
    dpi : int
        DPI for the output image
    """
    
    print(f"Creating coarse-grained raster with activity plot...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins = n_neurons // neuron_bin
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_neuron_bins, n_time_bins), dtype=bool)
        
        # Process raster data
        chunk_size = 100000
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                spike_times += chunk_start
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
        
        # Load and bin activity data
        activity_data = f['activity_E'][start_time:start_time + duration]
        # Reshape and average over time bins
        n_complete_bins = (len(activity_data) // time_bin) * time_bin
        activity_binned = activity_data[:n_complete_bins].reshape(-1, time_bin).mean(axis=1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(max_width_inches, 10), dpi=dpi)
        
        # Create grid spec for better control
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
        
        ax_raster = fig.add_subplot(gs[0])
        ax_activity = fig.add_subplot(gs[1], sharex=ax_raster)
        
        # Plot raster
        im = ax_raster.imshow(raster_matrix, 
                             aspect='auto', 
                             cmap='binary_r',
                             interpolation='nearest',
                             origin='lower')
        
        ax_raster.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + ")" if neuron_bin > 1 else ""}', 
                            fontsize=12)
        ax_raster.set_title(f'Raster Plot and Activity ({n_time_bins} time bins)', fontsize=14)
        
        # Plot activity
        time_axis = np.arange(len(activity_binned))
        ax_activity.fill_between(time_axis, 0, activity_binned, 
                               color='darkblue', alpha=0.7, linewidth=0)
        ax_activity.axhline(y=0.1, color='red', linewidth=2, alpha=0.8, 
                           linestyle='--', label='Target (h_ip=0.1)')
        
        ax_activity.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax_activity.set_ylabel('Mean Activity', fontsize=12)
        ax_activity.set_ylim(0, max(0.2, activity_binned.max() * 1.1))
        ax_activity.legend(loc='upper right')
        ax_activity.grid(True, alpha=0.3)
        
        # Set x-axis labels
        n_time_ticks = min(10, n_time_bins)
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        ax_activity.set_xticks(time_tick_positions)
        ax_activity.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                                   for t in time_tick_positions], fontsize=10)
        
        # Style
        ax_raster.grid(True, alpha=0.2, linewidth=0.5)
        
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Coarse-grained raster with activity saved to {save_path}")


def plot_network_statistics_from_hdf5(filename: str, save_path: str = "network_stats.png"):
    """
    Plot network statistics from HDF5 file
    """
    with h5py.File(filename, 'r') as f:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Connection fraction
        ax = axes[0, 0]
        conn_frac = f['connection_fraction'][:]
        record_interval = f.attrs.get('record_interval', 1000)
        steps = np.arange(len(conn_frac)) * record_interval
        ax.plot(steps, conn_frac * 100, 'b-', linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Active Connections (%)')
        ax.set_title('Connection Fraction Evolution')
        ax.grid(True, alpha=0.3)
        
        # Mean weight
        ax = axes[0, 1]
        mean_weight = f['mean_weight'][:]
        ax.plot(steps, mean_weight, 'g-', linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Mean Weight Evolution')
        ax.grid(True, alpha=0.3)
        
        # Activity over time (smoothed)
        ax = axes[1, 0]
        n_timesteps = f.attrs['n_timesteps']
        window = 1000
        chunk_size = 100000
        smoothed_activity = []
        
        for i in range(0, n_timesteps - window, chunk_size):
            end_idx = min(i + chunk_size + window, n_timesteps)
            chunk = f['activity_E'][i:end_idx]
            smoothed = np.convolve(chunk, np.ones(window)/window, mode='valid')
            smoothed_activity.extend(smoothed[:chunk_size] if i + chunk_size < n_timesteps 
                                   else smoothed)
        
        smoothed_activity = np.array(smoothed_activity)
        ax.plot(smoothed_activity, 'r-', alpha=0.7, linewidth=1)
        ax.axhline(y=0.1, color='k', linestyle='--', label='Target (h_ip=0.1)')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Mean Activity')
        ax.set_title('Excitatory Activity (smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Threshold distribution
        ax = axes[1, 1]
        T_E_first = f['thresholds_E'][0]
        T_E_last = f['thresholds_E'][-1]
        ax.hist(T_E_first, bins=30, alpha=0.5, label='Initial', color='blue')
        ax.hist(T_E_last, bins=30, alpha=0.5, label='Final', color='red')
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Count')
        ax.set_title('Threshold Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Network statistics saved to {save_path}")

if __name__ == "__main__":
    import time  
    
    # Example 1: Run parameter sweep with YOUR specified h_ip values
    sweep_results = run_parameter_sweep(
        param_ranges={
            'h_ip': [0.01, 0.1]  # YOUR VALUES
        },
        base_params={
            'N_E': 200,
            'N_I': 40,
            'eta_stdp': 0.004,
            'eta_istdp': 0.001,
            'eta_ip': 0.01,
            'lambda_': 20,
            'noise_sig': np.sqrt(0.05)
        },
        sim_steps=300000,  # 300k steps as your default
        n_seeds=3,
        n_processes= None,  # Use all available cores
        save_full_data=True,
        base_seed=int(time.time()) % 1000000  # Different seed each run
    )
    
    # # Example 2: Run batch simulations with unique seeds
    # batch_results = run_batch_simulations(
    #     n_runs=10,
    #     sim_steps=300000,
    #     base_seed=int(time.time()) % 1000000  # Different seed each run
    # )



def plot_raster_coarsegrained_from_hdf5(filename: str, 
                                       save_path: str = "raster_coarse.png",
                                       start_time: Optional[int] = None,
                                       duration: Optional[int] = None,
                                       time_bin: int = 100,
                                       neuron_bin: int = 1,
                                       max_width_inches: float = 20.0,
                                       max_height_inches: float = 10.0,
                                       dpi: int = 150):
    """
    Create a coarse-grained raster plot from HDF5 file
    Each spike is represented as a single pixel, no overlaps
    
    Parameters:
    -----------
    filename : str
        HDF5 file containing spike data
    save_path : str
        Path to save the raster plot
    start_time : int, optional
        Starting timestep (if None, start from beginning)
    duration : int, optional
        Number of timesteps to plot (if None, plot all)
    time_bin : int
        Number of timesteps to bin together (default: 100)
    neuron_bin : int
        Number of neurons to bin together (default: 1)
    max_width_inches : float
        Maximum figure width in inches
    max_height_inches : float
        Maximum figure height in inches
    dpi : int
        DPI for the output image
    """
    
    print(f"Creating coarse-grained raster plot (time_bin={time_bin}, neuron_bin={neuron_bin})...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons = f['spikes_E'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins = n_neurons // neuron_bin
        
        print(f"Original: {duration} timesteps x {n_neurons} neurons")
        print(f"Binned: {n_time_bins} time bins x {n_neuron_bins} neuron bins")
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_neuron_bins, n_time_bins), dtype=bool)
        
        # Process data in chunks for memory efficiency
        chunk_size = 100000
        print("Processing spike data...")
        
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            
            # Load chunk from HDF5
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            
            # Find spikes in this chunk
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                # Adjust spike times to be relative to chunk
                spike_times += chunk_start
                
                # Bin the spikes
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                
                # Keep only spikes within our binned dimensions
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                
                # Mark bins with spikes (using OR to avoid overlaps)
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
            
            if chunk_start % 500000 == 0:
                print(f"  Processed {chunk_start:,} / {duration:,} timesteps")
        
        # Calculate figure size based on matrix dimensions and constraints
        aspect_ratio = n_time_bins / n_neuron_bins
        
        if aspect_ratio > max_width_inches / max_height_inches:
            # Width-limited
            width = max_width_inches
            height = width / aspect_ratio
        else:
            # Height-limited
            height = max_height_inches
            width = height * aspect_ratio
        
        print(f"Creating figure: {width:.1f} x {height:.1f} inches at {dpi} DPI")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        
        # Plot raster using imshow for efficiency
        # Each spike is exactly one pixel
        ax.imshow(raster_matrix, 
                  aspect='auto', 
                  cmap='binary_r',  # White background, black spikes
                  interpolation='nearest',
                  origin='lower')
        
        # Set proper labels
        ax.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + " neurons)" if neuron_bin > 1 else ""}', 
                     fontsize=12)
        ax.set_title(f'Raster Plot ({n_time_bins} x {n_neuron_bins} bins)', fontsize=14)
        
        # Add tick labels showing actual time/neuron values
        n_time_ticks = min(10, n_time_bins)
        n_neuron_ticks = min(10, n_neuron_bins)
        
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        neuron_tick_positions = np.linspace(0, n_neuron_bins-1, n_neuron_ticks, dtype=int)
        
        ax.set_xticks(time_tick_positions)
        ax.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                           for t in time_tick_positions], fontsize=10)
        
        ax.set_yticks(neuron_tick_positions)
        ax.set_yticklabels([f'{t*neuron_bin}' for t in neuron_tick_positions], fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Save
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        spike_density = np.sum(raster_matrix) / (n_time_bins * n_neuron_bins) * 100
        print(f"Spike density in plot: {spike_density:.2f}%")
        print(f"Coarse-grained raster plot saved to {save_path}")