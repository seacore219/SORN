import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Tuple, Optional, List, Any
import os
import h5py
import json
from datetime import datetime
import multiprocessing as mp
import time

##################################################################################
                           ###  Model Logic (L16-316)  ###
##################################################################################

class SORN:    
    def __init__(self, 
                 N_E: int = 200,
                 N_I: int = 40,
                 N_U: int = 0,  # Set to 0 as per paper (no external input by default)
                 eta_stdp: float = 0.004,
                 eta_istdp: float = 0.001,
                 eta_ip: float = 0.01,
                 h_ip: float = 0.1,  # Target firing rate (called h_ip in paper)
                 lambda_: float = 20,  # Expected number of connections per neuron
                 T_e_min: float = 0.0,  # Minimum excitatory threshold
                 T_e_max: float = 0.5,  # Maximum excitatory threshold
                 T_i_min: float = 0.0,  # Minimum inhibitory threshold
                 T_i_max: float = 1.0,  # Maximum inhibitory threshold
                 noise_sig: float = np.sqrt(0.05),  # Noise standard deviation
                 W_ee_initial: float = 0.001,  # Initial weight for new connections
                 W_min: float = 0.0,  # Minimum weight value
                 W_max: float = 1.0,  # Maximum weight value
                 W_ei_min: float = 0.001):  # Minimum weight for W_EI (never fully remove inhibition)
        
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
        
        # Only remove connections that are effectively zero
        # This prevents over-pruning of the network
        self.W_EE[self.W_EE < 1e-10] = 0
        
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


##################################################################################
   ###  Plotting, Data Processing, Memory Efficiency Functions (L323-1791) ###
##################################################################################
                
    def get_connection_fraction(self) -> float:
        """Calculate fraction of active connections in W_EE"""
        # Exclude diagonal from calculation
        mask = ~np.eye(self.N_E, dtype=bool)
        return (self.W_EE[mask] > 0).sum() / (self.N_E * (self.N_E - 1))
    
    def get_mean_weight(self) -> float:
        """Calculate mean weight of active connections"""
        active_weights = self.W_EE[self.W_EE > 0]
        return np.mean(active_weights) if len(active_weights) > 0 else 0


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


def run_single_simulation_with_eigenvalues(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single simulation with eigenvalue analysis after stabilization
    
    Parameters:
    -----------
    sim_params : dict
        Dictionary containing simulation parameters
        
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
        # Create all the standard datasets
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
        
        # Add inhibitory thresholds dataset
        dset_T_I = f.create_dataset('thresholds_I',
                                  shape=(n_threshold_saves, sorn.N_I),
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
                dset_T_I[threshold_idx] = sorn.T_I  # Save inhibitory thresholds too
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
                    'activity_E_at_checkpoint': float(np.mean(sorn.x)),
                    'activity_I_at_checkpoint': float(np.mean(sorn.y)),  # Add inhibitory activity
                    'connection_fraction_at_checkpoint': float(sorn.get_connection_fraction())
                }
                
                print(f"Eigenvalue analysis complete:")
                print(f"  Spectral radius: {spectral_radius:.4f}")
                print(f"  Unstable eigenvalues: {n_unstable}/{sorn.N_E}")
                print(f"  Network stability: {'STABLE' if spectral_radius <= 1 else 'UNSTABLE'}")
            
            # Progress report
            if t % 100000 == 0:
                print(f"Step {t}/{sim_steps}, Activity E: {np.mean(sorn.x):.3f}, "
                      f"Activity I: {np.mean(sorn.y):.3f}, "
                      f"Conn. Frac: {sorn.get_connection_fraction():.3f}")
    
    print(f"\nSimulation complete!")
    
    # Create standard plots
    print("\nCreating plots...")
    
    # Create all four raster plots (now including inhibitory)
    plot_raster_coarsegrained_from_hdf5_both(
        filename,
        save_path=os.path.join(run_dir, "raster_coarse_3M_both.png"),
        time_bin=1000,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    plot_raster_with_activity_coarsegrained_both(
        filename,
        save_path=os.path.join(run_dir, "raster_activity_coarse_3M_both.png"),
        time_bin=100,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    plot_raster_coarsegrained_from_hdf5_both(
        filename,
        save_path=os.path.join(run_dir, "raster_zoom_last100k_coarse_both.png"),
        start_time=sim_steps-100000,
        duration=100000,
        time_bin=100,
        neuron_bin=1,
        max_width_inches=20.0,
        dpi=150
    )
    
    plot_raster_with_activity_from_hdf5_both(
        filename,
        save_path=os.path.join(run_dir, "raster_with_activity_both.pdf"),
        start_time=sim_steps-10000,
        duration=10000,
        activity_height=2.0
    )
    
    # Create connection fraction plot with activity (like Fig 1A)
    print("Creating connection fraction plot with activity...")
    plot_connection_fraction_with_activity(filename, run_dir, run_id, sorn_params, eigenvalue_checkpoint)
    
    print(f"\nAll plots saved to {run_dir}!")
    
    # Prepare results
    results = {
        'run_id': run_id,
        'seed': seed,
        'parameters': sorn_params,
        'sim_steps': sim_steps,
        'final_activity_E': float(np.mean(sorn.x)),
        'final_activity_I': float(np.mean(sorn.y)),
        'final_mean_threshold_E': float(sorn.T_E.mean()),
        'final_mean_threshold_I': float(sorn.T_I.mean()),  # Add inhibitory threshold
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


def plot_connection_fraction_with_activity(filename: str, run_dir: str, run_id: str, 
                                         sorn_params: Dict, eigenvalue_checkpoint: int):
    """Create connection fraction plot with activity subplot (like Fig 1A)"""
    with h5py.File(filename, 'r') as f:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), height_ratios=[2, 1, 1])
        
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
        ax1.plot(x_vals[:phase1_end], conn_frac[:phase1_end] * 100, 
                color=c_notstable, linewidth=1.5, label='Growth/Decay')
        ax1.plot(x_vals[phase1_end:], conn_frac[phase1_end:] * 100, 
                color=c_stable, linewidth=1.5, label='Stable')
        
        # Mark eigenvalue checkpoint
        if eigenvalue_checkpoint <= x_vals[-1]:
            ax1.axvline(x=eigenvalue_checkpoint, color='red', linestyle='--', 
                      alpha=0.7, label=f'Eigenvalue analysis ({eigenvalue_checkpoint//1000000}M)')
        
        # Annotations
        ax1.text(x_vals[phase1_end//2], conn_frac[phase1_end//2] * 100 + 0.5, 
                'growth/decay', fontsize=12, color=c_notstable)
        ax1.text(x_vals[phase1_end] + x_vals[-1] * 0.1, conn_frac[phase1_end] * 100 + 0.5, 
                'stable', fontsize=12, color=c_stable)
        
        # Formatting
        ax1.set_xlim([0, x_vals[-1]])
        ax1.set_ylabel('Active Connections (%)', fontsize=12)
        ax1.set_title(f'Run {run_id}: Connection Fraction Evolution (h_ip={sorn_params.get("h_ip", 0.1)})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Remove bottom spine and x-axis for top plot
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xticks([])
        
        # Plot excitatory activity
        activity_E = f['activity_E'][:]
        # Downsample for plotting
        downsample_factor = 1000
        activity_E_downsampled = activity_E[::downsample_factor]
        x_activity = np.arange(len(activity_E_downsampled)) * downsample_factor
        
        ax2.plot(x_activity, activity_E_downsampled, 'k-', linewidth=0.8, alpha=0.7)
        ax2.axhline(y=sorn_params.get('h_ip', 0.1), color='red', linestyle='--', 
                   alpha=0.7, label=f'Target (h_ip={sorn_params.get("h_ip", 0.1)})')
        
        ax2.set_xlim([0, x_vals[-1]])
        ax2.set_ylabel('Mean Activity (E)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Remove bottom spine for middle plot
        ax2.spines['bottom'].set_visible(False)
        ax2.set_xticks([])
        
        # Plot inhibitory activity
        activity_I = f['activity_I'][:]
        activity_I_downsampled = activity_I[::downsample_factor]
        
        ax3.plot(x_activity, activity_I_downsampled, 'b-', linewidth=0.8, alpha=0.7)
        
        ax3.set_xlim([0, x_vals[-1]])
        ax3.set_xlabel(r'$10^6$ time steps', fontsize=12)
        ax3.set_ylabel('Mean Activity (I)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis to show in millions
        n_millions = int(x_vals[-1] / 1e6)
        ax3.set_xticks(np.arange(0, n_millions + 1) * 1e6)
        ax3.set_xticklabels([str(i) for i in range(n_millions + 1)])
        
        # Remove top spine for bottom plot
        ax3.spines['top'].set_visible(False)
        
        # Remove gap between subplots
        plt.subplots_adjust(hspace=0.02)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'connection_fraction_with_activity_both.png'), dpi=300, bbox_inches='tight')
        plt.close()

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
    
    # Run simulations with parallel processing
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
        ax.imshow(raster_matrix, 
                  aspect='auto', 
                  cmap='binary_r',
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


def plot_raster_coarsegrained_from_hdf5_both(filename: str, 
                                            save_path: str = "raster_coarse_both.png",
                                            start_time: Optional[int] = None,
                                            duration: Optional[int] = None,
                                            time_bin: int = 100,
                                            neuron_bin: int = 1,
                                            max_width_inches: float = 20.0,
                                            max_height_inches: float = 10.0,
                                            dpi: int = 150):
    """
    Create a coarse-grained raster plot from HDF5 file for both E and I neurons
    """
    print(f"Creating coarse-grained raster plot for E and I neurons...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons_E = f['spikes_E'].shape
        _, n_neurons_I = f['spikes_I'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins_E = n_neurons_E // neuron_bin
        n_neuron_bins_I = n_neurons_I // neuron_bin
        n_total_neuron_bins = n_neuron_bins_E + n_neuron_bins_I
        
        print(f"Original: {duration} timesteps x {n_neurons_E}E + {n_neurons_I}I neurons")
        print(f"Binned: {n_time_bins} time bins x {n_neuron_bins_E}E + {n_neuron_bins_I}I neuron bins")
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_total_neuron_bins, n_time_bins), dtype=bool)
        
        # Process excitatory data
        chunk_size = 100000
        print("Processing excitatory spike data...")
        
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
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins_E)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                
                # Mark bins with spikes (excitatory neurons go in top part)
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
        
        # Process inhibitory data
        print("Processing inhibitory spike data...")
        
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            
            # Load chunk from HDF5
            spike_chunk = f['spikes_I'][start_time + chunk_start:start_time + chunk_end]
            
            # Find spikes in this chunk
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                # Adjust spike times to be relative to chunk
                spike_times += chunk_start
                
                # Bin the spikes
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                
                # Keep only spikes within our binned dimensions
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins_I)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                
                # Mark bins with spikes (inhibitory neurons go below excitatory)
                raster_matrix[n_neuron_bins_E + neuron_bin_indices, time_bin_indices] = True
        
        # Calculate figure size based on matrix dimensions and constraints
        aspect_ratio = n_time_bins / n_total_neuron_bins
        
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
        
        # Create custom colormap (blue for inhibitory, black for excitatory)
        colors = np.zeros((n_total_neuron_bins, n_time_bins, 3))
        # Excitatory spikes (black)
        exc_mask = raster_matrix[:n_neuron_bins_E]
        colors[:n_neuron_bins_E][exc_mask] = [0, 0, 0]
        # Inhibitory spikes (blue)
        inh_mask = raster_matrix[n_neuron_bins_E:]
        colors[n_neuron_bins_E:][inh_mask] = [0, 0, 1]
        # White background
        colors[~raster_matrix] = [1, 1, 1]
        
        # Plot raster using imshow
        ax.imshow(colors, 
                  aspect='auto', 
                  interpolation='nearest',
                  origin='lower')
        
        # Add separator line between E and I neurons
        ax.axhline(y=n_neuron_bins_E, color='red', linewidth=2, alpha=0.7)
        
        # Set proper labels
        ax.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + " neurons)" if neuron_bin > 1 else ""}', 
                     fontsize=12)
        ax.set_title(f'Raster Plot E & I ({n_time_bins} x {n_total_neuron_bins} bins)', fontsize=14)
        
        # Add tick labels showing actual time/neuron values
        n_time_ticks = min(10, n_time_bins)
        
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        
        ax.set_xticks(time_tick_positions)
        ax.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                           for t in time_tick_positions], fontsize=10)
        
        # Custom y-ticks to show E and I regions
        y_ticks = [0, n_neuron_bins_E//2, n_neuron_bins_E, 
                   n_neuron_bins_E + n_neuron_bins_I//2, n_total_neuron_bins-1]
        y_labels = ['E:0', f'E:{n_neuron_bins_E//2 * neuron_bin}', 
                    f'E:{n_neuron_bins_E * neuron_bin}', 
                    f'I:{n_neuron_bins_I//2 * neuron_bin}', 
                    f'I:{n_neuron_bins_I * neuron_bin}']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Save
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        spike_density_E = np.sum(raster_matrix[:n_neuron_bins_E]) / (n_time_bins * n_neuron_bins_E) * 100
        spike_density_I = np.sum(raster_matrix[n_neuron_bins_E:]) / (n_time_bins * n_neuron_bins_I) * 100
        print(f"Spike density E: {spike_density_E:.2f}%, I: {spike_density_I:.2f}%")
        print(f"Coarse-grained raster plot saved to {save_path}")


def plot_raster_with_activity_from_hdf5_both(filename: str,
                                            save_path: str = "raster_with_activity_both.pdf",
                                            start_time: Optional[int] = None,
                                            duration: Optional[int] = None,
                                            activity_height: float = 2.0):
    """
    Create a full-resolution raster plot with activity traces for both E and I from HDF5
    """
    print("Creating full-resolution raster plot with E and I activity from HDF5...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons_E = f['spikes_E'].shape
        _, n_neurons_I = f['spikes_I'].shape
        n_neurons_total = n_neurons_E + n_neurons_I
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate figure dimensions
        width_inches = duration / 72.0
        raster_height = n_neurons_total / 72.0
        total_height = raster_height + 2 * activity_height + 1.0  # Two activity plots
        
        print(f"Creating figure: {width_inches:.1f} x {total_height:.1f} inches")
        
        # Create figure
        fig = plt.figure(figsize=(width_inches, total_height), dpi=72)
        
        # Calculate height ratios
        height_ratio_raster = raster_height / total_height
        height_ratio_activity = activity_height / total_height
        height_ratio_gap = 0.5 / total_height
        
        # Create axes
        ax_raster = plt.axes([0, 2 * height_ratio_activity + height_ratio_gap, 
                             1, height_ratio_raster])
        ax_activity_E = plt.axes([0, height_ratio_activity, 1, height_ratio_activity])
        ax_activity_I = plt.axes([0, 0, 1, height_ratio_activity])
        
        # Plot excitatory raster
        print("Plotting excitatory raster...")
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
        
        # Plot inhibitory raster
        print("Plotting inhibitory raster...")
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_I'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                ax_raster.vlines(spike_times + chunk_start, 
                               n_neurons_E + neuron_ids - 0.5, 
                               n_neurons_E + neuron_ids + 0.5,
                               colors='blue',
                               linewidth=1.0,
                               antialiased=False)
        
        # Add separator line
        ax_raster.axhline(y=n_neurons_E, color='red', linewidth=2, alpha=0.7)
        
        # Set raster limits
        ax_raster.set_xlim(0, duration)
        ax_raster.set_ylim(-0.5, n_neurons_total - 0.5)
        ax_raster.axis('off')
        
        # Plot excitatory activity
        print("Plotting excitatory activity...")
        activity_E = f['activity_E'][start_time:start_time + duration]
        ax_activity_E.fill_between(np.arange(duration), 
                                 0, 
                                 activity_E,
                                 color='black',
                                 alpha=0.7,
                                 linewidth=0)
        
        # Add target activity line
        ax_activity_E.axhline(y=0.1, color='red', linewidth=1, alpha=0.8)
        
        # Set activity limits
        ax_activity_E.set_xlim(0, duration)
        ax_activity_E.set_ylim(0, max(0.2, activity_E.max() * 1.1))
        ax_activity_E.set_ylabel('Activity (E)', fontsize=10)
        ax_activity_E.spines['top'].set_visible(False)
        ax_activity_E.spines['right'].set_visible(False)
        
        # Plot inhibitory activity
        print("Plotting inhibitory activity...")
        activity_I = f['activity_I'][start_time:start_time + duration]
        ax_activity_I.fill_between(np.arange(duration), 
                                 0, 
                                 activity_I,
                                 color='blue',
                                 alpha=0.7,
                                 linewidth=0)
        
        # Set activity limits
        ax_activity_I.set_xlim(0, duration)
        ax_activity_I.set_ylim(0, max(0.2, activity_I.max() * 1.1))
        ax_activity_I.set_xlabel('Time Steps', fontsize=10)
        ax_activity_I.set_ylabel('Activity (I)', fontsize=10)
        ax_activity_I.spines['top'].set_visible(False)
        ax_activity_I.spines['right'].set_visible(False)
        
        # Add tick marks
        if duration > 1000000:
            tick_positions = np.arange(0, duration + 1, 1000000)
            ax_activity_I.set_xticks(tick_positions)
            ax_activity_I.set_xticklabels([f'{t/1e6:.0f}M' for t in tick_positions], fontsize=8)
        
       # Save as PDF
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=72)
        plt.close()
        
        print(f"Full-resolution raster plot with E and I activity saved to {save_path}")


def plot_raster_with_activity_coarsegrained_both(filename: str,
                                                save_path: str = "raster_activity_coarse_both.png",
                                                start_time: Optional[int] = None,
                                                duration: Optional[int] = None,
                                                time_bin: int = 100,
                                                neuron_bin: int = 1,
                                                max_width_inches: float = 20.0,
                                                dpi: int = 150):
    """
    Create a coarse-grained raster plot with activity traces for both E and I
    """
    print(f"Creating coarse-grained raster with E and I activity plot...")
    
    with h5py.File(filename, 'r') as f:
        # Get dimensions
        total_timesteps, n_neurons_E = f['spikes_E'].shape
        _, n_neurons_I = f['spikes_I'].shape
        
        # Determine time range
        if start_time is None:
            start_time = 0
        if duration is None:
            duration = total_timesteps - start_time
        else:
            duration = min(duration, total_timesteps - start_time)
        
        # Calculate binned dimensions
        n_time_bins = duration // time_bin
        n_neuron_bins_E = n_neurons_E // neuron_bin
        n_neuron_bins_I = n_neurons_I // neuron_bin
        n_total_neuron_bins = n_neuron_bins_E + n_neuron_bins_I
        
        # Create binary matrix for raster data
        raster_matrix = np.zeros((n_total_neuron_bins, n_time_bins), dtype=bool)
        
        # Process excitatory data
        chunk_size = 100000
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_E'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                spike_times += chunk_start
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins_E)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                raster_matrix[neuron_bin_indices, time_bin_indices] = True
        
        # Process inhibitory data
        for chunk_start in range(0, duration, chunk_size):
            chunk_end = min(chunk_start + chunk_size, duration)
            spike_chunk = f['spikes_I'][start_time + chunk_start:start_time + chunk_end]
            spike_times, neuron_ids = np.where(spike_chunk)
            
            if len(spike_times) > 0:
                spike_times += chunk_start
                time_bin_indices = spike_times // time_bin
                neuron_bin_indices = neuron_ids // neuron_bin
                valid_mask = (time_bin_indices < n_time_bins) & (neuron_bin_indices < n_neuron_bins_I)
                time_bin_indices = time_bin_indices[valid_mask]
                neuron_bin_indices = neuron_bin_indices[valid_mask]
                raster_matrix[n_neuron_bins_E + neuron_bin_indices, time_bin_indices] = True
        
        # Load and bin activity data
        activity_E = f['activity_E'][start_time:start_time + duration]
        activity_I = f['activity_I'][start_time:start_time + duration]
        
        # Reshape and average over time bins
        n_complete_bins = (len(activity_E) // time_bin) * time_bin
        activity_E_binned = activity_E[:n_complete_bins].reshape(-1, time_bin).mean(axis=1)
        activity_I_binned = activity_I[:n_complete_bins].reshape(-1, time_bin).mean(axis=1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(max_width_inches, 12), dpi=dpi)
        
        # Create grid spec for better control
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.15)
        
        ax_raster = fig.add_subplot(gs[0])
        ax_activity_E = fig.add_subplot(gs[1], sharex=ax_raster)
        ax_activity_I = fig.add_subplot(gs[2], sharex=ax_raster)
        
        # Create custom colormap for raster
        colors = np.zeros((n_total_neuron_bins, n_time_bins, 3))
        # Excitatory spikes (black)
        exc_mask = raster_matrix[:n_neuron_bins_E]
        colors[:n_neuron_bins_E][exc_mask] = [0, 0, 0]
        # Inhibitory spikes (blue)
        inh_mask = raster_matrix[n_neuron_bins_E:]
        colors[n_neuron_bins_E:][inh_mask] = [0, 0, 1]
        # White background
        colors[~raster_matrix] = [1, 1, 1]
        
        # Plot raster
        im = ax_raster.imshow(colors, 
                            aspect='auto', 
                            interpolation='nearest',
                            origin='lower')
        
        # Add separator line
        ax_raster.axhline(y=n_neuron_bins_E, color='red', linewidth=2, alpha=0.7)
        
        ax_raster.set_ylabel(f'Neuron ID {"(bins of " + str(neuron_bin) + ")" if neuron_bin > 1 else ""}', 
                            fontsize=12)
        ax_raster.set_title(f'Raster Plot E & I and Activity ({n_time_bins} time bins)', fontsize=14)
        
        # Plot excitatory activity
        time_axis = np.arange(len(activity_E_binned))
        ax_activity_E.fill_between(time_axis, 0, activity_E_binned, 
                                 color='black', alpha=0.7, linewidth=0)
        ax_activity_E.axhline(y=0.1, color='red', linewidth=2, alpha=0.8, 
                            linestyle='--', label='Target (h_ip=0.1)')
        
        ax_activity_E.set_ylabel('Activity (E)', fontsize=12)
        ax_activity_E.set_ylim(0, max(0.2, activity_E_binned.max() * 1.1))
        ax_activity_E.legend(loc='upper right')
        ax_activity_E.grid(True, alpha=0.3)
        
        # Plot inhibitory activity
        ax_activity_I.fill_between(time_axis, 0, activity_I_binned, 
                                 color='blue', alpha=0.7, linewidth=0)
        
        ax_activity_I.set_xlabel(f'Time (bins of {time_bin} steps)', fontsize=12)
        ax_activity_I.set_ylabel('Activity (I)', fontsize=12)
        ax_activity_I.set_ylim(0, max(0.2, activity_I_binned.max() * 1.1))
        ax_activity_I.grid(True, alpha=0.3)
        
        # Set x-axis labels
        n_time_ticks = min(10, n_time_bins)
        time_tick_positions = np.linspace(0, n_time_bins-1, n_time_ticks, dtype=int)
        ax_activity_I.set_xticks(time_tick_positions)
        ax_activity_I.set_xticklabels([f'{(start_time + t*time_bin)/1e6:.1f}M' 
                                     for t in time_tick_positions], fontsize=10)
        
        # Custom y-ticks for raster to show E and I regions
        y_ticks = [0, n_neuron_bins_E//2, n_neuron_bins_E, 
                   n_neuron_bins_E + n_neuron_bins_I//2, n_total_neuron_bins-1]
        y_labels = ['E:0', f'E:{n_neuron_bins_E//2 * neuron_bin}', 
                    f'E:{n_neuron_bins_E * neuron_bin}', 
                    f'I:{n_neuron_bins_I//2 * neuron_bin}', 
                    f'I:{n_neuron_bins_I * neuron_bin}']
        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_labels, fontsize=10)
        
        # Style
        ax_raster.grid(True, alpha=0.2, linewidth=0.5)
        
        plt.tight_layout()
        print(f"Saving to {save_path}...")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Coarse-grained raster with E and I activity saved to {save_path}")

##################################################################################
                             ###  Main (L1797-2026) ###
##################################################################################       

if __name__ == "__main__":
    # =========================================================
    PARAM_TO_SWEEP = 'h_ip'  # or 'noise_sig', 'eta_stdp', etc.
    PARAM_VALUES = [0.1]  # Sweep across multiple h_ip values
    N_SEEDS_PER_VALUE = 10
    SIM_STEPS = 6000000
    EIGENVALUE_CHECKPOINT = 2000000
    USE_CORES_PERCENT = 0.5  
    # ==========================================================
    
    # Calculate number of processes based on percentage
    total_cores = mp.cpu_count()
    N_PROCESSES = max(1, int(total_cores * USE_CORES_PERCENT))
    
    print("="*80)
    print(f"SORN Parameter Sweep: {PARAM_TO_SWEEP} = {PARAM_VALUES} with Eigenvalue Analysis")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define parameter sweep configuration
    param_ranges = {
        PARAM_TO_SWEEP: PARAM_VALUES
    }
    
    # Base parameters (all other parameters stay constant)
    base_params = {
        'N_E': 200,
        'N_I': 40,
        'eta_stdp': 0.004,
        'eta_istdp': 0.001,
        'eta_ip': 0.0001,
        'lambda_': 20,
        'noise_sig': np.sqrt(0.05),
        'T_e_min': 0.0,
        'T_e_max': 0.5,
        'T_i_min': 0.0,
        'T_i_max': 1.0,
        'W_ee_initial': 0.001,
        'W_min': 0.0,
        'W_max': 1.0,
        'W_ei_min': 0.001
    }
    
    # Use timestamp-based seed for reproducibility
    base_seed = int(time.time()) % 1000000
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{PARAM_TO_SWEEP}_sweep_eigenvalues_{timestamp}"
    
    print(f"Configuration:")
    print(f"  - {PARAM_TO_SWEEP} values: {PARAM_VALUES}")
    print(f"  - Seeds per value: {N_SEEDS_PER_VALUE}")
    print(f"  - Total runs: {len(PARAM_VALUES) * N_SEEDS_PER_VALUE}")
    print(f"  - Simulation steps: {SIM_STEPS:,} ({SIM_STEPS/1e6:.1f}M)")
    print(f"  - Eigenvalue checkpoint: {EIGENVALUE_CHECKPOINT:,} ({EIGENVALUE_CHECKPOINT/1e6:.1f}M)")
    print(f"  - Base random seed: {base_seed}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Parallel processes: {N_PROCESSES} of {total_cores} cores ({USE_CORES_PERCENT*100:.0f}%)")
    print()
    
    # Run the parameter sweep with eigenvalue analysis
    print("Starting parameter sweep...")
    print("-"*60)
    
    try:
        sweep_results = run_parameter_sweep_with_eigenvalues(
            param_ranges=param_ranges,
            base_params=base_params,
            sim_steps=SIM_STEPS,
            eigenvalue_checkpoint=EIGENVALUE_CHECKPOINT,
            output_dir=output_dir,
            n_processes=N_PROCESSES,
            n_seeds=N_SEEDS_PER_VALUE,
            base_seed=base_seed
        )
        
        print("\n" + "="*80)
        print("SWEEP COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Analyze and summarize results
        print("\nRESULTS SUMMARY:")
        print("-"*60)
        
        # Group results by parameter value
        results_by_param = {v: [] for v in PARAM_VALUES}
        eigenvalue_results_by_param = {v: [] for v in PARAM_VALUES}
        
        for result in sweep_results:
            param_value = result['parameters'][PARAM_TO_SWEEP]
            results_by_param[param_value].append(result)
            
            if 'eigenvalue_analysis' in result:
                eigenvalue_results_by_param[param_value].append(result['eigenvalue_analysis'])
        
        # Print detailed summary for each parameter value
        for param_val in PARAM_VALUES:
            print(f"\n>>> {PARAM_TO_SWEEP} = {param_val} <<<")
            print("-"*40)
            
            runs = results_by_param[param_val]
            eigen_analyses = eigenvalue_results_by_param[param_val]
            
            if runs:
                # Activity statistics
                final_activities_E = [r['final_activity_E'] for r in runs]
                final_activities_I = [r['final_activity_I'] for r in runs]
                print(f"\nFinal Activity (at {SIM_STEPS/1e6:.0f}M steps):")
                print(f"  Excitatory - Mean: {np.mean(final_activities_E):.4f} ± {np.std(final_activities_E):.4f}")
                print(f"  Excitatory - Range: [{np.min(final_activities_E):.4f}, {np.max(final_activities_E):.4f}]")
                print(f"  Inhibitory - Mean: {np.mean(final_activities_I):.4f} ± {np.std(final_activities_I):.4f}")
                print(f"  Inhibitory - Range: [{np.min(final_activities_I):.4f}, {np.max(final_activities_I):.4f}]")
                
                # Connection fraction statistics
                final_conn_fracs = [r['final_connection_fraction'] for r in runs]
                print(f"\nFinal Connection Fraction:")
                print(f"  Mean: {np.mean(final_conn_fracs):.4f} ± {np.std(final_conn_fracs):.4f}")
                print(f"  Range: [{np.min(final_conn_fracs):.4f}, {np.max(final_conn_fracs):.4f}]")
                
                # Eigenvalue analysis (at checkpoint)
                if eigen_analyses:
                    print(f"\nEigenvalue Analysis (at {EIGENVALUE_CHECKPOINT/1e6:.0f}M steps):")
                    
                    spectral_radii = [ea['spectral_radius'] for ea in eigen_analyses]
                    print(f"  Spectral Radius:")
                    print(f"    Mean: {np.mean(spectral_radii):.4f} ± {np.std(spectral_radii):.4f}")
                    print(f"    Range: [{np.min(spectral_radii):.4f}, {np.max(spectral_radii):.4f}]")
                    
                    n_unstable_list = [ea['n_unstable'] for ea in eigen_analyses]
                    print(f"  Unstable Eigenvalues (|λ| > 1):")
                    print(f"    Mean: {np.mean(n_unstable_list):.1f} ± {np.std(n_unstable_list):.1f}")
                    print(f"    Range: [{np.min(n_unstable_list)}, {np.max(n_unstable_list)}]")
                    
                    activities_E_at_checkpoint = [ea['activity_E_at_checkpoint'] for ea in eigen_analyses]
                    activities_I_at_checkpoint = [ea['activity_I_at_checkpoint'] for ea in eigen_analyses]
                    print(f"  Activity at checkpoint:")
                    print(f"    Excitatory - Mean: {np.mean(activities_E_at_checkpoint):.4f} ± {np.std(activities_E_at_checkpoint):.4f}")
                    print(f"    Inhibitory - Mean: {np.mean(activities_I_at_checkpoint):.4f} ± {np.std(activities_I_at_checkpoint):.4f}")
                    
                    # Stability assessment
                    all_stable = all(sr <= 1.0 for sr in spectral_radii)
                    print(f"  Network Stability: {'ALL STABLE' if all_stable else 'SOME UNSTABLE'}")
                    
                    # Individual run details
                    print(f"\n  Individual Runs:")
                    for i, (ea, r) in enumerate(zip(eigen_analyses, runs)):
                        stability = 'STABLE' if ea['spectral_radius'] <= 1.0 else 'UNSTABLE'
                        print(f"    Seed {i}: ρ={ea['spectral_radius']:.4f}, "
                              f"Activity E={ea['activity_E_at_checkpoint']:.3f}, "
                              f"Activity I={ea['activity_I_at_checkpoint']:.3f}, "
                              f"{stability}")
        
        # Comparison between parameter values (if more than one)
        if len(PARAM_VALUES) > 1:
            print("\n" + "="*60)
            print(f"COMPARISON ACROSS {PARAM_TO_SWEEP} VALUES")
            print("="*60)
            
            # Compare spectral radii
            print(f"\nSpectral Radius Summary:")
            for param_val in PARAM_VALUES:
                if eigenvalue_results_by_param[param_val]:
                    sr_vals = [ea['spectral_radius'] for ea in eigenvalue_results_by_param[param_val]]
                    print(f"  {PARAM_TO_SWEEP}={param_val}: {np.mean(sr_vals):.4f} ± {np.std(sr_vals):.4f}")
            
            # Compare activities
            print(f"\nActivity at Checkpoint Summary:")
            for param_val in PARAM_VALUES:
                if eigenvalue_results_by_param[param_val]:
                    act_E_vals = [ea['activity_E_at_checkpoint'] for ea in eigenvalue_results_by_param[param_val]]
                    act_I_vals = [ea['activity_I_at_checkpoint'] for ea in eigenvalue_results_by_param[param_val]]
                    print(f"  {PARAM_TO_SWEEP}={param_val}:")
                    print(f"    Excitatory: {np.mean(act_E_vals):.4f} ± {np.std(act_E_vals):.4f}")
                    print(f"    Inhibitory: {np.mean(act_I_vals):.4f} ± {np.std(act_I_vals):.4f}")
            
            # Stability comparison
            print(f"\nNetwork Stability Summary:")
            for param_val in PARAM_VALUES:
                if eigenvalue_results_by_param[param_val]:
                    sr_vals = [ea['spectral_radius'] for ea in eigenvalue_results_by_param[param_val]]
                    stable_count = sum(1 for sr in sr_vals if sr <= 1.0)
                    print(f"  {PARAM_TO_SWEEP}={param_val}: {stable_count}/{len(sr_vals)} networks stable")
        
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
        for param_val in PARAM_VALUES:
            print(f"\n  {PARAM_TO_SWEEP}_{param_val}/")
            for i in range(N_SEEDS_PER_VALUE):
                print(f"    └── seed_{i}/")
                print(f"        ├── spiking_data.h5")
                print(f"        ├── eigenvalue_spectrum_{EIGENVALUE_CHECKPOINT//1000000}M.png")
                print(f"        ├── raster_coarse_3M_both.png")
                print(f"        ├── raster_activity_coarse_3M_both.png")
                print(f"        ├── raster_zoom_last100k_coarse_both.png")
                print(f"        ├── raster_with_activity_both.pdf")
                print(f"        ├── connection_fraction_with_activity_both.png")
                print(f"        └── summary.json")
        
        # Final notes
        print("\n" + "="*60)
        print("NOTES:")
        print("="*60)
        print(f"1. Eigenvalue analysis was performed at {EIGENVALUE_CHECKPOINT/1e6:.0f}M timesteps (stabilized network)")
        print("2. Spectral radius > 1 indicates potential instability")
        print(f"3. Each {PARAM_TO_SWEEP} value was tested with {N_SEEDS_PER_VALUE} different random seeds")
        print(f"4. Full spike data ({SIM_STEPS/1e6:.0f}M timesteps) saved in HDF5 format")
        print("5. All plots include eigenvalue checkpoint markers where applicable")
        print("6. Inhibitory spikes are now saved and shown in blue in all plots")
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nParameter sweep completed successfully!")
        
    except Exception as e:
        print(f"\nERROR: Parameter sweep failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*80)