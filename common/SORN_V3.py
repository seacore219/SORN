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
    Based on: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178683
    """
    
    def __init__(self, 
                 N_E: int = 200,
                 N_I: int = 40,
                 N_U: int = 1,
                 eta_stdp: float = 0.004,
                 eta_istdp: float = 0.001,
                 eta_ip: float = 0.01,
                 mu_ip: float = 0.1,
                 lambda_: float = 20,
                 noise_sig_e: float = np.sqrt(0.05),
                 noise_sig_i: float = np.sqrt(0.05)):
        """
        Initialize SORN network
        
        Parameters:
        -----------
        N_E : int
            Number of excitatory neurons
        N_I : int  
            Number of inhibitory neurons
        N_U : int
            Number of input units
        eta_stdp : float
            STDP learning rate
        eta_istdp : float
            iSTDP learning rate  
        eta_ip : float
            Intrinsic plasticity learning rate
        mu_ip : float
            Target firing rate for IP
        lambda_ : float
            Expected number of connections per neuron
        noise_sig_e : float
            Excitatory noise standard deviation
        noise_sig_i : float
            Inhibitory noise standard deviation
        """
        
        # Network size
        self.N_E = N_E
        self.N_I = N_I
        self.N_U = N_U
        
        # Learning rates
        self.eta_stdp = eta_stdp
        self.eta_istdp = eta_istdp
        self.eta_ip = eta_ip
        self.mu_ip = mu_ip
        
        # Network parameters
        self.lambda_ = lambda_
        self.p_c = lambda_ / N_E  # Connection probability
        self.p_sp = N_E * (N_E - 1) * 0.1 / (200 * 199)  # Structural plasticity probability
        
        # Noise parameters
        self.noise_sig_e = noise_sig_e
        self.noise_sig_i = noise_sig_i
        
        # Initialize network components
        self._initialize_network()
        
    def _initialize_network(self):
        """Initialize all network components"""
        
        # State vectors
        self.x = np.random.rand(self.N_E) < self.mu_ip  # Excitatory states
        self.y = np.zeros(self.N_I)  # Inhibitory states
        self.u = np.zeros(self.N_U)  # Input states
        
        # Pre-threshold activities
        self.R_x = np.zeros(self.N_E)
        self.R_y = np.zeros(self.N_I)
        
        # Thresholds (uniformly distributed)
        self.T_E = np.random.uniform(0.5, 1.0, self.N_E)
        self.T_I = np.random.uniform(0.5, 1.0, self.N_I)
        
        # Initialize weight matrices
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize synaptic weight matrices"""
        
        # W_EE: Excitatory to excitatory
        self.W_EE = self._create_sparse_matrix(self.N_E, self.N_E, self.p_c, avoid_self=True)
        
        # W_EI: Inhibitory to excitatory  
        self.W_EI = self._create_sparse_matrix(self.N_E, self.N_I, 0.2)
        
        # W_IE: Excitatory to inhibitory
        self.W_IE = self._create_sparse_matrix(self.N_I, self.N_E, 1.0)
        
        # W_EU: Input to excitatory (full connectivity for now)
        self.W_EU = np.ones((self.N_E, self.N_U))
        
        # Normalize all weight matrices
        self._normalize_weights()
        
    def _create_sparse_matrix(self, rows: int, cols: int, p: float, avoid_self: bool = False) -> np.ndarray:
        """Create sparse weight matrix with connection probability p"""
        
        # Create mask for connections
        mask = np.random.rand(rows, cols) < p
        
        if avoid_self and rows == cols:
            np.fill_diagonal(mask, False)
            
        # Ensure each neuron has at least one input
        for i in range(rows):
            if not mask[i].any():
                # Add random connection
                j = np.random.randint(cols)
                if avoid_self and i == j:
                    j = (j + 1) % cols
                mask[i, j] = True
                
        # Initialize weights
        W = np.random.rand(rows, cols) * mask
        
        return W
        
    def _normalize_weights(self):
        """Normalize all incoming weights to sum to 1"""
        
        # Normalize W_EE (each row sums to 1)
        row_sums = self.W_EE.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.W_EE /= row_sums
        
        # Normalize W_EI
        row_sums = self.W_EI.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.W_EI /= row_sums
        
        # Normalize W_IE
        row_sums = self.W_IE.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.W_IE /= row_sums
        
        # Normalize W_EU
        row_sums = self.W_EU.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.W_EU /= row_sums
        
    def step(self, u_new: np.ndarray):
        """
        Perform one update step of the SORN network
        
        Parameters:
        -----------
        u_new : np.ndarray
            New input vector
        """
        
        # Store previous states for plasticity
        x_prev = self.x.copy()
        y_prev = self.y.copy()
        
        # Update input
        self.u = u_new
        
        # Compute pre-threshold excitatory activity (Equation from Image 1)
        self.R_x = (self.W_EE @ self.x - self.W_EI @ self.y + 
                    self.W_EU @ self.u + 
                    self.noise_sig_e * np.random.randn(self.N_E) - self.T_E)
        
        # Apply threshold function (Heaviside)
        x_new = (self.R_x >= 0).astype(float)
        
        # Compute pre-threshold inhibitory activity (Equation from Image 2)
        self.R_y = (self.W_IE @ self.x + 
                    self.noise_sig_i * np.random.randn(self.N_I) - self.T_I)
        
        # Apply threshold function
        y_new = (self.R_y >= 0).astype(float)
        
        # Apply plasticity rules
        self._apply_plasticity(x_prev, x_new, y_prev, y_new)
        
        # Update states
        self.x = x_new
        self.y = y_new
        
    def _apply_plasticity(self, x_prev: np.ndarray, x_new: np.ndarray, 
                         y_prev: np.ndarray, y_new: np.ndarray):
        """Apply all plasticity rules"""
        
        # 1. STDP (Equation from Image 3)
        self._stdp(x_prev, x_new)
        
        # 2. iSTDP (Equation from Image 4)
        self._istdp(y_prev, x_new)
        
        # 3. Intrinsic plasticity
        self._ip(x_new)
        
        # 4. Structural plasticity
        self._structural_plasticity()
        
        # 5. Synaptic normalization (must be done after all weight updates)
        self._normalize_weights()
        
    def _stdp(self, x_prev: np.ndarray, x_new: np.ndarray):
        """
        Spike-Timing Dependent Plasticity
        ΔW_ij^EE(t) = η_STDP[x_i(t)x_j(t-1) - x_i(t-1)x_j(t)]
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
        
        # Clip weights to [0, 1]
        self.W_EE = np.clip(self.W_EE, 0, 1)
        
    def _istdp(self, y_prev: np.ndarray, x_new: np.ndarray):
        """
        Inhibitory STDP
        ΔW_ij^EI(t) = -η_inhib * y_j(t-1) * [1 - x_i(t)(1 + 1/μ_IP)]
        """
        
        # Compute weight changes
        factor = 1 - x_new * (1 + 1/self.mu_ip)
        dW = -self.eta_istdp * np.outer(factor, y_prev)
        
        # Update weights
        self.W_EI += dW
        
        # Clip weights to [0.001, 1] (avoid zero weights)
        self.W_EI = np.clip(self.W_EI, 0.001, 1)
        
    def _ip(self, x_new: np.ndarray):
        """
        Intrinsic Plasticity
        ΔT_i^E = η_IP(x_i - μ_IP)
        """
        
        # Update thresholds
        self.T_E += self.eta_ip * (x_new - self.mu_ip)
        
        # Keep thresholds in reasonable range
        self.T_E = np.clip(self.T_E, 0.01, 2.0)
        
    def _structural_plasticity(self):
        """
        Structural Plasticity
        With probability p_SP, create new synapse with weight 0.001
        """
        
        if np.random.rand() < self.p_sp:
            # Find zero entries in W_EE
            zero_mask = self.W_EE == 0
            # Exclude diagonal
            np.fill_diagonal(zero_mask, False)
            
            # Get indices of zero entries
            zero_indices = np.argwhere(zero_mask)
            
            if len(zero_indices) > 0:
                # Select random zero entry
                idx = np.random.randint(len(zero_indices))
                i, j = zero_indices[idx]
                
                # Create new connection
                self.W_EE[i, j] = 0.001
                
    def get_connection_fraction(self) -> float:
        """Calculate fraction of active connections in W_EE"""
        return (self.W_EE > 0).sum() / (self.N_E * (self.N_E - 1))
        
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
        if input_pattern is None:
            input_pattern = np.zeros((steps, self.N_U))
            
        # Initialize history
        history = {
            'x': np.zeros((steps, self.N_E)),
            'y': np.zeros((steps, self.N_I)),
            'T_E': np.zeros((steps, self.N_E)),
            'activity_E': np.zeros(steps),
            'activity_I': np.zeros(steps),
            'connection_fraction': []
        }
        
        # Run simulation
        for t in range(steps):
            # Get input for this timestep
            u_t = input_pattern[t] if t < len(input_pattern) else np.zeros(self.N_U)
            
            # Update network
            self.step(u_t)
            
            # Record states
            history['x'][t] = self.x
            history['y'][t] = self.y
            history['T_E'][t] = self.T_E
            history['activity_E'][t] = self.x.sum()  # Total active neurons
            history['activity_I'][t] = self.y.mean()
            
            # Record connection fraction at intervals
            if t % record_interval == 0:
                history['connection_fraction'].append(self.get_connection_fraction())
            
            # Progress report
            if t % 100000 == 0:
                print(f"Step {t}/{steps}, Activity: {history['activity_E'][t]:.1f}, "
                      f"Conn. Frac: {self.get_connection_fraction():.3f}")
        
        history['connection_fraction'] = np.array(history['connection_fraction'])
        
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
        
        # Save metadata
        f.attrs['N_E'] = history['x'].shape[1]
        f.attrs['N_I'] = history['y'].shape[1]
        f.attrs['n_timesteps'] = history['x'].shape[0]
        
    print(f"Spiking data saved to {filename}")

def plot_raster(history: Dict[str, np.ndarray], 
                start_time: int = 0, 
                duration: int = 1000,
                save_path: str = "raster_plot.png"):
    """
    Create and save a raster plot of network activity
    
    Parameters:
    -----------
    history : dict
        Simulation history containing spike data
    start_time : int
        Starting time for the plot
    duration : int
        Duration to plot
    save_path : str
        Path to save the raster plot
    """
    
    # Extract spike data
    spike_data = history['x'][start_time:start_time+duration]
    
    # Create figure with better aspect ratio for raster plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                   gridspec_kw={'height_ratios': [4, 1]})
    
    # Find spike locations (time, neuron) - no transpose needed
    spike_times, neuron_ids = np.where(spike_data)
    
    # Raster plot
    ax1.scatter(spike_times + start_time, neuron_ids, s=0.5, c='black', marker='|', alpha=0.8)
    ax1.set_xlim(start_time, start_time + duration)
    ax1.set_ylim(-1, history['x'].shape[1])  # Add small margin
    ax1.set_ylabel('Neuron ID', fontsize=12)
    ax1.set_title(f'Raster Plot (t = {start_time} to {start_time + duration})', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Activity plot
    activity = history['activity_E'][start_time:start_time+duration]
    time_axis = np.arange(start_time, start_time + duration)
    ax2.plot(time_axis, activity, 'k-', linewidth=1)
    ax2.fill_between(time_axis, 0, activity, alpha=0.3, color='gray')
    ax2.set_xlim(start_time, start_time + duration)
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Active Neurons', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Adjust spacing
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Raster plot saved to {save_path}")
    plt.close()


def plot_figure1(history: Dict[str, np.ndarray], save_path: str = "figure1.pdf"):
    """
    Create and save Figure 1 from the paper
    
    Parameters:
    -----------
    history : dict
        Simulation history
    save_path : str
        Path to save the figure
    """
    
    # Figure parameters
    width = 10
    height = 3
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.7])
    
    letter_size = 13
    line_width = 1.0
    
    # Colors from the paper
    c_size = '#B22400'
    c_duration = '#006BB2'
    c_stable = '#2E4172'
    c_notstable = '#7887AB'
    
    # Panel A: Connection Fraction
    fig_1a = plt.subplot(gs[0])
    
    # Get connection fraction data
    conn_frac = history['connection_fraction']
    steps = len(history['activity_E'])
    steps_per_record = steps // len(conn_frac)
    x_vals = np.arange(len(conn_frac)) * steps_per_record
    
    # Identify phases (adjust based on your simulation)
    # Find transition point (where connection fraction stabilizes)
    # Simple heuristic: when change becomes small
    diff = np.abs(np.diff(conn_frac))
    transition_idx = np.where(diff < 0.0001)[0]
    if len(transition_idx) > 0:
        transition_point = transition_idx[0]
    else:
        transition_point = len(conn_frac) // 2
    
    # Plot connection fraction
    fig_1a.plot(x_vals[:transition_point], conn_frac[:transition_point] * 100, 
                c_notstable, linewidth=line_width)
    fig_1a.plot(x_vals[transition_point:], conn_frac[transition_point:] * 100, 
                c_stable, linewidth=line_width)
    
    # Annotations
    if conn_frac[0] > conn_frac[transition_point//2]:
        fig_1a.text(x_vals[transition_point//4], conn_frac[transition_point//4] * 100 + 1, 
                    'decay', fontsize=letter_size, color=c_notstable)
    else:
        fig_1a.text(x_vals[transition_point//4], conn_frac[transition_point//4] * 100 + 1, 
                    'growth', fontsize=letter_size, color=c_notstable)
    
    fig_1a.text(x_vals[transition_point] + 0.1e6, conn_frac[transition_point] * 100 + 0.5, 
                'stable', fontsize=letter_size, color=c_stable)
    
    # Axis formatting
    fig_1a.set_xlim([0, x_vals[-1]])
    fig_1a.set_ylim([5, 15])
    fig_1a.set_xlabel(r'$10^6$ time steps', fontsize=letter_size)
    fig_1a.set_ylabel('Active Connections (%)', fontsize=letter_size)
    
    # Set x-ticks in millions
    xticks_vals = np.arange(0, x_vals[-1] + 1e6, 1e6)
    fig_1a.set_xticks(xticks_vals)
    fig_1a.set_xticklabels([str(int(x/1e6)) for x in xticks_vals])
    
    fig_1a.set_yticks([0, 10, 15, 30])
    fig_1a.set_yticklabels(['0%', '10%', '15%', '30%'])

    fig_1a.grid(True, alpha=0.3)
    fig_1a.spines['right'].set_visible(False)
    fig_1a.spines['top'].set_visible(False)
    
    # Panel B: Avalanche Definition
    fig_1b = plt.subplot(gs[1])
    
    # Get last 150 steps of activity
    plot_last_steps = 150
    activity = history['activity_E'][-plot_last_steps:]
    
    # Threshold
    theta = 10
    boundary = theta * np.ones(plot_last_steps)
    
    # Plot activity and threshold
    fig_1b.plot(activity, 'k', label='network activity', linewidth=line_width)
    fig_1b.plot(boundary, '--k', label=r'$\theta$', linewidth=line_width)
    
    # Fill avalanches
    fig_1b.fill_between(np.arange(plot_last_steps), activity, boundary,
                        alpha=0.5, where=activity >= boundary, 
                        facecolor=c_size, interpolate=True)
    
    # Annotations
    fig_1b.text(20, 45, 'avalanches', fontsize=letter_size, color='k')
    fig_1b.text(70, 4, 'duration', fontsize=letter_size, color=c_duration)
    fig_1b.text(80, 12, 'size', fontsize=letter_size, color=c_size)
    fig_1b.text(62, -4, '100 time steps', fontsize=letter_size, color='k')
    
    # Duration and time indicators
    fig_1b.plot([58, 122], [8, 8], c_duration, linewidth=2.0)
    fig_1b.plot([50, 150], [0, 0], 'k', linewidth=2.5)
    
    # Add arrows
    arrow1 = patches.FancyArrowPatch((35, 44), (12, 29), arrowstyle='-|>',
                                     fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow1)
    arrow2 = patches.FancyArrowPatch((55, 44), (40, 35), arrowstyle='-|>',
                                     fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow2)
    arrow3 = patches.FancyArrowPatch((65, 44), (75, 38), arrowstyle='-|>',
                                     fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow3)
    
    # Axis formatting
    fig_1b.set_xlim([0, plot_last_steps])
    fig_1b.set_ylim([0, 50])
    fig_1b.set_ylabel(r'$a(t)$ [# neurons]', fontsize=letter_size)
    fig_1b.set_yticks([0, theta, 20, 40])
    fig_1b.set_yticklabels(['0', r'$\theta$', '20', '40'])
    
    fig_1b.spines['right'].set_visible(False)
    fig_1b.spines['top'].set_visible(False)
    fig_1b.spines['bottom'].set_visible(False)
    fig_1b.xaxis.set_visible(False)
    
    # Panel labels
    fig.text(0.01, 0.9, "A", weight="bold", fontsize=16,
             horizontalalignment='left', verticalalignment='center')
    fig.text(0.55, 0.9, "B", weight="bold", fontsize=16,
             horizontalalignment='left', verticalalignment='center')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.17, wspace=0.4)
    
    # Save figure
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved to {save_path}")
    plt.close()


# Main execution
if __name__ == "__main__":
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Create SORN network
    print("Creating SORN network...")
    sorn = SORN(N_E=200, N_I=40, N_U=1)
    
    # Run simulation
    print("\nRunning simulation...")
    steps = 3000000  # 3 million steps as in your code
    history = sorn.simulate(steps, record_interval=1000)
    
    print(f"\nSimulation complete!")
    print(f"Average excitatory activity: {history['activity_E'].mean():.3f}")
    print(f"Average inhibitory activity: {history['activity_I'].mean():.3f}")
    print(f"Final average threshold: {sorn.T_E.mean():.3f}")
    print(f"Final connection fraction: {history['connection_fraction'][-1]:.3f}")
    
    # Create and save raster plots at different time points
    print("\nCreating raster plots...")

    # Save spiking data
    print("\nSaving spiking data...")
    save_spiking_data(history, "plots/spiking_data.h5")
    
    # Early phase raster
    plot_raster(history, start_time=100000, duration=10000, 
                save_path="plots/raster_early.png")
    
    # Middle phase raster
    plot_raster(history, start_time=steps//2, duration=10000, 
                save_path="plots/raster_middle.png")
    
    # Late phase raster (stable)
    plot_raster(history, start_time=steps-100000, duration=10000, 
                save_path="plots/raster_late.png")
    
    # Create and save Figure 1
    print("\nCreating Figure 1...")
    plot_figure1(history, save_path="plots/figure1.pdf")
    
    print("\nAll plots saved to 'plots' directory!")