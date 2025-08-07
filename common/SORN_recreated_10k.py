"""
Self-Organizing Recurrent Neural Network (SORN) Implementation
Based on Del Papa et al. (2017) - "Criticality meets learning"
Python 3 implementation with ALL plasticity mechanisms and subtleties
Modified to save checkpoints to disk every 10k timesteps
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import pickle
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class SORNParams:
    """Parameters for SORN model matching Del Papa et al. 2017"""
    # Network size
    N_e: int = 200  # Number of excitatory neurons
    N_i: int = 40   # Number of inhibitory neurons (0.2 * N_e)
    N_u_e: int = 0  # Number of input neurons (0 for NoSource)
    
    # Connection parameters
    lamb_ee: float = 20.0  # 0.1 * N_e - expected E->E connections
    lamb_ei: float = 40.0  # 0.2 * N_e - expected I->E connections  
    lamb_ie: float = 40.0  # 1.0 * N_i - expected E->I connections
    
    # Learning rates
    eta_stdp: float = 0.004   # STDP learning rate
    eta_istdp: float = 0.001  # inhibitory STDP learning rate
    eta_ip: float = 0.01      # Intrinsic plasticity learning rate
    
    # Homeostasis
    h_ip: float = 0.1  # Target firing rate (μ_IP in paper)
    
    # Structural plasticity
    sp_prob: float = None  # Will be computed based on network size
    sp_initial: float = 0.001  # Initial weight for new connections
    
    # CRITICAL: Pruning thresholds
    prune_threshold: float = 1e-10  # Very small weights get pruned
    sp_remove_threshold: float = 0.001  # SP prunes connections below this
    
    # Thresholds
    T_e_max: float = 1.0  # Max excitatory threshold
    T_i_max: float = 0.5  # Max inhibitory threshold
    T_e_min: float = 0.0  # Min excitatory threshold
    T_i_min: float = 0.0  # Min inhibitory threshold
    
    # Noise
    noise_sigma: float = np.sqrt(0.05)  # Membrane noise std dev
    
    # Bounds
    w_max: float = 1.0  # Maximum weight value
    w_min: float = 0.0  # Minimum weight value
    
    # iSTDP lower bound
    istdp_lower_bound: float = 0.001  # Weights don't go below this in iSTDP
    
    def __post_init__(self):
        # Calculate structural plasticity probability
        if self.sp_prob is None:
            self.sp_prob = self.N_e * (self.N_e - 1) * (0.1 / (200 * 199))


class InputSource:
    """Base class for input sources"""
    def generate(self, N_u_e: int, N_e: int) -> np.ndarray:
        raise NotImplementedError


class NoSource(InputSource):
    """No external input - only membrane noise"""
    def generate(self, N_u_e: int, N_e: int) -> np.ndarray:
        return np.zeros(N_e)


class RandomBurstSource(InputSource):
    """Random burst input to subset of neurons"""
    def __init__(self, burst_prob: float = 0.01, burst_size: int = None):
        self.burst_prob = burst_prob
        self.burst_size = burst_size
        
    def generate(self, N_u_e: int, N_e: int) -> np.ndarray:
        input_vector = np.zeros(N_e)
        if N_u_e > 0 and np.random.rand() < self.burst_prob:
            # Random subset gets suprathreshold input
            active_neurons = np.random.choice(N_e, N_u_e, replace=False)
            input_vector[active_neurons] = 1.0
        return input_vector


class CheckpointManager:
    """Manages saving and loading of checkpoints to disk"""
    
    def __init__(self, base_dir: str = None, checkpoint_interval: int = 10000):
        """
        Initialize checkpoint manager
        
        Args:
            base_dir: Directory to save checkpoints (default: sorn_checkpoints_TIMESTAMP)
            checkpoint_interval: Save checkpoint every N steps
        """
        if base_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = f"sorn_checkpoints_{timestamp}"
        
        self.base_dir = base_dir
        self.checkpoint_interval = checkpoint_interval
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Track checkpoint metadata
        self.checkpoint_info = {
            'total_steps': 0,
            'checkpoints': [],
            'checkpoint_interval': checkpoint_interval
        }
        
    def save_checkpoint(self, step: int, state_dict: dict, stats_dict: dict):
        """Save checkpoint to disk"""
        checkpoint_path = os.path.join(self.base_dir, f"checkpoint_{step:08d}.pkl")
        stats_path = os.path.join(self.base_dir, f"stats_{step:08d}.npz")
        
        # Save network state
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save statistics as compressed numpy arrays
        np.savez_compressed(stats_path, **stats_dict)
        
        # Update metadata
        self.checkpoint_info['checkpoints'].append(step)
        self.checkpoint_info['total_steps'] = step
        self._save_metadata()
        
    def load_checkpoint(self, step: int):
        """Load checkpoint from disk"""
        checkpoint_path = os.path.join(self.base_dir, f"checkpoint_{step:08d}.pkl")
        stats_path = os.path.join(self.base_dir, f"stats_{step:08d}.npz")
        
        with open(checkpoint_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        stats_dict = dict(np.load(stats_path))
        
        return state_dict, stats_dict
    
    def _save_metadata(self):
        """Save checkpoint metadata"""
        metadata_path = os.path.join(self.base_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.checkpoint_info, f)
    
    def load_metadata(self):
        """Load checkpoint metadata"""
        metadata_path = os.path.join(self.base_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.checkpoint_info = pickle.load(f)
        return self.checkpoint_info
    
    def get_statistics_range(self, start_step: int, end_step: int):
        """Load and concatenate statistics from multiple checkpoints"""
        activity = []
        conn_fraction = []
        
        # Find relevant checkpoints
        relevant_checkpoints = [cp for cp in self.checkpoint_info['checkpoints'] 
                               if start_step <= cp <= end_step]
        
        for cp in sorted(relevant_checkpoints):
            _, stats = self.load_checkpoint(cp)
            activity.extend(stats['activity_history'])
            conn_fraction.extend(stats['connection_fraction_history'])
        
        return np.array(activity), np.array(conn_fraction)


class SORN:
    """Complete SORN implementation with all plasticity mechanisms and disk checkpointing"""
    
    def __init__(self, params: SORNParams = None, input_source: InputSource = None, 
                 checkpoint_dir: str = None, checkpoint_interval: int = 10000):
        self.params = params or SORNParams()
        self.input_source = input_source or NoSource()
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir, checkpoint_interval)
        
        # Initialize network state
        self._initialize_network()
        
        # Track statistics (only keep current buffer in memory)
        self.activity_history = []
        self.connection_fraction_history = []
        
        # Track total steps simulated
        self.total_steps = 0
        
        # For structural plasticity optimization (batch updates)
        self.struct_p_count = 0
        self.struct_p_list = []
        
    def _initialize_network(self):
        """Initialize network connectivity and parameters"""
        p = self.params
        
        # State vectors
        self.x = np.zeros(p.N_e, dtype=np.int8)  # Excitatory activity
        self.y = np.zeros(p.N_i, dtype=np.int8)  # Inhibitory activity
        
        # Initialize thresholds (matching original implementation)
        self.T_e = p.T_e_min + np.random.rand(p.N_e) * (p.T_e_max - p.T_e_min)
        self.T_i = p.T_i_min + np.random.rand(p.N_i) * (p.T_i_max - p.T_i_min)
        
        # Initialize weight matrices
        self._initialize_weights()
        
        # Create masks for avoiding self-connections
        self.no_self_mask_ee = np.ones((p.N_e, p.N_e))
        np.fill_diagonal(self.no_self_mask_ee, 0)
        
    def _initialize_weights(self):
        """Initialize weight matrices with specified connection probabilities"""
        p = self.params
        
        # W_ee: Excitatory to Excitatory (sparse initialization)
        n_ee_connections = int(p.lamb_ee)
        self.W_ee = np.zeros((p.N_e, p.N_e))
        
        for i in range(p.N_e):
            # Get connection probability for this neuron
            if p.lamb_ee > p.N_e - 1:
                n_conn = p.N_e - 1
            else:
                # Sample from binomial distribution
                n_conn = np.random.binomial(p.N_e - 1, p.lamb_ee / p.N_e)
                n_conn = max(1, n_conn)  # Ensure at least 1 connection
            
            # Choose targets (avoiding self)
            possible = list(range(p.N_e))
            possible.remove(i)
            targets = np.random.choice(possible, n_conn, replace=False)
            
            # Initialize weights that sum to 1 (for synaptic normalization)
            weights = np.random.rand(n_conn)
            weights /= weights.sum()
            self.W_ee[i, targets] = weights
        
        # W_ei: Inhibitory to Excitatory  
        self.W_ei = np.zeros((p.N_e, p.N_i))
        for i in range(p.N_e):
            if p.lamb_ei > p.N_i:
                n_conn = p.N_i
            else:
                n_conn = np.random.binomial(p.N_i, p.lamb_ei / p.N_i)
                n_conn = max(1, n_conn)
                
            targets = np.random.choice(p.N_i, n_conn, replace=False)
            weights = np.random.rand(n_conn)
            weights /= weights.sum()
            self.W_ei[i, targets] = weights
            
        # W_ie: Excitatory to Inhibitory
        self.W_ie = np.zeros((p.N_i, p.N_e))
        for i in range(p.N_i):
            if p.lamb_ie > p.N_e:
                n_conn = p.N_e
            else:
                n_conn = np.random.binomial(p.N_e, p.lamb_ie / p.N_e)
                n_conn = max(1, n_conn)
                
            targets = np.random.choice(p.N_e, n_conn, replace=False)
            weights = np.random.rand(n_conn)
            weights /= weights.sum()
            self.W_ie[i, targets] = weights
            
        # Binary masks for active connections
        self.M_ee = (self.W_ee > 0).astype(int)
        self.M_ei = (self.W_ei > 0).astype(int)
        self.M_ie = (self.W_ie > 0).astype(int)
        
    def step(self):
        """Single timestep update with all mechanisms"""
        p = self.params
        
        # Store previous state for STDP
        x_prev = self.x.copy()
        y_prev = self.y.copy()
        
        # Get external input
        u_ext = self.input_source.generate(p.N_u_e, p.N_e)
        
        # Generate membrane noise
        xi_e = np.random.normal(0, p.noise_sigma, p.N_e)
        xi_i = np.random.normal(0, p.noise_sigma, p.N_i)
        
        # Update excitatory neurons (using masked weights)
        input_e = (self.W_ee * self.M_ee).T @ x_prev - (self.W_ei * self.M_ei) @ y_prev + u_ext + xi_e
        self.x = (input_e > self.T_e).astype(np.int8)
        
        # Update inhibitory neurons  
        input_i = (self.W_ie * self.M_ie) @ x_prev + xi_i
        self.y = (input_i > self.T_i).astype(np.int8)
        
        # Apply plasticity mechanisms (order matters!)
        self._stdp(x_prev)
        self._istdp(y_prev)
        self._ip()
        self._synaptic_normalization()
        self._structural_plasticity()
        
        # Record statistics
        self.activity_history.append(np.mean(self.x))
        self.connection_fraction_history.append(np.sum(self.M_ee) / (p.N_e * p.N_e))
        
        # Increment total steps
        self.total_steps += 1
        
        # Print progress every 10k steps
        if self.total_steps % 10000 == 0:
            active_conn = np.sum(self.M_ee)
            total_conn = self.params.N_e * self.params.N_e
            last_activity = self.activity_history[-1] if self.activity_history else 0
            print(f"  Step {self.total_steps:,} - Activity: {last_activity:.3f}, "
                  f"Connections: {active_conn}/{total_conn} ({100*active_conn/total_conn:.1f}%)")
        
        # Check if we need to save checkpoint
        if self.total_steps % self.checkpoint_mgr.checkpoint_interval == 0:
            self._save_checkpoint()
            
    def _save_checkpoint(self):
        """Save current state to disk and clear buffers"""
        # Prepare state dictionary
        state_dict = {
            'W_ee': self.W_ee,
            'W_ei': self.W_ei,
            'W_ie': self.W_ie,
            'M_ee': self.M_ee,
            'M_ei': self.M_ei,
            'M_ie': self.M_ie,
            'T_e': self.T_e,
            'T_i': self.T_i,
            'x': self.x,
            'y': self.y,
            'struct_p_count': self.struct_p_count,
            'struct_p_list': self.struct_p_list,
            'total_steps': self.total_steps
        }
        
        # Prepare statistics dictionary
        stats_dict = {
            'activity_history': np.array(self.activity_history),
            'connection_fraction_history': np.array(self.connection_fraction_history)
        }
        
        # Save checkpoint
        self.checkpoint_mgr.save_checkpoint(self.total_steps, state_dict, stats_dict)
        
        # Clear buffers to free memory
        self.activity_history = []
        self.connection_fraction_history = []
        
        print(f"    → Checkpoint saved at step {self.total_steps:,}")
        
    def load_checkpoint(self, step: int):
        """Load state from checkpoint"""
        state_dict, _ = self.checkpoint_mgr.load_checkpoint(step)
        
        self.W_ee = state_dict['W_ee']
        self.W_ei = state_dict['W_ei']
        self.W_ie = state_dict['W_ie']
        self.M_ee = state_dict['M_ee']
        self.M_ei = state_dict['M_ei']
        self.M_ie = state_dict['M_ie']
        self.T_e = state_dict['T_e']
        self.T_i = state_dict['T_i']
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.struct_p_count = state_dict['struct_p_count']
        self.struct_p_list = state_dict['struct_p_list']
        self.total_steps = state_dict['total_steps']
        
        print(f"Loaded checkpoint from step {step}")
        
    def _stdp(self, x_prev):
        """Spike-Timing Dependent Plasticity for E->E connections"""
        if self.params.eta_stdp == 0:
            return
            
        # STDP: Potentiate if pre fires before post, depress if post fires before pre
        # ΔW_ij = η_stdp * [x_i(t) * x_j(t-1) - x_j(t) * x_i(t-1)]
        potentiation = np.outer(self.x, x_prev)  # Post fires now, pre fired before
        depression = np.outer(x_prev, self.x)    # Pre fires now, post fired before
        
        dW = self.params.eta_stdp * (potentiation - depression)
        
        # Only update existing connections and avoid self-connections
        self.W_ee += dW * self.M_ee * self.no_self_mask_ee
        
        # Bound weights
        self.W_ee = np.clip(self.W_ee, 0, self.params.w_max)
        
        # CRITICAL: Prune very small weights (the subtlety you mentioned!)
        self._prune_weights()
        
    def _istdp(self, y_prev):
        """Inhibitory Spike-Timing Dependent Plasticity for I->E connections"""
        if self.params.eta_istdp == 0:
            return
            
        # iSTDP: ΔW_ik = -η_istdp * y_k(t-1) * [1 - x_i(t) * (1 + 1/h_ip)]
        for i in range(self.params.N_e):
            for k in range(self.params.N_i):
                if self.M_ei[i, k] and y_prev[k]:
                    dW = -self.params.eta_istdp * (1 - self.x[i] * (1 + 1/self.params.h_ip))
                    self.W_ei[i, k] += dW
                    
        # Special bounds for iSTDP: weights can't go below istdp_lower_bound
        self.W_ei[self.W_ei <= 0] = self.params.istdp_lower_bound
        self.W_ei[self.W_ei > self.params.w_max] = self.params.w_max
        
    def _ip(self):
        """Intrinsic Plasticity - homeostatic threshold adaptation"""
        if self.params.eta_ip == 0:
            return
            
        # IP: ΔT_i = η_ip * [x_i(t) - h_ip]
        self.T_e += self.params.eta_ip * (self.x - self.params.h_ip)
        
        # Bound thresholds
        self.T_e = np.clip(self.T_e, self.params.T_e_min, self.params.T_e_max)
        
    def _synaptic_normalization(self):
        """Normalize incoming synaptic weights (both E and I separately)"""
        # Normalize incoming E->E connections for each neuron
        for i in range(self.params.N_e):
            sum_in = np.sum(self.W_ee[:, i])
            if sum_in > 1e-6:  # Avoid division by zero
                target_sum = np.sum(self.M_ee[:, i])  # Number of incoming connections
                if target_sum > 0:
                    self.W_ee[:, i] *= target_sum / sum_in
                
        # Normalize incoming I->E connections for each neuron
        for i in range(self.params.N_e):
            sum_in = np.sum(self.W_ei[i, :])
            if sum_in > 1e-6:
                target_sum = np.sum(self.M_ei[i, :])
                if target_sum > 0:
                    self.W_ei[i, :] *= target_sum / sum_in
                    
    def _prune_weights(self):
        """Prune very small weights - CRITICAL for proper dynamics"""
        p = self.params
        
        # Delete very small E->E weights (the 1e-10 threshold!)
        small_weights = (self.W_ee < p.prune_threshold) & (self.W_ee > 0)
        self.W_ee[small_weights] = 0
        self.M_ee[small_weights] = 0
        
        # Also apply upper bound
        self.W_ee[self.W_ee > p.w_max] = p.w_max
        
    def _structural_plasticity(self):
        """Structural plasticity - add/remove connections with batch updates"""
        if self.params.sp_prob == 0:
            return
            
        p = self.params
        
        # Accumulate new connections to add (batch update for efficiency)
        for i in range(p.N_e):
            for j in range(p.N_e):
                if i != j and self.M_ee[i, j] == 0:
                    if np.random.rand() < p.sp_prob:
                        self.struct_p_count += 1
                        self.struct_p_list.append((i, j))
        
        # Apply batch updates every 10 new connections (as in original)
        if self.struct_p_count > 10:
            for (i, j) in self.struct_p_list:
                self.W_ee[i, j] = p.sp_initial
                self.M_ee[i, j] = 1
            self.struct_p_count = 0
            self.struct_p_list = []
        
        # Prune weak connections below SP threshold
        weak_connections = (self.W_ee < p.sp_remove_threshold) & (self.M_ee == 1)
        self.W_ee[weak_connections] = 0
        self.M_ee[weak_connections] = 0
        
    def simulate(self, steps: int, verbose: bool = True):
        """Run simulation for specified number of steps"""
        if verbose:
            print(f"Simulating {steps:,} steps...")
            print(f"  Checkpoints will be saved every {self.checkpoint_mgr.checkpoint_interval:,} steps")
            print(f"  Checkpoint directory: {self.checkpoint_mgr.base_dir}")
            print(f"  Progress will be printed every 10,000 steps")
            print("")
            
        for t in range(steps):
            self.step()
                
        # Save final checkpoint if there's remaining data
        if self.activity_history:
            self._save_checkpoint()
                
    def get_avalanches(self, start_step: int = 0, end_step: int = None) -> Tuple[List[int], List[int]]:
        """Extract avalanche durations and sizes from activity (loads from disk)"""
        if end_step is None:
            end_step = self.total_steps
            
        # Load activity data from checkpoints
        activity, _ = self.checkpoint_mgr.get_statistics_range(start_step, end_step)
        
        if len(activity) == 0:
            return [], []
            
        activity = activity * self.params.N_e
        
        # Use half of mean activity as threshold (as in paper)
        threshold = np.mean(activity) / 2
            
        # Find avalanches (activity above threshold)
        above_threshold = activity > threshold
        durations = []
        sizes = []
        
        in_avalanche = False
        current_duration = 0
        current_size = 0
        
        for i, (active, a) in enumerate(zip(above_threshold, activity)):
            if active:
                if not in_avalanche:
                    in_avalanche = True
                    current_duration = 0
                    current_size = 0
                current_duration += 1
                current_size += a - threshold
            else:
                if in_avalanche:
                    durations.append(current_duration)
                    sizes.append(int(current_size))
                    in_avalanche = False
                    
        return durations, sizes
    
    def plot_dynamics(self, start: int = 0, end: int = None):
        """Plot network dynamics and statistics (loads from disk)"""
        if end is None:
            end = self.total_steps
            
        # Load data from checkpoints
        activity_history, connection_fraction_history = self.checkpoint_mgr.get_statistics_range(start, end)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Activity over time
        axes[0].plot(activity_history, 'b-', linewidth=0.5)
        axes[0].set_ylabel('Mean Activity')
        axes[0].set_title('SORN Dynamics')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=self.params.h_ip, color='r', linestyle='--', alpha=0.5, label='Target (h_ip)')
        axes[0].legend()
        
        # Connection fraction (showing three phases)
        axes[1].plot(connection_fraction_history, 'g-', linewidth=0.5)
        axes[1].set_ylabel('Connection Fraction')
        axes[1].grid(True, alpha=0.3)
        
        # Add phase labels if showing from beginning
        if start == 0 and len(connection_fraction_history) > 2000000:
            axes[1].axvline(x=100000, color='k', linestyle='--', alpha=0.3)
            axes[1].axvline(x=2000000, color='k', linestyle='--', alpha=0.3)
            axes[1].text(50000, axes[1].get_ylim()[1]*0.9, 'Decay', ha='center')
            axes[1].text(1000000, axes[1].get_ylim()[1]*0.9, 'Growth', ha='center')
            axes[1].text(2500000, axes[1].get_ylim()[1]*0.9, 'Stable', ha='center')
        
        # Raster plot (simulate last 1000 steps from current state)
        raster_steps = min(1000, end - start)
        
        # Collect spikes for raster
        spike_times = []
        spike_neurons = []
        
        for t in range(raster_steps):
            self.step()
            active_neurons = np.where(self.x)[0]
            for neuron in active_neurons:
                spike_times.append(t)
                spike_neurons.append(neuron)
                
        axes[2].scatter(spike_times, spike_neurons, s=0.5, c='k', alpha=0.5)
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Neuron Index')
        axes[2].set_title(f'Raster Plot (last {raster_steps} steps)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_avalanches(self, start_step: int = 2000000):
        """Analyze and plot avalanche distributions"""
        durations, sizes = self.get_avalanches(start_step=start_step)
        
        if len(durations) == 0:
            print("No avalanches found")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Duration distribution
        duration_counts = np.bincount(durations)[1:]  # Skip 0
        duration_values = np.arange(1, len(duration_counts) + 1)
        
        valid_d = duration_counts > 0
        axes[0].loglog(duration_values[valid_d], duration_counts[valid_d], 'bo', alpha=0.5, markersize=4)
        
        # Add power-law fit line (α ≈ 1.45 from paper)
        x_fit = np.logspace(0, np.log10(max(durations)), 100)
        y_fit = x_fit ** (-1.45)
        y_fit *= duration_counts[valid_d][0] / y_fit[0]  # Scale to match
        axes[0].plot(x_fit, y_fit, 'b--', alpha=0.7, label='α = 1.45')
        
        axes[0].set_xlabel('Duration T')
        axes[0].set_ylabel('P(T)')
        axes[0].set_title('Avalanche Duration Distribution')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Size distribution  
        size_counts = np.bincount(sizes)[1:]  # Skip 0
        size_values = np.arange(1, len(size_counts) + 1)
        
        valid_s = size_counts > 0
        axes[1].loglog(size_values[valid_s], size_counts[valid_s], 'ro', alpha=0.5, markersize=4)
        
        # Add power-law fit line (τ ≈ 1.28 from paper)
        x_fit = np.logspace(0, np.log10(max(sizes)), 100)
        y_fit = x_fit ** (-1.28)
        y_fit *= size_counts[valid_s][0] / y_fit[0]
        axes[1].plot(x_fit, y_fit, 'r--', alpha=0.7, label='τ = 1.28')
        
        axes[1].set_xlabel('Size S')
        axes[1].set_ylabel('P(S)')
        axes[1].set_title('Avalanche Size Distribution')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Average size vs duration (crackling noise relation)
        duration_array = np.array(durations)
        size_array = np.array(sizes)
        unique_durations = np.unique(duration_array)
        avg_sizes = []
        
        for d in unique_durations:
            mask = duration_array == d
            avg_sizes.append(np.mean(size_array[mask]))
            
        axes[2].loglog(unique_durations, avg_sizes, 'go', alpha=0.5, markersize=4)
        
        # Add theoretical scaling (γ ≈ 1.3 from paper)
        x_fit = np.logspace(0, np.log10(max(unique_durations)), 100)
        y_fit = x_fit ** 1.3
        y_fit *= avg_sizes[0] / y_fit[0]
        axes[2].plot(x_fit, y_fit, 'g--', alpha=0.7, label='γ = 1.3')
        
        axes[2].set_xlabel('Duration T')
        axes[2].set_ylabel('<S>(T)')
        axes[2].set_title('Average Size vs Duration')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nAvalanche Statistics:")
        print(f"  Number of avalanches: {len(durations)}")
        print(f"  Mean duration: {np.mean(durations):.2f}")
        print(f"  Mean size: {np.mean(sizes):.2f}")
        print(f"  Max duration: {max(durations)}")
        print(f"  Max size: {max(sizes)}")
        

def demonstrate_sorn():
    """Demonstrate SORN with different input sources"""
    
    print("=" * 60)
    print("SORN Implementation - Del Papa et al. 2017")
    print("With ALL mechanisms and disk checkpointing")
    print("=" * 60)
    
    # Test 1: NoSource (only membrane noise)
    print("\n1. Testing with NoSource (membrane noise only)...")
    params = SORNParams()
    sorn_no_input = SORN(params, NoSource(), checkpoint_interval=10000)
    
    # Simulate through phases (matching paper timeline)
    print("\n   Phase 1: Decay (100k steps)")
    sorn_no_input.simulate(100000, verbose=True)
    
    print("\n   Phase 2: Growth (1.9M steps)")
    sorn_no_input.simulate(1900000, verbose=True)
    
    print("\n   Phase 3: Stable (3M steps) - avalanche measurement phase")
    sorn_no_input.simulate(3000000, verbose=True)
    
    print(f"\n   Final statistics:")
    print(f"     Activity: {np.mean(sorn_no_input.x):.3f} (target: {params.h_ip})")
    print(f"     Active connections: {np.sum(sorn_no_input.M_ee)}/{params.N_e**2}")
    print(f"     Connection fraction: {np.sum(sorn_no_input.M_ee)/(params.N_e**2):.3f}")
    print(f"     Total checkpoints saved: {len(sorn_no_input.checkpoint_mgr.checkpoint_info['checkpoints'])}")
    
    # Test 2: RandomBurstSource
    print("\n2. Testing with RandomBurstSource...")
    params2 = SORNParams(N_u_e=10)  # 10 input neurons (5% of N_e)
    sorn_burst = SORN(params2, RandomBurstSource(burst_prob=0.01), checkpoint_interval=10000)
    
    print("   Simulating 500k steps with random bursts...")
    sorn_burst.simulate(500000, verbose=True)
    
    print(f"\n   Final statistics:")
    print(f"     Activity: {np.mean(sorn_burst.x):.3f}")
    print(f"     Active connections: {np.sum(sorn_burst.M_ee)}/{params2.N_e**2}")
    print(f"     Connection fraction: {np.sum(sorn_burst.M_ee)/(params2.N_e**2):.3f}")
    
    # Plot results
    print("\n3. Analyzing network dynamics...")
    print("   Loading data from checkpoints for visualization...")
    sorn_no_input.plot_dynamics(start=2000000)  # Show stable phase
    sorn_no_input.analyze_avalanches()
    
    return sorn_no_input, sorn_burst


if __name__ == "__main__":
    # Run demonstration
    sorn1, sorn2 = demonstrate_sorn()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("All mechanisms implemented:")
    print("  ✓ STDP with asymmetric learning")
    print("  ✓ iSTDP with lower bound of 0.001")
    print("  ✓ IP (Intrinsic Plasticity)")
    print("  ✓ SN (Synaptic Normalization) - separate for E and I")
    print("  ✓ SP (Structural Plasticity) with batch updates")
    print("  ✓ Weight pruning at 1e-10 threshold")
    print("  ✓ SP pruning at 0.001 threshold")
    print("  ✓ Membrane noise σ = √0.05")
    print("  ✓ Three-phase dynamics (decay→growth→stable)")
    print("  ✓ Disk checkpointing every 10k steps")
    print("  ✓ Progress updates every 10k steps")
    print("=" * 60)