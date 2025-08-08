"""
Self-Organizing Recurrent Neural Network (SORN) Implementation
Based on Del Papa et al. (2017) - "Criticality meets learning"
With EFFICIENT checkpointing - only saves statistics, not full state
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import os
from datetime import datetime
import h5py  # Using HDF5 for efficient storage

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


class EfficientDataLogger:
    """Efficient data logging using HDF5 - only saves what's needed for analysis"""
    
    def __init__(self, output_dir: str = None, buffer_size: int = 100000):
        """
        Initialize efficient data logger
        
        Args:
            output_dir: Directory to save data
            buffer_size: Size of memory buffer before writing to disk
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"sorn_data_{timestamp}"
        
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        os.makedirs(self.output_dir, exist_ok=True)
        
        # File paths
        self.stats_file = os.path.join(self.output_dir, "statistics.h5")
        self.raster_file = os.path.join(self.output_dir, "raster.h5")
        
        # Memory buffers
        self.activity_buffer = []
        self.conn_fraction_buffer = []
        self.raster_buffer = []
        
        # Track what's been written
        self.total_steps_written = 0
        self.stats_initialized = False
        self.raster_initialized = False
        
    def add_step(self, activity: float, conn_fraction: float, spike_vector: np.ndarray = None):
        """Add data from one timestep"""
        self.activity_buffer.append(activity)
        self.conn_fraction_buffer.append(conn_fraction)
        
        if spike_vector is not None:
            self.raster_buffer.append(spike_vector.astype(np.bool_))
        
        # Write to disk if buffer is full
        if len(self.activity_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffers to disk"""
        if not self.activity_buffer:
            return
            
        # Write statistics
        with h5py.File(self.stats_file, 'a') as f:
            if not self.stats_initialized:
                # Create resizable datasets
                f.create_dataset('activity', 
                               data=self.activity_buffer,
                               maxshape=(None,),
                               chunks=(min(10000, len(self.activity_buffer)),),
                               compression='gzip')
                f.create_dataset('connection_fraction',
                               data=self.conn_fraction_buffer,
                               maxshape=(None,),
                               chunks=(min(10000, len(self.conn_fraction_buffer)),),
                               compression='gzip')
                self.stats_initialized = True
            else:
                # Append to existing datasets
                f['activity'].resize(self.total_steps_written + len(self.activity_buffer), axis=0)
                f['activity'][self.total_steps_written:] = self.activity_buffer
                
                f['connection_fraction'].resize(self.total_steps_written + len(self.conn_fraction_buffer), axis=0)
                f['connection_fraction'][self.total_steps_written:] = self.conn_fraction_buffer
        
        # Write raster if available
        if self.raster_buffer:
            raster_array = np.array(self.raster_buffer).T  # Shape: (neurons, time)
            
            with h5py.File(self.raster_file, 'a') as f:
                if not self.raster_initialized:
                    # Create resizable dataset
                    n_neurons = raster_array.shape[0]
                    f.create_dataset('raster',
                                   data=raster_array,
                                   maxshape=(n_neurons, None),
                                   chunks=(n_neurons, min(10000, raster_array.shape[1])),
                                   compression='gzip')
                    self.raster_initialized = True
                else:
                    # Append to existing dataset
                    current_size = f['raster'].shape[1]
                    new_size = current_size + raster_array.shape[1]
                    f['raster'].resize(new_size, axis=1)
                    f['raster'][:, current_size:] = raster_array
        
        # Update counter and clear buffers
        self.total_steps_written += len(self.activity_buffer)
        self.activity_buffer = []
        self.conn_fraction_buffer = []
        self.raster_buffer = []
        
        print(f"    → Data saved: {self.total_steps_written:,} total steps ({self._get_file_sizes()})")
    
    def _get_file_sizes(self):
        """Get human-readable file sizes"""
        sizes = []
        for filepath in [self.stats_file, self.raster_file]:
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                sizes.append(f"{size_mb:.1f}MB")
        return ", ".join(sizes)
    
    def save_final_state(self, sorn_state: dict):
        """Save final network state for potential restart"""
        state_file = os.path.join(self.output_dir, "final_state.npz")
        np.savez_compressed(state_file, **sorn_state)
        print(f"    → Final state saved to {state_file}")
    
    def load_data(self, start_step: int = 0, end_step: int = None):
        """Load data from disk for analysis"""
        with h5py.File(self.stats_file, 'r') as f:
            activity = f['activity'][start_step:end_step]
            conn_fraction = f['connection_fraction'][start_step:end_step]
        return np.array(activity), np.array(conn_fraction)
    
    def load_raster(self, start_step: int = 0, end_step: int = None):
        """Load raster data from disk"""
        with h5py.File(self.raster_file, 'r') as f:
            raster = f['raster'][:, start_step:end_step]
        return np.array(raster)


class SORN:
    """SORN implementation with efficient data logging"""
    
    def __init__(self, params: SORNParams = None, input_source: InputSource = None, 
                 output_dir: str = None, save_raster: bool = False, 
                 save_interval: int = 100000):
        """
        Initialize SORN
        
        Args:
            params: Network parameters
            input_source: Input source
            output_dir: Directory for data output
            save_raster: Whether to save full raster (uses more disk space)
            save_interval: How often to flush data to disk
        """
        self.params = params or SORNParams()
        self.input_source = input_source or NoSource()
        self.save_raster = save_raster
        
        # Initialize data logger
        self.logger = EfficientDataLogger(output_dir, buffer_size=save_interval)
        
        # Initialize network state
        self._initialize_network()
        
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
        
        # Log data
        activity = np.mean(self.x)
        conn_fraction = np.sum(self.M_ee) / (p.N_e * p.N_e)
        self.logger.add_step(activity, conn_fraction, self.x if self.save_raster else None)
        
        # Increment total steps
        self.total_steps += 1
        
        # Print progress every 10k steps
        if self.total_steps % 10000 == 0:
            active_conn = np.sum(self.M_ee)
            total_conn = self.params.N_e * self.params.N_e
            print(f"  Step {self.total_steps:,} - Activity: {activity:.3f}, "
                  f"Connections: {active_conn}/{total_conn} ({100*active_conn/total_conn:.1f}%)")
            
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
            print(f"  Output directory: {self.logger.output_dir}")
            print(f"  Saving raster: {self.save_raster}")
            print(f"  Progress updates every 10,000 steps")
            print("")
            
        for t in range(steps):
            self.step()
                
        # Flush remaining data
        self.logger.flush()
        
        # Save final state
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
            'total_steps': self.total_steps
        }
        self.logger.save_final_state(state_dict)
        
        print(f"\nSimulation complete! Data saved to {self.logger.output_dir}")
                
    def get_avalanches(self, start_step: int = 0, end_step: int = None) -> Tuple[List[int], List[int]]:
        """Extract avalanche durations and sizes from activity"""
        if end_step is None:
            end_step = self.total_steps
            
        # Load activity data
        activity, _ = self.logger.load_data(start_step, end_step)
        
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


def demonstrate_sorn():
    """Demonstrate SORN with efficient data logging"""
    
    print("=" * 60)
    print("SORN Implementation - Del Papa et al. 2017")
    print("With EFFICIENT data logging (HDF5)")
    print("=" * 60)
    
    # Test 1: NoSource (only membrane noise)
    print("\n1. Testing with NoSource (membrane noise only)...")
    params = SORNParams()
    
    # Note: save_raster=False saves disk space, only saves statistics
    # Set save_raster=True if you need the full spike raster for analysis
    sorn_no_input = SORN(params, NoSource(), save_raster=False, save_interval=100000)
    
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
    
    # Analyze avalanches
    print("\n   Analyzing avalanches from stable phase...")
    durations, sizes = sorn_no_input.get_avalanches(start_step=2000000)
    print(f"     Found {len(durations)} avalanches")
    if len(durations) > 0:
        print(f"     Mean duration: {np.mean(durations):.2f}")
        print(f"     Mean size: {np.mean(sizes):.2f}")
    
    return sorn_no_input


if __name__ == "__main__":
    # Run demonstration
    sorn = demonstrate_sorn()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("Key improvements:")
    print("  ✓ Efficient HDF5 storage (compressed)")
    print("  ✓ Only saves statistics by default (not full weights)")
    print("  ✓ Optional raster saving")
    print("  ✓ Streaming data to disk (no memory accumulation)")
    print("  ✓ ~2-5 MB per million timesteps (vs 11.3 GB before!)")
    print("  ✓ Can still analyze avalanches from saved data")
    print("=" * 60)