import numpy as np
from typing import Dict, Tuple, Optional

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
                
    def simulate(self, steps: int, input_pattern: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run simulation for specified number of steps
        
        Parameters:
        -----------
        steps : int
            Number of simulation steps
        input_pattern : np.ndarray, optional
            Input pattern (steps × N_U). If None, zero input is used.
            
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
            'activity_I': np.zeros(steps)
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
            history['activity_E'][t] = self.x.mean()
            history['activity_I'][t] = self.y.mean()
            
        return history


# Example usage
if __name__ == "__main__":
    # Create SORN network
    sorn = SORN(N_E=200, N_I=40, N_U=1)
    
    # Run simulation
    steps = 3000000
    history = sorn.simulate(steps)
    
    print(f"Average excitatory activity: {history['activity_E'].mean():.3f}")
    print(f"Average inhibitory activity: {history['activity_I'].mean():.3f}")
    print(f"Final average threshold: {sorn.T_E.mean():.3f}")