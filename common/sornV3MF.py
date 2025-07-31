import numpy as np
from scipy.integrate import odeint
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SORNMeanFieldComplete:
    """
    Complete mean field model for SORN including:
    - Second-order moments (fluctuations)
    - Finite-size corrections
    - Self-organization to criticality
    
    Based on the full SORN implementation with eigenvalue analysis
    """
    
    def __init__(self, 
                 N_E=200,
                 N_I=40,
                 h_ip=0.1,
                 eta_stdp=0.004,
                 eta_istdp=0.001,
                 eta_ip=0.01,
                 lambda_=20,
                 noise_sig=np.sqrt(0.05),
                 T_e_min=0.0,
                 T_e_max=0.5,
                 T_i_min=0.0,
                 T_i_max=1.0,
                 W_min=0.0,
                 W_max=1.0,
                 W_ei_min=0.001):
        
        # Network parameters
        self.N_E = N_E
        self.N_I = N_I
        self.h_ip = h_ip
        
        # Learning rates with finite-size scaling
        self.eta_stdp_base = eta_stdp
        self.eta_istdp_base = eta_istdp
        self.eta_ip_base = eta_ip
        
        # Finite-size corrected learning rates
        self.eta_stdp = eta_stdp * np.sqrt(N_E / 200.0)
        self.eta_istdp = eta_istdp * np.sqrt(N_E / 200.0)
        self.eta_ip = eta_ip * (200.0 / N_E)  # IP scales inversely
        
        # Network topology
        self.lambda_ = lambda_
        self.p_ee = lambda_ / N_E
        self.p_ei = 0.2
        self.p_ie = 1.0
        
        # Structural plasticity with finite-size correction
        expected_connections = self.p_ee * N_E * (N_E - 1)
        self.p_sp_base = expected_connections / (200.0 * 199.0)
        self.p_sp = self.p_sp_base * (200.0 / N_E)**2  # Scales with 1/N²
        
        # Noise and bounds
        self.noise_sig = noise_sig
        self.T_e_min = T_e_min
        self.T_e_max = T_e_max
        self.T_i_min = T_i_min
        self.T_i_max = T_i_max
        self.W_min = W_min
        self.W_max = W_max
        self.W_ei_min = W_ei_min
        
        # Finite-size corrections
        self.finite_size_factor = 1.0 / np.sqrt(N_E)
        self.critical_rho = 1.0 - 2.0 * self.finite_size_factor  # Shifted critical point
        
    def firing_rate_with_fluctuations(self, I_mean, I_var, threshold):
        """
        Compute firing rate including fluctuations
        Uses error function approximation for Gaussian input through threshold
        """
        if I_var <= 0:
            I_var = self.noise_sig**2  # Minimum variance from intrinsic noise
            
        z = (I_mean - threshold) / np.sqrt(2 * I_var)
        return 0.5 * (1 + erf(z))
    
    def spectral_radius_dynamics(self, W_EE, C, var_E, m_E):
        """
        Approximate dynamics of the spectral radius
        Including finite-size fluctuations
        """
        # Base spectral radius approximation
        # For sparse random matrix: ρ ≈ sqrt(K) * W_EE
        K_eff = C * self.N_E
        rho_mean = np.sqrt(K_eff) * W_EE
        
        # Finite-size fluctuations in spectral radius
        # Fluctuations scale as 1/sqrt(N)
        rho_fluct = self.finite_size_factor * np.sqrt(var_E)
        
        # Near criticality, fluctuations are enhanced
        if rho_mean > 0.9:
            enhancement = 1.0 + 10.0 * (rho_mean - 0.9)**2
            rho_fluct *= enhancement
            
        return rho_mean, rho_fluct
    
    def mean_field_dynamics(self, state, t):
        """
        Complete mean field dynamics with fluctuations and finite-size effects
        
        State vector: [m_E, m_I, var_E, var_I, cov_EI, W_EE, W_EI, W_IE, T_E, T_I, C, rho]
        """
        m_E, m_I, var_E, var_I, cov_EI, W_EE, W_EI, W_IE, T_E, T_I, C, rho = state
        
        # Clip state variables to valid ranges
        m_E = np.clip(m_E, 0, 1)
        m_I = np.clip(m_I, 0, 1)
        var_E = np.clip(var_E, 0, 0.25)  # Max variance for binary variable
        var_I = np.clip(var_I, 0, 0.25)
        W_EE = np.clip(W_EE, self.W_min, self.W_max)
        W_EI = np.clip(W_EI, self.W_ei_min, self.W_max)
        W_IE = np.clip(W_IE, self.W_min, self.W_max)
        T_E = np.clip(T_E, self.T_e_min, self.T_e_max)
        T_I = np.clip(T_I, self.T_i_min, self.T_i_max)
        C = np.clip(C, 0, 1)
        
        # Effective connectivity with finite-size corrections
        K_EE = C * self.N_E * (1 - 1/self.N_E)  # Exclude self-connections
        K_EI = self.p_ei * self.N_I
        K_IE = self.p_ie * self.N_E
        
        # Mean input currents
        I_E_mean = K_EE * W_EE * m_E - K_EI * W_EI * m_I
        I_I_mean = K_IE * W_IE * m_E
        
        # Input current variances including network and intrinsic noise
        # Finite-size shot noise
        shot_noise_E = m_E * (1 - m_E) / np.sqrt(K_EE + 1)
        shot_noise_I = m_I * (1 - m_I) / np.sqrt(K_EI + 1)
        
        I_E_var = (K_EE * W_EE**2 * (var_E + shot_noise_E) + 
                   K_EI * W_EI**2 * (var_I + shot_noise_I) + 
                   self.noise_sig**2)
        
        I_I_var = K_IE * W_IE**2 * (var_E + shot_noise_E) + self.noise_sig**2
        
        # Firing rates with fluctuations
        m_E_new = self.firing_rate_with_fluctuations(I_E_mean, I_E_var, T_E)
        m_I_new = self.firing_rate_with_fluctuations(I_I_mean, I_I_var, T_I)
        
        # Variance dynamics for binary neurons
        var_E_new = m_E_new * (1 - m_E_new)
        var_I_new = m_I_new * (1 - m_I_new)
        
        # Near criticality, enhance variance (critical slowing down)
        if rho > 0.95:
            critical_enhancement = 1.0 + 20.0 * (rho - 0.95)**2
            var_E_new *= critical_enhancement
            var_I_new *= critical_enhancement
        
        # Covariance dynamics
        # STDP induces correlations, especially near criticality
        tau_corr = 100.0 / (1.0 + 10.0 * (1.0 - rho)**2)  # Faster near criticality
        target_cov = self.eta_stdp * m_E * m_I * 0.1 * (1.0 + 5.0 * (rho - 0.9))
        dcov_EI_dt = (target_cov - cov_EI) / tau_corr
        
        # Neural activity dynamics (fast timescale)
        tau_m = 1.0
        dm_E_dt = (m_E_new - m_E) / tau_m
        dm_I_dt = (m_I_new - m_I) / tau_m
        dvar_E_dt = (var_E_new - var_E) / tau_m
        dvar_I_dt = (var_I_new - var_I) / tau_m
        
        # Intrinsic plasticity with fluctuation correction
        # Fluctuations bias activity upward, need stronger correction
        fluctuation_bias = var_E / (2 * self.N_E)
        dT_E_dt = self.eta_ip * (m_E + fluctuation_bias - self.h_ip)
        dT_I_dt = 0  # No IP for inhibitory in original model
        
        # STDP with correlations and finite-size effects
        # Near criticality, STDP is modified by fluctuations
        correlation_factor = 1.0 + cov_EI / (m_E * m_I + 1e-10)
        
        # Finite-size prevents complete weight elimination
        if W_EE < 0.1:
            stdp_protection = np.exp(-1.0 / (W_EE + 0.01))
        else:
            stdp_protection = 1.0
            
        dW_EE_dt = self.eta_stdp * m_E * (m_E - m_E_new) * correlation_factor * stdp_protection
        
        # iSTDP
        dW_EI_dt = -self.eta_istdp * m_I * (1 - m_E_new * (1 + 1/self.h_ip))
        
        # Keep W_EI above minimum
        if W_EI <= self.W_ei_min * 1.1:
            dW_EI_dt = max(0, dW_EI_dt)
            
        dW_IE_dt = 0  # No plasticity for W_IE
        
        # Structural plasticity with finite-size effects
        # Creation rate decreases as network fills
        creation_rate = self.p_sp * (1 - C) * (1 - C)**2  # Harder to find empty spots
        
        # Removal rate from STDP depression
        if W_EE > 0:
            removal_rate = C * max(0, -dW_EE_dt / W_EE) * 0.1
        else:
            removal_rate = 0
            
        # Finite-size fluctuations in connectivity
        C_noise = self.finite_size_factor * np.sqrt(C * (1 - C)) * 0.01
        
        dC_dt = creation_rate - removal_rate + C_noise * np.random.randn()
        
        # Spectral radius dynamics
        rho_mean, rho_fluct = self.spectral_radius_dynamics(W_EE, C, var_E, m_E)
        
        # Spectral radius evolves toward criticality
        # Self-organized criticality mechanism
        if rho < self.critical_rho:
            # Below criticality: weights grow
            rho_drift = 0.001 * (self.critical_rho - rho)
        elif rho > 1.0:
            # Above criticality: weights shrink (stability)
            rho_drift = -0.01 * (rho - 1.0)
        else:
            # In critical region: fluctuation-dominated
            rho_drift = 0
            
        # Add finite-size noise to spectral radius
        drho_dt = rho_drift + rho_fluct * np.random.randn() * 0.001
        
        # Ensure spectral radius tracks actual weight/connectivity changes
        drho_dt += 0.1 * (rho_mean - rho)
        
        return [dm_E_dt, dm_I_dt, dvar_E_dt, dvar_I_dt, dcov_EI_dt,
                dW_EE_dt, dW_EI_dt, dW_IE_dt, dT_E_dt, dT_I_dt, dC_dt, drho_dt]
    
    def simulate(self, t_span, initial_state=None, dt=1.0):
        """
        Run mean field simulation
        
        Parameters:
        -----------
        t_span : float
            Total simulation time
        initial_state : array-like, optional
            Initial state vector
        dt : float
            Time step for integration
            
        Returns:
        --------
        t : array
            Time points
        solution : array
            Solution array with shape (n_timepoints, n_variables)
        """
        if initial_state is None:
            # Default initial conditions
            initial_state = [
                0.05,       # m_E: low initial activity
                0.0,        # m_I: inhibition starts silent
                0.0025,     # var_E: small initial variance
                0.0,        # var_I
                0.0,        # cov_EI
                0.5,        # W_EE: intermediate weight
                0.5,        # W_EI
                0.5,        # W_IE
                0.25,       # T_E: middle of range
                0.5,        # T_I
                self.lambda_ / self.N_E,  # C: initial connection fraction
                0.5         # rho: initial spectral radius
            ]
        
        # Time points
        t = np.arange(0, t_span, dt)
        
        # Solve ODE
        solution = odeint(self.mean_field_dynamics, initial_state, t)
        
        # Clip solution to valid ranges
        solution[:, 0] = np.clip(solution[:, 0], 0, 1)  # m_E
        solution[:, 1] = np.clip(solution[:, 1], 0, 1)  # m_I
        solution[:, 2] = np.clip(solution[:, 2], 0, 0.25)  # var_E
        solution[:, 3] = np.clip(solution[:, 3], 0, 0.25)  # var_I
        solution[:, 10] = np.clip(solution[:, 10], 0, 1)  # C
        
        return t, solution
    
    def plot_results(self, t, solution, save_path=None):
        """
        Create comprehensive plots of mean field results
        """
        # Unpack solution
        m_E = solution[:, 0]
        m_I = solution[:, 1]
        var_E = solution[:, 2]
        var_I = solution[:, 3]
        cov_EI = solution[:, 4]
        W_EE = solution[:, 5]
        W_EI = solution[:, 6]
        W_IE = solution[:, 7]
        T_E = solution[:, 8]
        T_I = solution[:, 9]
        C = solution[:, 10]
        rho = solution[:, 11]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Activities with variance bands
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, m_E, 'b-', linewidth=2, label='E (mean)')
        ax1.fill_between(t, m_E - np.sqrt(var_E), m_E + np.sqrt(var_E),
                        color='blue', alpha=0.2, label='E (±σ)')
        ax1.plot(t, m_I, 'r-', linewidth=2, label='I (mean)')
        ax1.axhline(y=self.h_ip, color='k', linestyle='--', alpha=0.5, label=f'Target={self.h_ip}')
        ax1.set_ylabel('Activity')
        ax1.set_title('Population Activities with Fluctuations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 0.5)
        
        # 2. Spectral radius evolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, rho, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='ρ=1 (critical)')
        ax2.axhline(y=self.critical_rho, color='orange', linestyle='--', 
                   label=f'ρ_c(N)={self.critical_rho:.3f}')
        ax2.fill_between(t, self.critical_rho, 1.0, color='yellow', alpha=0.2, label='Critical region')
        ax2.set_ylabel('Spectral Radius ρ')
        ax2.set_title('Self-Organization to Criticality')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.2)
        
        # 3. Connection fraction
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(t, C * 100, 'k-', linewidth=2)
        ax3.set_ylabel('Connection Fraction (%)')
        ax3.set_title('Network Connectivity Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Weights evolution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, W_EE, 'g-', linewidth=2, label='W_EE')
        ax4.plot(t, W_EI, 'm-', linewidth=2, label='W_EI')
        ax4.plot(t, W_IE, 'c-', linewidth=2, label='W_IE')
        ax4.set_ylabel('Mean Weight')
        ax4.set_title('Synaptic Weight Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 5. Variance dynamics
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.semilogy(t, var_E, 'b-', linewidth=2, label='Var[E]')
        ax5.semilogy(t, var_I, 'r-', linewidth=2, label='Var[I]')
        ax5.axhline(y=1/self.N_E, color='k', linestyle='--', alpha=0.5, 
                   label=f'1/N={1/self.N_E:.4f}')
        ax5.set_ylabel('Variance')
        ax5.set_title('Fluctuation Amplitude (log scale)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Phase space: Activity vs Spectral radius
        ax6 = fig.add_subplot(gs[1, 2])
        # Color code by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
        for i in range(0, len(t)-1, max(1, len(t)//100)):
            ax6.plot(m_E[i:i+2], rho[i:i+2], color=colors[i], linewidth=1)
        ax6.scatter(m_E[0], rho[0], color='green', s=100, marker='o', zorder=5, label='Start')
        ax6.scatter(m_E[-1], rho[-1], color='red', s=100, marker='*', zorder=5, label='End')
        ax6.axvline(x=self.h_ip, color='k', linestyle='--', alpha=0.3)
        ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Mean Activity (E)')
        ax6.set_ylabel('Spectral Radius ρ')
        ax6.set_title('Phase Space Trajectory')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Threshold dynamics
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(t, T_E, 'b-', linewidth=2, label='T_E')
        ax7.plot(t, T_I, 'r-', linewidth=2, label='T_I')
        ax7.set_ylabel('Threshold')
        ax7.set_title('Firing Thresholds')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, max(self.T_e_max, self.T_i_max) * 1.1)
        
        # 8. Effective input currents
        ax8 = fig.add_subplot(gs[2, 1])
        K_EE = C * self.N_E
        K_EI = self.p_ei * self.N_I
        I_E = K_EE * W_EE * m_E - K_EI * W_EI * m_I
        I_I = self.N_E * W_IE * m_E
        ax8.plot(t, I_E, 'b-', linewidth=2, label='I_E')
        ax8.plot(t, T_E, 'b--', alpha=0.5, label='T_E')
        ax8.plot(t, I_I, 'r-', linewidth=2, label='I_I')
        ax8.plot(t, T_I, 'r--', alpha=0.5, label='T_I')
        ax8.set_ylabel('Current / Threshold')
        ax8.set_title('Input Currents vs Thresholds')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Distance from criticality
        ax9 = fig.add_subplot(gs[2, 2])
        distance_from_critical = np.abs(rho - 1.0)
        distance_from_target = np.abs(m_E - self.h_ip)
        ax9.semilogy(t, distance_from_critical, 'g-', linewidth=2, label='|ρ - 1|')
        ax9.semilogy(t, distance_from_target, 'b-', linewidth=2, label=f'|m_E - {self.h_ip}|')
        ax9.set_ylabel('Distance')
        ax9.set_title('Convergence to Critical State')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Finite-size effects visualization
        ax10 = fig.add_subplot(gs[3, 0])
        # Show how variance scales with activity near criticality
        critical_mask = (rho > 0.95) & (rho < 1.05)
        if np.any(critical_mask):
            ax10.scatter(m_E[critical_mask], var_E[critical_mask], 
                       c=rho[critical_mask], cmap='hot', s=20, alpha=0.6)
            ax10.plot(m_E * (1 - m_E), 'k--', linewidth=1, label='Binomial var')
            cb = ax10.scatter([], [], c=[], cmap='hot')
            plt.colorbar(cb, ax=ax10, label='ρ')
        ax10.set_xlabel('Mean Activity')
        ax10.set_ylabel('Variance')
        ax10.set_title('Enhanced Fluctuations Near Criticality')
        ax10.grid(True, alpha=0.3)
        
        # 11. Correlation evolution
        ax11 = fig.add_subplot(gs[3, 1])
        ax11.plot(t, cov_EI, 'purple', linewidth=2)
        ax11.set_ylabel('Cov(E,I)')
        ax11.set_title('E-I Correlation')
        ax11.grid(True, alpha=0.3)
        
        # 12. Summary statistics
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')
        
        # Calculate summary statistics
        final_idx = -min(1000, len(t)//10)  # Last 10% of simulation
        stats_text = f"""Final State Summary:
        
Activity: {np.mean(m_E[final_idx:]):.4f} ± {np.std(m_E[final_idx:]):.4f}
Target: {self.h_ip}

Spectral Radius: {np.mean(rho[final_idx:]):.4f} ± {np.std(rho[final_idx:]):.4f}
Critical (N={self.N_E}): {self.critical_rho:.4f}

Connection Fraction: {np.mean(C[final_idx:])*100:.1f}%
Mean Weight W_EE: {np.mean(W_EE[final_idx:]):.4f}

Variance E: {np.mean(var_E[final_idx:]):.4e}
Finite-size scale: {1/self.N_E:.4e}

Network size: N_E = {self.N_E}
Finite-size factor: {self.finite_size_factor:.4f}
"""
        ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Overall title
        fig.suptitle('SORN Mean Field Model with Fluctuations and Finite-Size Effects', 
                    fontsize=16)
        
        # Add time axis label to bottom plots
        for ax in [ax7, ax8, ax9, ax10, ax11]:
            ax.set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_phase_diagram(self, param_name='h_ip', param_range=None, n_sims=20):
        """
        Create phase diagram showing different dynamical regimes
        """
        if param_range is None:
            if param_name == 'h_ip':
                param_range = np.linspace(0.05, 0.2, n_sims)
            elif param_name == 'noise_sig':
                param_range = np.linspace(0.1, 0.5, n_sims)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
        
        # Store results
        results = {
            'param_values': param_range,
            'final_activity': [],
            'final_rho': [],
            'final_variance': [],
            'stability': []
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for i, param_val in enumerate(param_range):
            # Set parameter
            if param_name == 'h_ip':
                self.h_ip = param_val
            elif param_name == 'noise_sig':
                self.noise_sig = param_val
            
            # Run simulation
            t, sol = self.simulate(50000, dt=1.0)
            
            # Extract final values (last 10%)
            final_idx = -len(t)//10
            final_m_E = np.mean(sol[final_idx:, 0])
            final_rho = np.mean(sol[final_idx:, 11])
            final_var = np.mean(sol[final_idx:, 2])
            
            # Determine stability
            if final_rho > 1.05:
                stability = 'unstable'
                color = 'red'
            elif final_rho > self.critical_rho:
                stability = 'critical'
                color = 'orange'
            else:
                stability = 'subcritical'
                color = 'blue'
            
            results['final_activity'].append(final_m_E)
            results['final_rho'].append(final_rho)
            results['final_variance'].append(final_var)
            results['stability'].append(stability)
            
            # Plot trajectory in phase space
            ax = axes[0, 0]
            ax.plot(sol[:, 0], sol[:, 11], color=color, alpha=0.5, linewidth=0.5)
            ax.scatter(final_m_E, final_rho, color=color, s=50, 
                      edgecolor='black', linewidth=1)
        
        # Finalize phase space plot
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].axhline(y=self.critical_rho, color='orange', linestyle='--')
        axes[0, 0].set_xlabel('Activity m_E')
        axes[0, 0].set_ylabel('Spectral Radius ρ')
        axes[0, 0].set_title('Phase Space Trajectories')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter vs final values
        ax = axes[0, 1]
        ax.plot(param_range, results['final_activity'], 'b-', linewidth=2, 
                label='Activity', marker='o')
        ax.plot(param_range, results['final_rho'], 'g-', linewidth=2, 
                label='Spectral radius', marker='s')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Final values')
        ax.set_title('Steady State vs Parameter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Variance scaling
        ax = axes[1, 0]
        ax.semilogy(param_range, results['final_variance'], 'purple', 
                   linewidth=2, marker='d')
        ax.axhline(y=1/self.N_E, color='k', linestyle='--', 
                  label=f'1/N={1/self.N_E:.4f}')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Final Variance')
        ax.set_title('Fluctuation Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stability diagram
        ax = axes[1, 1]
        colors = {'unstable': 'red', 'critical': 'orange', 'subcritical': 'blue'}
        for i, (param, stab) in enumerate(zip(param_range, results['stability'])):
            ax.scatter(param, i, color=colors[stab], s=100, marker='s')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Simulation #')
        ax.set_title('Stability Classification')
        ax.set_ylim(-1, len(param_range))
        
        # Add legend
        for stab, color in colors.items():
            ax.scatter([], [], color=color, s=100, marker='s', label=stab)
        ax.legend()
        
        plt.suptitle(f'Phase Diagram: Varying {param_name}', fontsize=14)
        plt.tight_layout()
        
        return fig, results


# Example usage and validation
if __name__ == "__main__":
    # Create mean field model with same parameters as full SORN
    print("Creating SORN mean field model with fluctuations and finite-size effects...")
    
    # Test with different network sizes to show finite-size effects
    fig_comparison, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    N_values = [50, 200, 1000]
    colors = ['red', 'blue', 'green']
    
    for N, color in zip(N_values, colors):
        print(f"\nSimulating N_E = {N}...")
        
        mf = SORNMeanFieldComplete(
            N_E=N,
            N_I=int(N/5),  # Keep ratio
            h_ip=0.1,
            eta_stdp=0.004,
            eta_istdp=0.001,
            eta_ip=0.01,
            lambda_=20,
            noise_sig=np.sqrt(0.05)
        )
        
        # Run simulation
        t, solution = mf.simulate(t_span=20000, dt=1.0)
        
        # Plot comparisons
        # Activity
        ax = axes[0, 0]
        ax.plot(t/1000, solution[:, 0], color=color, linewidth=2, 
                label=f'N={N}', alpha=0.8)
        
        # Spectral radius
        ax = axes[0, 1]
        ax.plot(t/1000, solution[:, 11], color=color, linewidth=2, 
                label=f'N={N}', alpha=0.8)
        ax.axhline(y=mf.critical_rho, color=color, linestyle='--', alpha=0.5)
        
        # Variance
        ax = axes[1, 0]
        ax.semilogy(t/1000, solution[:, 2], color=color, linewidth=2, 
                   label=f'N={N}', alpha=0.8)
        ax.axhline(y=1/N, color=color, linestyle='--', alpha=0.5)
        
        # Connection fraction
        ax = axes[1, 1]
        ax.plot(t/1000, solution[:, 10] * 100, color=color, linewidth=2, 
                label=f'N={N}', alpha=0.8)
    
    # Format comparison plots
    axes[0, 0].set_ylabel('Mean Activity')
    axes[0, 0].set_title('Activity: Finite-Size Effects')
    axes[0, 0].axhline(y=0.1, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_ylabel('Spectral Radius')
    axes[0, 1].set_title('Critical Points Shift with N')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='-', linewidth=2, alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].set_xlabel('Time (×1000 steps)')
    axes[1, 0].set_title('Fluctuation Scaling with System Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_ylabel('Connection Fraction (%)')
    axes[1, 1].set_xlabel('Time (×1000 steps)')
    axes[1, 1].set_title('Connectivity Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Finite-Size Effects in SORN Mean Field Model', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Now run a longer simulation with N=200 (matching the full model)
    print("\n" + "="*60)
    print("Running extended simulation with N=200 (matching full SORN)...")
    print("="*60)
    
    mf_full = SORNMeanFieldComplete(N_E=200, N_I=40)
    t_full, sol_full = mf_full.simulate(t_span=100000, dt=1.0)
    
    # Create comprehensive plots
    fig_full = mf_full.plot_results(t_full, sol_full)
    plt.show()
    
    # Create phase diagram
    print("\nGenerating phase diagram...")
    fig_phase, phase_results = mf_full.plot_phase_diagram(param_name='h_ip', 
                                                         param_range=np.linspace(0.05, 0.2, 15))
    plt.show()
    
    print("\nMean field simulation complete!")
    print(f"Critical spectral radius for N=200: {mf_full.critical_rho:.4f}")
    print(f"Finite-size factor: {mf_full.finite_size_factor:.4f}")