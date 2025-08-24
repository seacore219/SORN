import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import powerlaw
import h5py
import os

class SORN:
    def __init__(self, NE=200, NI=40):
        # QUOTE: "composed of a set of threshold neurons divided into NE excitatory and NI inhibitory units, with NI = 0.2 × NE"
        self.NE = NE
        self.NI = NI
        
        # QUOTE: "ξ represents the unit's independent Gaussian noise, with mean zero and variance σ² = 0.05"
        # Since np.random.normal takes std dev, not variance: σ = sqrt(0.05)
        self.sigma = np.sqrt(0.05)
        
        # QUOTE: "We set μIP = 0.1"
        self.mu_IP = 0.1
        
        # QUOTE: "Unless stated otherwise, all simulations were performed with the learning rates from [35]: ηSTDP = 0.004, ηinh = 0.001 and ηIP = 0.01"
        self.eta_STDP = 0.004
        self.eta_inh = 0.001
        self.eta_IP = 0.01
        
        # QUOTE: "The new synapses were set to a small value ηSP = 0.001"
        self.eta_SP = 0.001
        
        # Initialize connectivity matrices
        self._initialize_weights()
        
        # Initialize thresholds
        self._initialize_thresholds()
        
        # QUOTE: "The state of the network, at each discrete time step t, was given by the binary vectors x(t) ∈ {0, 1}^NE and y(t) ∈ {0, 1}^NI"
        self.x = np.zeros(self.NE, dtype=int)
        self.y = np.zeros(self.NI, dtype=int)
        self.x_prev = np.zeros(self.NE, dtype=int)
        self.y_prev = np.zeros(self.NI, dtype=int)
        
        # QUOTE: "The probability was set to pSP(NE = 200) = 0.1 for a network of size NE = 200, and pSP scaled with the square of the network size:"
        # QUOTE: "pSP(NE) = NE(NE-1)/(200×199) × pSP(200)"
        self.p_SP = (self.NE * (self.NE - 1)) / (200 * 199) * 0.1
        
        # Data collection
        self.connection_fractions = []
        self.activities = []
        
    def _initialize_weights(self):
        # QUOTE: "WEE and WEI started as sparse matrices with connection probability of 0.1 and 0.2, respectively"
        # QUOTE: "synaptic weights drawn from a uniform distribution over the interval [0, 0.1]"
        
        # WEE initialization - using dense for faster operations
        self.WEE = np.zeros((self.NE, self.NE))
        mask = np.random.rand(self.NE, self.NE) < 0.1
        self.WEE[mask] = np.random.uniform(0, 0.1, size=np.sum(mask))
        
        # QUOTE: "self-connections were absent"
        np.fill_diagonal(self.WEE, 0)
        
        # WEI initialization  
        self.WEI = np.zeros((self.NE, self.NI))
        mask = np.random.rand(self.NE, self.NI) < 0.2
        self.WEI[mask] = np.random.uniform(0, 0.1, size=np.sum(mask))
        
        # QUOTE: "WIE was a fixed fully connected matrix"
        self.WIE = np.random.uniform(0, 0.1, size=(self.NI, self.NE))
        
        # QUOTE: "normalized separately for incoming excitatory and inhibitory inputs to each neuron"
        self._normalize_weights()
        
    def _normalize_weights(self):
        # QUOTE: "normalized separately for incoming excitatory and inhibitory inputs to each neuron"
        
        # Normalize incoming excitatory connections
        row_sums = self.WEE.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.WEE /= row_sums
                
        # Normalize incoming inhibitory connections
        row_sums = self.WEI.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.WEI /= row_sums
                
    def _initialize_thresholds(self):
        # QUOTE: "The thresholds TI and TE were drawn from uniform distributions over the intervals [0, TImax] and [0, TEmax], respectively, with TImax = 1 and TEmax = 0.5"
        self.TE = np.random.uniform(0, 0.5, size=self.NE)
        self.TI = np.random.uniform(0, 1, size=self.NI)
        
    def update_neurons(self, u_ext=None):
        # Store previous states for STDP
        self.x_prev = self.x.copy()
        self.y_prev = self.y.copy()
        
        # QUOTE: "The external input uExt was zero for all neurons, except during the external input experiment"
        if u_ext is None:
            u_ext = np.zeros(self.NE)
            
        # QUOTE: "ξ represents the unit's independent Gaussian noise, with mean zero and variance σ² = 0.05"
        xi_E = np.random.normal(0, self.sigma, size=self.NE)
        xi_I = np.random.normal(0, self.sigma, size=self.NI)
        
        # QUOTE: "xi(t+1) = Θ[∑j WEEij(t)xj(t) - ∑k WEIik(t)yk(t) + uExt_i(t) + ξE_i(t) - TE_i(t)]"
        # CORRECTED: Use previous states for consistency
        exc_input = self.WEE.dot(self.x_prev) - self.WEI.dot(self.y_prev) + u_ext + xi_E - self.TE
        self.x = (exc_input > 0).astype(int)
        
        # QUOTE: "yi(t+1) = Θ[∑j WIEij(t)xj(t) + ξI_i(t) - TI_i]"
        # This is correct - uses x_prev
        inh_input = self.WIE.dot(self.x_prev) + xi_I - self.TI
        self.y = (inh_input > 0).astype(int)
        
    def apply_stdp(self):
        # QUOTE: "ΔWEEij(t) = ηSTDP[xi(t)xj(t-1) - xj(t)xi(t-1)]"
        # Vectorized implementation for speed
        
        # Pre-post term: xi(t) * xj(t-1)
        pre_post = np.outer(self.x, self.x_prev)
        # Post-pre term: xj(t) * xi(t-1)  
        post_pre = np.outer(self.x_prev, self.x).T
        
        # Update weights
        self.WEE += self.eta_STDP * (pre_post - post_pre)
        
        # QUOTE: "self-connections were absent"
        np.fill_diagonal(self.WEE, 0)
        
        # QUOTE: "Negative and null weights were pruned after every time step"
        self.WEE[self.WEE <= 0] = 0
        
    def apply_istdp(self):
        # QUOTE: "ΔWEIij(t) = -ηinh·yj(t-1)[1-xi(t)(1+1/μIP)]"
        # CORRECTED: Fixed parentheses error
        
        # Calculate the term [1 - xi(t)*(1+1/μIP)]
        factor = 1 - self.x * (1 + 1/self.mu_IP)
        
        # Update weights: for each i,j: -ηinh * yj(t-1) * factor[i]
        delta = -self.eta_inh * np.outer(factor, self.y_prev)
        self.WEI += delta
        
        # Ensure weights stay non-negative (not explicitly stated but implied)
        self.WEI[self.WEI < 0] = 0
        
    def apply_structural_plasticity(self):
        # QUOTE: "Fourth, the structural plasticity (SP) added new synapses between unconnected neurons"
        # QUOTE: "It added a random directed connection between two unconnected neurons (at a particular time step) with a small probability pSP"
        
        # Find unconnected pairs (excluding diagonal)
        unconnected = (self.WEE == 0)
        np.fill_diagonal(unconnected, False)
        
        # Random selection for new connections
        new_connections = unconnected & (np.random.rand(self.NE, self.NE) < self.p_SP)
        
        # QUOTE: "The new synapses were set to a small value ηSP = 0.001"
        self.WEE[new_connections] = self.eta_SP
        
    def apply_synaptic_normalization(self):
        # QUOTE: "SN could be written as an update equation, applicable to WEE and WEI, and executed at each time step after all other synaptic plasticity rules"
        # QUOTE: "Wij(t) ← Wij(t)/∑j Wij(t)"
        
        # Normalize incoming excitatory connections
        row_sums = self.WEE.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.WEE /= row_sums
                
        # Normalize incoming inhibitory connections
        row_sums = self.WEI.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.WEI /= row_sums
                
    def apply_intrinsic_plasticity(self):
        # QUOTE: "ΔTE_i = ηIP[xi(t) - μIP]"
        # QUOTE: "for simplicity, it could be set to the network average firing rate μIP, thus being equal for all neurons"
        self.TE += self.eta_IP * (self.x - self.mu_IP)
        
        # Keep thresholds in valid range
        self.TE = np.clip(self.TE, 0, 0.5)
        
    def step(self, u_ext=None):
        # Update neurons first
        self.update_neurons(u_ext)
        
        # Apply plasticity rules
        self.apply_stdp()
        self.apply_istdp()
        self.apply_structural_plasticity()
        
        # QUOTE: "executed at each time step after all other synaptic plasticity rules"
        self.apply_synaptic_normalization()
        
        self.apply_intrinsic_plasticity()
        
    def get_activity(self):
        # QUOTE: "a(t) = ∑i xi(t)"
        return np.sum(self.x)
    
    def get_connection_fraction(self):
        # QUOTE: "Fraction of active connections in the SORN"
        total_possible = self.NE * (self.NE - 1)  # excluding self-connections
        active = np.sum(self.WEE > 0) - np.sum(np.diag(self.WEE) > 0)  # exclude diagonal
        return active / total_possible
        
    def run(self, steps, with_plasticity=True, save_raster=True, save_every=1000, progress_bar=True):
        activities = []
        exc_raster = []
        inh_raster = []
        
        iterator = tqdm(range(steps)) if progress_bar else range(steps)
        
        for t in iterator:
            if with_plasticity:
                self.step()
            else:
                self.update_neurons()
                
            activities.append(self.get_activity())
            
            # Save rasters at specified interval
            if save_raster and t % save_every == 0:
                exc_raster.append(self.x.copy())
                inh_raster.append(self.y.copy())
            
            # Store connection fraction periodically
            if t % 100 == 0:  # Less frequent for speed
                self.connection_fractions.append((t, self.get_connection_fraction()))
            
        self.activities = np.array(activities)
        return np.array(activities), np.array(exc_raster), np.array(inh_raster)
    
    def extract_avalanches(self, activities=None):
        if activities is None:
            activities = self.activities
            
        # QUOTE: "θ was set to half of the mean network activity ⟨a(t)⟩, which by definition is ⟨a(t)⟩ = μIP = 0.1"
        # QUOTE: "For simplicity, θ was rounded to the nearest integer, as a(t) can only assume integer values"
        # CORRECTED: Use theoretical value based on μIP
        mean_activity = self.mu_IP * self.NE  # Should be 0.1 * 200 = 20 for default
        theta = int(round(mean_activity / 2))  # Should be 10 for default
        
        avalanches = []
        in_avalanche = False
        start_time = 0
        size = 0
        
        # QUOTE: "An avalanche started when the network activity went above θ, and T was the number of subsequent time steps during which the activity remained above θ"
        # QUOTE: "S was the sum of spikes exceeding the threshold at each time step during the avalanche"
        for t, a in enumerate(activities):
            if a > theta:
                if not in_avalanche:
                    in_avalanche = True
                    start_time = t
                    size = 0
                # QUOTE: "S = ∑(a(t) - θ)"
                size += (a - theta)
            else:
                if in_avalanche:
                    duration = t - start_time
                    avalanches.append((duration, size))
                    in_avalanche = False
                    
        return avalanches

def plot_figure_1_individual(connection_fractions, activities, avalanches, sim_num, save_path):
    """Plot Figure 1 for an individual simulation matching paper style exactly"""
    from matplotlib import gridspec
    import matplotlib.patches as patches
    
    # Figure parameters from paper
    width = 10
    height = 3
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.7])
    letter_size = 13
    line_width = 1.0
    
    # Color parameters from paper
    c_size = '#B22400'
    c_duration = '#006BB2'
    c_stable = '#2E4172'
    c_notstable = '#7887AB'
    
    # Panel A: Connection fraction
    fig_1a = plt.subplot(gs[0])
    
    if len(connection_fractions) > 0:
        times, fractions = zip(*connection_fractions)
        times = np.array(times)
        fractions = np.array(fractions) * 100  # Convert to percentage
        
        # Plot with different colors for different phases
        # Find transition points (approximate)
        decay_end = int(1e6)
        growth_end = int(2e6)
        
        # Plot different phases
        mask_decay = times < decay_end
        mask_growth = (times >= decay_end) & (times < growth_end)
        mask_stable = times >= growth_end
        
        if np.any(mask_decay):
            fig_1a.plot(times[mask_decay], fractions[mask_decay], c_notstable, linewidth=line_width)
        if np.any(mask_growth):
            fig_1a.plot(times[mask_growth], fractions[mask_growth], c_notstable, linewidth=line_width)
        if np.any(mask_stable):
            fig_1a.plot(times[mask_stable], fractions[mask_stable], c_stable, linewidth=line_width)
        
        # Annotations
        fig_1a.text(2e5, 1, r'decay', fontsize=letter_size, color=c_notstable)
        fig_1a.text(8e5, 10, r'growth', fontsize=letter_size, color=c_notstable)
        fig_1a.text(26e5, 13.5, r'stable', fontsize=letter_size, color=c_stable)
    
    # Axis formatting
    fig_1a.set_xlim([0, 4e6])
    fig_1a.set_ylim([0, 100])
    
    # Remove spines
    fig_1a.spines['right'].set_visible(False)
    fig_1a.spines['top'].set_visible(False)
    fig_1a.spines['left'].set_visible(False)
    fig_1a.spines['bottom'].set_visible(False)
    fig_1a.yaxis.set_ticks_position('left')
    fig_1a.xaxis.set_ticks_position('bottom')
    fig_1a.tick_params(axis='both', which='both', length=0)
    fig_1a.grid()
    
    # Set ticks
    fig_1a.set_xticks(np.arange(0, 4.1e6, 1e6))
    fig_1a.set_xticklabels(['0', '1', '2', '3', '4'])
    fig_1a.set_yticks([5, 10, 15])
    fig_1a.set_yticklabels(['5%', '10%', '15%'])
    
    fig_1a.set_xlabel(r'$10^6$ time steps', fontsize=letter_size)
    fig_1a.set_ylabel(r'Active Connections', fontsize=letter_size)
    fig_1a.tick_params(axis='both', which='major', labelsize=letter_size)
    
    # Panel B: Activity and avalanches
    fig_1b = plt.subplot(gs[1])
    
    # Sample activity
    plot_last_steps = 150
    activity_sample = activities[-plot_last_steps:] if len(activities) > plot_last_steps else activities
    
    # Use theoretical threshold
    mean_activity = 0.1 * 200  # μIP * NE
    theta = int(round(mean_activity / 2))
    
    # Plot activity and threshold
    boundary = theta * np.ones(len(activity_sample))
    fig_1b.plot(activity_sample, 'k', linewidth=line_width)
    fig_1b.plot(boundary, '--k', linewidth=line_width)
    fig_1b.fill_between(np.arange(len(activity_sample)), activity_sample, boundary,
                       alpha=0.5, where=activity_sample>=boundary, facecolor=c_size, interpolate=True)
    
    # Annotations
    fig_1b.text(20, 45, r'avalanches', fontsize=letter_size, color='k')
    fig_1b.text(70, 4, r'duration', fontsize=letter_size, color=c_duration)
    fig_1b.text(80, 12, r'size', fontsize=letter_size, color=c_size)
    fig_1b.text(62, -4, r'100 time steps', fontsize=letter_size, color='k')
    
    # Duration line
    fig_1b.plot((58, 122), (8, 8), c_duration, linewidth=2.0)
    fig_1b.plot((50, 150), (0, 0), 'k', linewidth=2.5)
    
    # Arrows
    arrow1 = patches.FancyArrowPatch((35,44), (12,29), arrowstyle='-|>',
                                    fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow1)
    arrow2 = patches.FancyArrowPatch((55,44), (40,35), arrowstyle='-|>',
                                    fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow2)
    arrow3 = patches.FancyArrowPatch((65,44), (75,38), arrowstyle='-|>',
                                    fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow3)
    arrow4 = patches.FancyArrowPatch((60,44), (54.5,15), arrowstyle='-|>',
                                    fc='k', lw=1, mutation_scale=10)
    fig_1b.add_patch(arrow4)
    
    # Axis formatting
    fig_1b.set_xlim([0, len(activity_sample)])
    fig_1b.set_ylim([0, 50])
    
    fig_1b.spines['right'].set_visible(False)
    fig_1b.spines['top'].set_visible(False)
    fig_1b.spines['bottom'].set_visible(False)
    fig_1b.yaxis.set_ticks_position('left')
    fig_1b.axes.get_xaxis().set_visible(False)
    
    fig_1b.set_yticks([0, theta, 20, 40])
    fig_1b.set_yticklabels(['0', r'$\theta$', '20', '40'])
    
    fig_1b.set_ylabel(r'$a(t)$' + r' [# neurons]', fontsize=letter_size)
    fig_1b.tick_params(axis='both', which='major', labelsize=letter_size)
    
    # Panel labels
    fig.text(0.01, 0.9, "A", weight="bold",
             horizontalalignment='left', verticalalignment='center')
    fig.text(0.55, 0.9, "B", weight="bold",
             horizontalalignment='left', verticalalignment='center')
    
    plt.gcf().subplots_adjust(bottom=0.17)
    fig.subplots_adjust(wspace=.4)
    
    plt.savefig(os.path.join(save_path, f'figure_1_sim_{sim_num}.pdf'), format='pdf')
    plt.close()

def plot_figure_2_compiled(all_avalanches, NE, save_path, 
                         T_min=6, T_max=60, S_min=10, S_max=1500):
    """Plot Figure 2 matching paper style exactly"""
    from matplotlib import gridspec
    
    # Figure parameters from paper
    width = 8
    height = width / 1.718
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(2, 3)
    letter_size = 10
    letter_size_panel = 12
    line_width = 1.5
    line_width_fit = 2.0
    subplot_letter = (-0.25, 1.15)
    
    # Color parameters from paper
    c_size = '#B22400'
    c_duration = '#006BB2'
    c_rawdata = 'gray'
    c_expcut = 'k'
    
    durations, sizes = zip(*all_avalanches) if all_avalanches else ([], [])
    durations = np.array(durations)
    sizes = np.array(sizes)
    
    # Panel A: Duration distribution
    fig_2a = plt.subplot(gs[0])
    print('Fig. 2A...')
    
    if len(durations) > 0:
        # Raw data
        T_x, inverse = np.unique(durations, return_inverse=True)
        y_freq = np.bincount(inverse)
        T_y = y_freq / float(y_freq.sum())
        fig_2a.plot(T_x, T_y, '.', color=c_rawdata, markersize=2, zorder=1)
        
        # Power law fit
        try:
            T_fit = powerlaw.Fit(durations, xmin=T_min, xmax=T_max, discrete=True)
            T_alpha = T_fit.alpha
            
            # Plot fit
            x_plot = np.logspace(np.log10(T_min), np.log10(T_max), 100)
            y_plot = T_fit.power_law.pdf(x_plot)
            fig_2a.plot(x_plot, y_plot, color=c_duration,
                       label=r'$ \alpha = $' + str(round(T_alpha, 2)),
                       linewidth=line_width_fit, zorder=3)
            
            # Exponential cutoff
            T_fit_exp = powerlaw.Fit(durations, xmin=T_min, discrete=True)
            T_trunc_alpha = T_fit_exp.truncated_power_law.parameter1
            T_trunc_beta = T_fit_exp.truncated_power_law.parameter2
            y_exp = T_fit_exp.truncated_power_law.pdf(x_plot)
            fig_2a.plot(x_plot, y_exp, color=c_expcut,
                       label=r'$ \alpha^* = $' + str(round(T_trunc_alpha, 2)) + ', ' +
                       r'$ \beta_{\alpha}^* = $' + str(round(T_trunc_beta, 3)),
                       linewidth=line_width, zorder=2)
        except:
            print("Duration fitting failed")
    
    # Axis formatting
    fig_2a.set_xscale('log')
    fig_2a.set_yscale('log')
    fig_2a.set_xlabel(r'$T$', fontsize=letter_size)
    fig_2a.set_ylabel(r'$f(T)$', fontsize=letter_size)
    
    fig_2a.spines['right'].set_visible(False)
    fig_2a.spines['top'].set_visible(False)
    fig_2a.xaxis.set_ticks_position('bottom')
    fig_2a.yaxis.set_ticks_position('left')
    fig_2a.tick_params(labelsize=letter_size)
    
    fig_2a.set_xlim([1, 300])
    fig_2a.set_ylim([0.0001, 1])
    fig_2a.set_xticks([1, 10, 100])
    fig_2a.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$'])
    fig_2a.set_yticks([1, 0.01, 0.0001])
    fig_2a.set_yticklabels(['$10^0$', '$10^{-2}$', '$10^{-4}$'])
    
    fig_2a.legend(loc=(0.0, 0.85), prop={'size': letter_size},
                 title='Fit parameters', frameon=False)
    fig_2a.get_legend().get_title().set_fontsize(letter_size)
    
    # Panel B: Size distribution
    fig_2b = plt.subplot(gs[1])
    print('Fig. 2B...')
    
    if len(sizes) > 0:
        # Raw data
        S_x, inverse = np.unique(sizes, return_inverse=True)
        y_freq = np.bincount(inverse)
        S_y = y_freq / float(y_freq.sum())
        fig_2b.plot(S_x, S_y, '.', color=c_rawdata, markersize=2, zorder=1)
        
        # Power law fit
        try:
            S_fit = powerlaw.Fit(sizes, xmin=S_min, xmax=S_max, discrete=True)
            S_alpha = S_fit.alpha
            
            # Plot fit
            x_plot = np.logspace(np.log10(S_min), np.log10(S_max), 100)
            y_plot = S_fit.power_law.pdf(x_plot)
            fig_2b.plot(x_plot, y_plot, color=c_size,
                       label=r'$ \tau = $' + str(round(S_alpha, 2)),
                       linewidth=line_width_fit, zorder=3)
            
            # Exponential cutoff
            S_fit_exp = powerlaw.Fit(sizes, xmin=S_min, discrete=True)
            S_trunc_alpha = S_fit_exp.truncated_power_law.parameter1
            S_trunc_beta = S_fit_exp.truncated_power_law.parameter2
            y_exp = S_fit_exp.truncated_power_law.pdf(x_plot)
            fig_2b.plot(x_plot, y_exp, color=c_expcut,
                       label=r'$ \tau^* = $' + str(round(S_trunc_alpha, 2)) +
                       '; ' + r'$\beta_{\tau}^* = $' + str(round(S_trunc_beta, 3)),
                       linewidth=line_width, zorder=2)
        except:
            print("Size fitting failed")
    
    # Axis formatting
    fig_2b.set_xscale('log')
    fig_2b.set_yscale('log')
    fig_2b.set_xlabel(r'$S$', fontsize=letter_size)
    fig_2b.set_ylabel(r'$f(S)$', fontsize=letter_size)
    
    fig_2b.spines['right'].set_visible(False)
    fig_2b.spines['top'].set_visible(False)
    fig_2b.xaxis.set_ticks_position('bottom')
    fig_2b.yaxis.set_ticks_position('left')
    fig_2b.tick_params(labelsize=letter_size)
    
    fig_2b.set_xlim([1, 3000])
    fig_2b.set_ylim([0.00001, 0.1])
    fig_2b.set_xticks([1, 10, 100, 1000])
    fig_2b.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$', '$10^{3}$'])
    fig_2b.set_yticks([0.1, 0.001, 0.00001])
    fig_2b.set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
    
    fig_2b.legend(loc=(0.0, 0.85), prop={'size': letter_size},
                 title='Fit parameters', frameon=False)
    fig_2b.get_legend().get_title().set_fontsize(letter_size)
    
    # Panel C: Average size vs duration
    fig_2c = plt.subplot(gs[2])
    print('Fig 2C...')
    
    if len(durations) > 0 and len(sizes) > 0:
        # Calculate average size for each duration
        T_unique = np.unique(durations)
        T_bins = []
        S_avg = []
        
        for T in T_unique:
            mask = durations == T
            if np.sum(mask) >= 5:
                T_bins.append(T)
                S_avg.append(np.mean(sizes[mask]))
        
        if len(T_bins) > 2:
            T_bins = np.array(T_bins)
            S_avg = np.array(S_avg)
            
            # Normalize
            S_avg_norm = S_avg / S_avg.sum()
            
            # Plot data
            fig_2c.plot(T_bins, S_avg_norm, '.', color=c_rawdata,
                       markersize=2, zorder=1, label=r'$\gamma_{\rm data}$')
            
            # Plot theoretical ratio if fits exist
            if 'T_alpha' in locals() and 'S_alpha' in locals():
                gamma = (T_alpha - 1) / (S_alpha - 1)
                x_range = np.arange(1, T_bins.max())
                y_theory = S_avg_norm.min() * x_range**gamma
                fig_2c.plot(x_range, y_theory, 'r',
                           label=r'$ \frac{\alpha-1}{\tau-1} $ = ' + str(round(gamma, 2)),
                           linewidth=line_width)
            
            # Plot reference line
            fig_2c.plot(x_range, S_avg_norm.min() * x_range**1.3, '--k',
                       label=r'$\gamma = $' + str(1.3), linewidth=line_width)
    
    # Axis formatting
    fig_2c.set_xscale('log')
    fig_2c.set_yscale('log')
    fig_2c.set_xlabel(r'$T$', fontsize=letter_size)
    fig_2c.set_ylabel(r'$ \langle S \rangle (T)$', fontsize=letter_size)
    
    fig_2c.spines['right'].set_visible(False)
    fig_2c.spines['top'].set_visible(False)
    fig_2c.xaxis.set_ticks_position('bottom')
    fig_2c.yaxis.set_ticks_position('left')
    fig_2c.tick_params(labelsize=letter_size)
    
    fig_2c.set_xlim([1, 200])
    fig_2c.set_ylim([0.000001, 0.1])
    fig_2c.set_xticks([1, 10, 100])
    fig_2c.set_xticklabels(['$10^0$', '$10^{1}$', '$10^{2}$'])
    fig_2c.set_yticks([0.1, 0.001, 0.00001])
    fig_2c.set_yticklabels(['$10^{-1}$', '$10^{-3}$', '$10^{-5}$'])
    
    fig_2c.legend(loc=(0.0, 0.65), prop={'size': letter_size},
                 title='Exponent ratio', frameon=False, numpoints=1)
    fig_2c.get_legend().get_title().set_fontsize(letter_size)
    
    # Panel labels
    fig_2a.annotate('A', xy=subplot_letter, xycoords='axes fraction',
                   fontsize=letter_size_panel, fontweight='bold',
                   horizontalalignment='right', verticalalignment='bottom')
    fig_2b.annotate('B', xy=subplot_letter, xycoords='axes fraction',
                   fontsize=letter_size_panel, fontweight='bold',
                   horizontalalignment='right', verticalalignment='bottom')
    fig_2c.annotate('C', xy=subplot_letter, xycoords='axes fraction',
                   fontsize=letter_size_panel, fontweight='bold',
                   horizontalalignment='right', verticalalignment='bottom')
    
    # Empty panels D, E, F for network size scaling
    for idx, label in enumerate(['D', 'E', 'F']):
        ax = plt.subplot(gs[3 + idx])
        ax.text(0.5, 0.5, f'{label}\n\nNetwork size\nscaling', 
                ha='center', va='center', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    fig.subplots_adjust(wspace=.5, hspace=.65)
    
    plt.savefig(os.path.join(save_path, 'figure_2_compiled.pdf'), format='pdf', dpi=300)
    plt.close()

def plot_raster(exc_raster, inh_raster, sim_num, save_path, sample_time=1000):
    """Plot spike raster distinguishing between excitatory and inhibitory neurons"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Take a sample of the raster for visualization
    exc_sample = exc_raster[:sample_time] if len(exc_raster) > sample_time else exc_raster
    inh_sample = inh_raster[:sample_time] if len(inh_raster) > sample_time else inh_raster
    
    # Find spike times for excitatory neurons
    exc_spike_times, exc_spike_neurons = np.where(exc_sample.T)
    
    # Find spike times for inhibitory neurons  
    inh_spike_times, inh_spike_neurons = np.where(inh_sample.T)
    
    # Plot excitatory spikes in blue
    ax.scatter(exc_spike_times, exc_spike_neurons, s=1, c='blue', label='Excitatory', alpha=0.5)
    
    # Plot inhibitory spikes in red, shifted up
    ax.scatter(inh_spike_times, inh_spike_neurons + exc_sample.shape[1] + 5, 
               s=1, c='red', label='Inhibitory', alpha=0.5)
    
    # Add dividing line
    ax.axhline(y=exc_sample.shape[1] + 2.5, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('Neuron index')
    ax.set_title(f'Spike Raster - Simulation {sim_num}')
    ax.legend()
    
    # Set y-axis labels
    ax.set_ylim(-5, exc_sample.shape[1] + inh_sample.shape[1] + 10)
    ax.set_yticks([exc_sample.shape[1]/2, exc_sample.shape[1] + 5 + inh_sample.shape[1]/2])
    ax.set_yticklabels(['Excitatory', 'Inhibitory'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'raster_sim_{sim_num}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_simulation_data(sim_num, sorn, activities, exc_raster, inh_raster, avalanches, save_path):
    """Save simulation data to HDF5 file"""
    filename = os.path.join(save_path, f'simulation_{sim_num}.h5')
    
    with h5py.File(filename, 'w') as f:
        # Create groups
        params = f.create_group('parameters')
        data = f.create_group('data')
        matrices = f.create_group('matrices')
        
        # Save parameters
        params.attrs['NE'] = sorn.NE
        params.attrs['NI'] = sorn.NI
        params.attrs['sigma'] = sorn.sigma
        params.attrs['mu_IP'] = sorn.mu_IP
        params.attrs['eta_STDP'] = sorn.eta_STDP
        params.attrs['eta_inh'] = sorn.eta_inh
        params.attrs['eta_IP'] = sorn.eta_IP
        params.attrs['eta_SP'] = sorn.eta_SP
        params.attrs['p_SP'] = sorn.p_SP
        
        # Save data
        data.create_dataset('activities', data=activities)
        data.create_dataset('excitatory_raster', data=exc_raster)
        data.create_dataset('inhibitory_raster', data=inh_raster)
        
        # Save connection fractions
        if len(sorn.connection_fractions) > 0:
            cf_times, cf_values = zip(*sorn.connection_fractions)
            data.create_dataset('connection_fraction_times', data=cf_times)
            data.create_dataset('connection_fraction_values', data=cf_values)
        
        # Save avalanches
        if len(avalanches) > 0:
            durations, sizes = zip(*avalanches)
            data.create_dataset('avalanche_durations', data=durations)
            data.create_dataset('avalanche_sizes', data=sizes)
        
        # Save final weight matrices
        matrices.create_dataset('WEE', data=sorn.WEE)
        matrices.create_dataset('WEI', data=sorn.WEI)
        matrices.create_dataset('WIE', data=sorn.WIE)
        
        # Save final thresholds
        matrices.create_dataset('TE', data=sorn.TE)
        matrices.create_dataset('TI', data=sorn.TI)

def run_multiple_simulations(n_simulations=50, steps_per_simulation=4000000, 
                           transient_fraction=0.5, NE=200, NI=40, save_path='sorn_results'):
    """
    Run multiple SORN simulations and collect avalanche statistics.
    
    QUOTE: "The raw data points of 50 independent SORN simulations are shown in gray"
    QUOTE: "All distributions show combined data of 50 independent simulations"
    
    Parameters:
    -----------
    transient_fraction : float
        Fraction of total steps to discard as transient (default 0.5 ≈ 2M/4M as per paper)
    """
    all_avalanches = []
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate transient steps based on fraction
    transient_steps = int(steps_per_simulation * transient_fraction)
    
    print(f"Running {n_simulations} independent SORN simulations...")
    print(f"Each simulation: {steps_per_simulation:,} steps")
    print(f"Transient period: {transient_steps:,} steps ({transient_fraction*100:.1f}% of total)")
    print(f"Avalanche measurement period: {steps_per_simulation - transient_steps:,} steps ({(1-transient_fraction)*100:.1f}% of total)")
    print(f"Results will be saved to: {save_path}")
    
    for sim in range(n_simulations):
        print(f"\nSimulation {sim+1}/{n_simulations}")
        
        # Create new SORN instance for each simulation
        sorn = SORN(NE=NE, NI=NI)
        
        # Calculate save_every based on total steps to avoid too much data
        save_every = max(1, steps_per_simulation // 10000)  # Save at most 10000 time points
        
        # Run simulation
        activities, exc_raster, inh_raster = sorn.run(steps_per_simulation, 
                                                      with_plasticity=True, 
                                                      save_raster=True,
                                                      save_every=save_every,
                                                      progress_bar=True)
        
        # Extract avalanches from stable phase only
        stable_activities = activities[transient_steps:]
        
        avalanches = sorn.extract_avalanches(stable_activities)
        all_avalanches.extend(avalanches)
        
        # Extract stable phase rasters
        stable_start_idx = transient_steps // save_every
        stable_exc_raster = exc_raster[stable_start_idx:] if stable_start_idx < len(exc_raster) else exc_raster
        stable_inh_raster = inh_raster[stable_start_idx:] if stable_start_idx < len(inh_raster) else inh_raster
        
        print(f"Avalanches found in this simulation: {len(avalanches)}")
        
        # Plot individual Figure 1
        plot_figure_1_individual(sorn.connection_fractions, stable_activities, 
                               avalanches, sim+1, save_path)
        
        # Plot raster
        if len(stable_exc_raster) > 0:
            plot_raster(stable_exc_raster, stable_inh_raster, sim+1, save_path)
        
        # Save simulation data to HDF5
        save_simulation_data(sim+1, sorn, activities, exc_raster, inh_raster, 
                           avalanches, save_path)
    
    return all_avalanches



# Example usage
if __name__ == "__main__":
    # QUOTE: "The raw data points of 50 independent SORN simulations are shown in gray"
    n_simulations = 10  # Use 50 as per paper for full reproduction
    
    # The paper uses at least 3M steps total with 2M transient
    steps_per_sim = 4000000  # 4M steps for better statistics
    
    # The paper uses 2M transient out of total steps
    transient_fraction = 0.5  # Use 50% of steps as transient
    
    # Output directory
    save_path = 'sorn_results_corrected'
    
    print("SORN simulation parameters:")
    print(f"Number of simulations: {n_simulations}")
    print(f"Steps per simulation: {steps_per_sim:,}")
    print(f"Transient fraction: {transient_fraction*100:.1f}%")
    print(f"Network size: NE={200}, NI={40}")
    print()
    
    # Run multiple simulations
    all_avalanches = run_multiple_simulations(
        n_simulations=n_simulations,
        steps_per_simulation=steps_per_sim,
        transient_fraction=transient_fraction,
        save_path=save_path
    )
    
    print(f"\nTotal avalanches collected: {len(all_avalanches)}")
    if n_simulations > 0:
        print(f"Average avalanches per simulation: {len(all_avalanches)/n_simulations:.1f}")
    
    # Plot compiled Figure 2
    if len(all_avalanches) > 0:
        plot_figure_2_compiled(all_avalanches, NE=200, save_path=save_path)
    else:
        print("No avalanches found - skipping Figure 2")