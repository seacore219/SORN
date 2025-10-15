from __future__ import division
import numpy as np
import utils
utils.backup(__file__)

from common.defaults import *
import random

# Network parameters
c.N_e = 200
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i

c.N_u_e = int(np.floor(0.06*c.N_e))  # 6% of excitatory neurons
c.N_u_i = 0

# Connection parameters (matching original)
c.W_ee = utils.Bunch(use_sparse=True,
                     lamb = 0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.004,
                     sp_prob = c.N_e*(c.N_e-1)*(0.1/(200*199)),
                     sp_initial = 0.001,
                     no_prune = False,
                     upper_bound = 1)

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=0.2*c.N_e,
                     avoid_self_connections=True,
                     eta_istdp = 0.001,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=1.0*c.N_i,
                     avoid_self_connections=True)

# Use same as original but scale up for proper statistics
section_steps = 2000  # 2M steps per section
c.steps_transient = section_steps
c.steps_noExternalInput = section_steps  
c.steps_ExternalInput = section_steps*4   

c.N_steps = c.steps_transient + c.steps_noExternalInput + c.steps_ExternalInput

# Plasticity parameters
c.eta_ip = 0.01
c.h_ip = 0.06
c.eta_ip = 0.0001

# Noise parameters
c.noise_sig = np.sqrt(0.05)
c.noise_sig_e = np.sqrt(0.05)
c.noise_sig_i = np.sqrt(0.05)

# Input parameters
c.input_gain = 100000000  # Very high gain
c.noise_fire = 0
c.noise_fire_struc = 0

# Stats parameters
c.stats.file_suffix = 'ExtraInput'
c.display = True
c.stats.save_spikes = True

# Experiment configuration
c.experiment.module = 'delpapa.experiment_ExtraInputNew'
c.experiment.name = 'Experiment_test'