from __future__ import division
import numpy as np
import utils
utils.backup(__file__)
from common.defaults import *
import random

c.N_e = 200
c.N_i = int(np.floor(0.2*c.N_e))
c.N = c.N_e + c.N_i
c.N_u_e = int(np.floor(0.05*c.N_e))
c.N_u_i = 0

c.W_ee = utils.Bunch(use_sparse=True,
                     lamb = 0.1*c.N_e,
                     avoid_self_connections=True,
                     eta_stdp = 0.004,
                     sp_prob =  c.N_e*(c.N_e-1)*(0.1/(200*199)),
                     sp_initial = 0.001,
                     no_prune = False,
                     upper_bound = 1
                     )

c.W_ei = utils.Bunch(use_sparse=False,
                     lamb=0.2*c.N_e,
                     avoid_self_connections=True,
                     eta_istdp = 0.001,
                     h_ip=0.1)

c.W_ie = utils.Bunch(use_sparse=False,
                     lamb=1.0*c.N_i,
                     avoid_self_connections=True)

c.steps_plastic = 6000000
c.steps_avalanches = 0000000
c.steps_readouttrain = 100000
c.steps_readouttest = 100000

c.N_steps =  c.steps_plastic + c.steps_readouttrain + \
                               c.steps_readouttest + c.steps_avalanches
c.C_steps = c.steps_readouttrain + c.steps_readouttest

c.eta_ip = 0.01
c.h_ip = 0.02
c.noise_sig = np.sqrt(0.05)
c.noise_sig_e = np.sqrt(0.05)
c.noise_sig_i = np.sqrt(0.05)
c.noise_fire = 0
c.noise_fire_struc = 0


c.stats.file_suffix = 'test'
c.display = True
#c.stats.only_last_spikes = 1000 # number of last spikes to save
c.stats.only_last_spikes = c.steps_plastic + c.steps_readouttrain + c.steps_readouttest + c.steps_avalanches
#c.stats.only_last_spikes = c.steps_avalanches
c.stats.save_spikes = True
c.input_gain = 1 # needs to be 1! 

c.experiment.module = 'delpapa.experiment_MyCountingLearningTask'
c.experiment.name = 'Experiment_test'
