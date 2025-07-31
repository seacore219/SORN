from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

import random as randomstr

from delpapa.plot import plot_results as plot_results_single

from common.sources import CountingSource, NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

#n_middle = 4
n_middle = 0

class Experiment_test(AbstractExperiment):
    def start(self):
        super(Experiment_test,self).start()
        c = self.params.c

        self.inputsource = NoSource(N_i=c.N_u_e)

        stats_single = [
                         ActivityStat(),
                         CountingLetterStat(),
                         CountingActivityStat(),
                         ConnectionFractionStat(),
                         SpikesStat()
                        ]
        return (self.inputsource, stats_single)

    def reset(self,sorn):
        super(Experiment_test,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats

    def run(self,sorn):
        super(Experiment_test,self).run(sorn)
        c = self.params.c

        # 1. No Source - just plasticity
        print '\nInput plastic period:'
        sorn.simulation(c.steps_plastic)

        # Create new source
        word1 = 'A'
        word2 = 'D'
        for i in range(n_middle):
            word1 += 'B'
            word2 += 'E'
        word1 += 'C'
        word2 += 'F'
        m_trans = np.ones((2,2))*0.5

        newsource = CountingSource([word1, word2],m_trans,
                           c.N_u_e,c.N_u_i,avoid=False)
        sorn.source = newsource
        sorn.W_eu = newsource.generate_connection_e(c.N_e)


        # # Turn off noise
        # c.noise_sig = np.sqrt(0.0)
        # c.noise_sig_e = np.sqrt(0.0)
        # c.noise_sig_i = np.sqrt(0.0)

        # Turn off plasticity
        # sorn.W_ee.c.eta_stdp = 0
        # sorn.W_ei.c.eta_istdp = 0
        # sorn.W_ee.c.sp_prob = 0
        # c.eta_ip = 0
        
        # 2. Turn on source - compute avalanches
        print '\nAvalanche measurement period:'
        sorn.simulation(c.steps_avalanches)

        # Turn off noise
        c.noise_sig = np.sqrt(0.0)
        c.noise_sig_e = np.sqrt(0.0)
        c.noise_sig_i = np.sqrt(0.0)

        # Turn off plasticity
        sorn.W_ee.c.eta_stdp = 0
        sorn.W_ei.c.eta_istdp = 0
        sorn.W_ee.c.sp_prob = 0
        c.eta_ip = 0

        # 3. Train readout
        print '\nInput train period:'
        sorn.simulation(c.steps_readouttrain)

        # 4. Test readout
        print '\nInput test period:'
        sorn.simulation(c.steps_readouttest)

        return {'source_plastic':self.inputsource,'source_test':newsource}

    def plot_single(self,path,filename):
        plot_results_single(path,filename)
