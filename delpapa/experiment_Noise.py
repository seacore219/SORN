from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from delpapa.plot import plot_results_perturbation as plot_results_single

from common.sources import RandomSource, GatedSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_test(AbstractExperiment):
    def start(self):
        super(Experiment_test,self).start()
        c = self.params.c
        
        print("[DEBUG] Creating RandomSource")
        self.inputsource = RandomSource(
            firing_rate=0.05,
            N_neurons=c.N_u_e,
            connection_density=0.1,
            eta_stdp=c.W_eu.eta_stdp if hasattr(c, 'W_eu') else 0.004,
        )
        print("[DEBUG] RandomSource created")

        stats_single = [
                         ActivityStat(),
                         SpikesStat(),
                         ConnectionFractionStat(),
                        ]
        print("[DEBUG] RandomSource setup complete (no GatedSource used)")
        return (self.inputsource,stats_single)
        
    
    def reset(self,sorn):
        print("[DEBUG] Before experiment.reset()")
        super(Experiment_test,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        print("[DEBUG] After sorn.__init__()")
        print("[DEBUG] SORN.source after init:", type(sorn.source))
        sorn.stats = stats

    def run(self,sorn):
        super(Experiment_test,self).run(sorn)
        c = self.params.c
        print '\nInput plastic period:'
        sorn.simulation(c.steps_plastic, start_step=0)

        # Initial pahse with plsaticity
        print("Initial Phase:")
        sorn.simulation(c.steps_plastic)
        newseed = randint(999999) # random seed
        print("Now running simulation...:")

        tmpstats = sorn.stats
        sorn.stats = 0
        filename = utils.logfilename("net_before_pert.pickle")
        sorn.quicksave(filename)
        sorn.stats = tmpstats

        # Reset point: next line runs the 'normal' SORN
        # seed(newseed)
        # sorn.simulation(c.steps_perturbation)
        print '\nPerturbation (noise) period:'

        # Return to the reset point, freezes plasticity and run again
        print '\n\nFrozen steps:'
        #sorn = sorn.quickload(filename)0q23
        #sorn.stats = tmpstats
        #sorn.stats.obj = sorn

        # Freeze plasticity
        # comment a line NOT to freeze a specific plasticity mechanism
        # sorn.W_ee.c.eta_stdp = 0       # freezes STDP
        # sorn.W_ei.c.eta_istdp = 0      # freezes iSTDP
        # sorn.W_ee.c.sp_prob = 0        # freezes SP
        # c.eta_ip = 0                   # freezes IP
        #c.noise_sig_e = np.sqrt(0.05)   # freezes noise
        #c.noise_sig_i = np.sqrt(0.05)   # freezes noise

        seed(newseed)
        sorn.simulation(c.steps_perturbation)
        return {'source_plastic':self.inputsource}

    def plot_single(self,path,filename):
        plot_results_single(path,filename)
