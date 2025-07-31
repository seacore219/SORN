from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from delpapa.plot import plot_results_perturbation as plot_results_single

from common.sources import NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_test(AbstractExperiment):
    def start(self):
        super(Experiment_test,self).start()
        c = self.params.c

        self.inputsource = NoSource()

        stats_single = [
                         ActivityStat(),
                         SpikesStat(),
                         SpikesInhStat(),
                         ConnectionFractionStat(),
                        ]
        return (self.inputsource,stats_single)

    def reset(self,sorn):
        super(Experiment_test,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats

    def run(self,sorn):
        super(Experiment_test,self).run(sorn)
        c = self.params.c


        # Initial pahse with plsaticity
        print("Initial Phase:")
        print("Current input source:", self.inputsource.__class__.__name__)
        sorn.simulation(c.steps_plastic)

        newseed = randint(999999) # random seed
        tmpstats = sorn.stats
        sorn.stats = 0
        filename = utils.logfilename("net_before_pert.pickle")
        sorn.quicksave(filename)
        sorn.stats = tmpstats

        # Reset point: next line runs the 'normal' SORN
        # print '\n\nNon-frozen steps:'
        # seed(newseed)
        # sorn.simulation(c.steps_perturbation)


        # Return to the reset point, freezes plasticity and run again
        print '\n\Perturbation steps:'
        #sorn = sorn.quickload(filename)
        #sorn.stats = tmpstats
        #sorn.stats.obj = sorn

        # Freeze plasticity
        # comment a line NOT to freeze a specific plasticity mechanism
        # sorn.W_ee.c.eta_stdp = 0       # freezes STDP
        # sorn.W_ei.c.eta_istdp = 0      # freezes iSTDP
        # sorn.W_ee.c.sp_prob = 0        # freezes SP
        # c.eta_ip = 0                   # freezes IP
        # c.T_e_max = 0.01               # reset T_e_max 
        # c.noise_sig = np.sqrt(0.08)    # freezes noise
        if hasattr(self.params, 'perturbation_source'):
            print "Switching to perturbation_source"
            self.inputsource = self.params.perturbation_source
            sorn.source = self.inputsource

            # Regenerate input connections for the new source
            sorn.W_eu = self.inputsource.generate_connection_e(c.N_e)
            sorn.W_iu = self.inputsource.generate_connection_i(c.N_i)
            print "Now using:", self.inputsource.__class__.__name__
            
        # Increase internal noise
        c.noise_sig = np.sqrt(0.05)
        c.noise_sig_e = np.sqrt(0.05)
        c.noise_sig_i = np.sqrt(0.05)
        
        seed(newseed)
        sorn.simulation(c.steps_perturbation)

        return {'source_plastic':self.inputsource}

    def plot_single(self,path,filename):
        plot_results_single(path,filename)
