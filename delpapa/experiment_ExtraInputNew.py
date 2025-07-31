from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from delpapa.plot import plot_results as plot_results_single
from common.sources import NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_test(AbstractExperiment):
    
    def start(self):
        super(Experiment_test,self).start()
        c = self.params.c
        # Use NoSource throughout - no external input
        self.inputsource = NoSource()
        
        stats_single = [
            ActivityStat(),
            SpikesStat(),              # Excitatory spikes
            SpikesStat(inhibitory=True),  # Inhibitory spikes
            ConnectionFractionStat(),
        ]
        
        return (self.inputsource, stats_single)
    
    def reset(self,sorn):
        super(Experiment_test,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats  # init sets sorn.stats to None
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats
        
    def run(self,sorn):
        super(Experiment_test,self).run(sorn)
        c = self.params.c
        
        print '\n\nTransient phase (initial settling)...'
        sorn.simulation(c.steps_transient)
        
        print '\n\nNormal phase (before perturbation):'
        sorn.simulation(c.steps_noExternalInput)
        
        print '\n\nPerturbation phase (increased noise):'
        
        # Instead of external input, increase internal noise
        # This is what creates the perturbation for Figure 6
        old_noise_e = c.noise_sig_e
        old_noise_i = c.noise_sig_i
        
        # Increase noise significantly
        c.noise_sig_e = np.sqrt(0.2)  # Increased from sqrt(0.05)
        c.noise_sig_i = np.sqrt(0.2)
        
        print("Increased noise levels:")
        print("  E noise:", c.noise_sig_e)
        print("  I noise:", c.noise_sig_i)
        
        # Run with increased noise
        sorn.simulation(c.steps_ExternalInput)
        
        # Restore original noise (for any future use)
        c.noise_sig_e = old_noise_e
        c.noise_sig_i = old_noise_i
        
        return {'source':self.inputsource}
    
    def plot_single(self,path,filename):
        plot_results_single(path,filename)