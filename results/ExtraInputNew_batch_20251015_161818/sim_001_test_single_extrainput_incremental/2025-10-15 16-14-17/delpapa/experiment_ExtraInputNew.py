from __future__ import division
from pylab import *
import utils
utils.backup(__file__)

from delpapa.plot import plot_results as plot_results_single
from common.sources import CountingSource, NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class Experiment_test(AbstractExperiment):
    
    def start(self):
        super(Experiment_test,self).start()
        c = self.params.c
        self.inputsource = NoSource(N_i=c.N_u_e)
        
        stats_single = [
            ActivityStat(),
            SpikesStat(),              
            ConnectionFractionStat(),
        ]
        
        return (self.inputsource, stats_single)
    
    def reset(self,sorn):
        super(Experiment_test,self).reset(sorn)
        c = self.params.c
        stats = sorn.stats
        sorn.__init__(c,self.inputsource)
        sorn.stats = stats
        
    def run(self,sorn):
        super(Experiment_test,self).run(sorn)
        c = self.params.c
        
        # Run phases quietly
        sorn.simulation(c.steps_transient)
        sorn.simulation(c.steps_noExternalInput)
        
        # External input
        words = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        words_len = len(words)
        words_trans = np.ones([words_len,words_len])*(1./words_len)
        
        newsource = CountingSource(words, words_trans,
                                   c.N_u_e, c.N_u_i, avoid=False)
        sorn.source = newsource
        sorn.W_eu = newsource.generate_connection_e(c.N_e)
        sorn.simulation(c.steps_ExternalInput)
        
        return {'source_plastic':self.inputsource,'source_test':newsource}
    
    def plot_single(self,path,filename):
        plot_results_single(path,filename)