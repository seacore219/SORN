#!/usr/bin/env python
"""
Modified experiment_ExtraInputNew.py with checkpoint saving support
This version adds checkpoint saving capabilities to the ExtraInput experiment
"""

from __future__ import division, print_function
from pylab import *
import os
import time
from datetime import datetime

# Handle Python 2/3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Safely handle utils.backup
try:
    import utils
    # Only call backup if __file__ is defined
    try:
        utils.backup(__file__)
    except NameError:
        pass  # __file__ not defined, skip backup
except ImportError:
    pass  # utils not available

# Import plotting and sources
from delpapa.plot import plot_results as plot_results_single
from common.sources import CountingSource, NoSource
from common.experiments import AbstractExperiment
from common.sorn_stats import *

class CheckpointManager:
    """Manages checkpoint saving during experiment phases"""
    
    def __init__(self, save_dir=None, interval=100000):
        self.save_dir = save_dir
        self.interval = interval
        self.checkpoint_count = 0
        self.enabled = save_dir is not None
        
        if self.enabled and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def save_phase_checkpoint(self, sorn, phase_name, step, data=None):
        """Save checkpoint for a specific phase"""
        if not self.enabled:
            return
            
        filename = "phase_{0}_checkpoint_{1:04d}_step_{2:08d}.pkl".format(
            phase_name, self.checkpoint_count, step)
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint_data = {
            'phase': phase_name,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'network_state': self.extract_network_state(sorn),
            'additional_data': data
        }
        
        with open(filepath, 'wb') as f:
            cPickle.dump(checkpoint_data, f, protocol=2)
        
        self.checkpoint_count += 1
        print("[CHECKPOINT] Saved {0} at step {1}".format(phase_name, step))
        
        return filepath
    
    def extract_network_state(self, sorn):
        """Extract current network state from SORN"""
        state = {}
        
        # Weight matrices
        if hasattr(sorn, 'W_ee'):
            state['W_ee'] = sorn.W_ee.W if hasattr(sorn.W_ee, 'W') else None
        if hasattr(sorn, 'W_ei'):
            state['W_ei'] = sorn.W_ei.W if hasattr(sorn.W_ei, 'W') else None
        if hasattr(sorn, 'W_ie'):
            state['W_ie'] = sorn.W_ie.W if hasattr(sorn.W_ie, 'W') else None
        if hasattr(sorn, 'W_eu'):
            state['W_eu'] = sorn.W_eu.W if hasattr(sorn.W_eu, 'W') else sorn.W_eu
            
        # Thresholds
        if hasattr(sorn, 'T_e'):
            state['T_e'] = sorn.T_e
        if hasattr(sorn, 'T_i'):
            state['T_i'] = sorn.T_i
            
        # Activity
        if hasattr(sorn, 'X'):
            state['X'] = sorn.X
        if hasattr(sorn, 'Y'):
            state['Y'] = sorn.Y
            
        return state

class Experiment_test(AbstractExperiment):
    
    def __init__(self, params):
        super(Experiment_test, self).__init__(params)
        
        # Initialize checkpoint manager if save path is specified
        save_dir = None
        if hasattr(params.c, 'checkpoint_dir'):
            save_dir = params.c.checkpoint_dir
        elif hasattr(params.c, 'logfilepath'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(params.c.logfilepath, 
                                   "checkpoints_{0}".format(timestamp))
        
        self.checkpoint_mgr = CheckpointManager(save_dir, interval=100000)
        self.total_steps = 0
    
    def start(self):
        super(Experiment_test, self).start()
        c = self.params.c
        self.inputsource = NoSource(N_i=c.N_u_e)
        
        stats_single = [
            ActivityStat(),
            SpikesStat(),              
            ConnectionFractionStat(),
        ]
        
        return (self.inputsource, stats_single)
    
    def reset(self, sorn):
        super(Experiment_test, self).reset(sorn)
        c = self.params.c
        stats = sorn.stats
        sorn.__init__(c, self.inputsource)
        sorn.stats = stats
    
    def run_with_checkpoints(self, sorn, steps, phase_name):
        """Run simulation with periodic checkpoint saving"""
        
        print('\n[PHASE] {0}: {1} steps'.format(phase_name, steps))
        
        batch_size = 10000  # Run in batches
        steps_done = 0
        phase_start_time = time.time()
        
        while steps_done < steps:
            # Calculate batch size
            current_batch = min(batch_size, steps - steps_done)
            
            # Run simulation batch
            sorn.simulation(current_batch)
            
            steps_done += current_batch
            self.total_steps += current_batch
            
            # Check if we should save checkpoint
            if self.total_steps % self.checkpoint_mgr.interval == 0:
                self.checkpoint_mgr.save_phase_checkpoint(
                    sorn, phase_name, self.total_steps)
            
            # Progress update
            if steps_done % 50000 == 0 and steps_done > 0:
                elapsed = time.time() - phase_start_time
                rate = steps_done / elapsed
                eta = (steps - steps_done) / rate
                print("  Progress: {0}/{1} steps ({2:.1f}%), "
                      "Rate: {3:.0f} steps/sec, ETA: {4:.1f} min".format(
                    steps_done, steps, 100.0*steps_done/steps,
                    rate, eta/60.0))
        
        phase_elapsed = time.time() - phase_start_time
        print("  Phase complete in {0:.1f} minutes".format(phase_elapsed/60.0))
    
    def run(self, sorn):
        super(Experiment_test, self).run(sorn)
        c = self.params.c
        
        # Phase 1: Transient
        print('\n\n=== Phase 1: Transient ===')
        self.run_with_checkpoints(sorn, c.steps_transient, "transient")
        
        # Phase 2: No External Input (plastic phase)
        print('\n\n=== Phase 2: No External Input (Plastic) ===')
        self.run_with_checkpoints(sorn, c.steps_noExternalInput, "no_external_input")
        
        # Save checkpoint before changing to external input
        self.checkpoint_mgr.save_phase_checkpoint(
            sorn, "before_external_input", self.total_steps,
            data={'phase_transition': 'switching_to_external_input'})
        
        # Phase 3: External Input
        print('\n\n=== Phase 3: External Input ===')
        
        # External input definition (same as original)
        words = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        words_len = len(words)
        words_trans = np.ones([words_len, words_len]) * (1./words_len)
        
        newsource = CountingSource(words, words_trans,
                                 c.N_u_e, c.N_u_i, avoid=False)
        sorn.source = newsource
        sorn.W_eu = newsource.generate_connection_e(c.N_e)
        
        # Run with external input
        self.run_with_checkpoints(sorn, c.steps_ExternalInput, "external_input")
        
        # Final checkpoint
        final_data = {
            'source_plastic': self.inputsource,
            'source_test': newsource,
            'total_steps': self.total_steps,
            'experiment_complete': True
        }
        
        self.checkpoint_mgr.save_phase_checkpoint(
            sorn, "final", self.total_steps, data=final_data)
        
        print("\n=== Experiment Complete ===")
        print("Total steps: {0}".format(self.total_steps))
        print("Checkpoints saved: {0}".format(self.checkpoint_mgr.checkpoint_count))
        
        return {'source_plastic': self.inputsource, 'source_test': newsource}
    
    def plot_single(self, path, filename):
        plot_results_single(path, filename)

# For backward compatibility - create an alias
Experiment = Experiment_test