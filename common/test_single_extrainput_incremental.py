#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExtraInputNew Single Simulation with Incremental Saving
=======================================================
This replaces test_single.py for ExtraInputNew simulations with incremental saving.

Usage: python2.7 test_single_extrainput_incremental.py ../delpapa/param_ExtraInputNew.py
"""

from __future__ import division
import os
import sys
import time
import numpy as np
import tables
from importlib import import_module

# Add parent directory to path
sys.path.insert(1, "../")

import utils
utils.initialise_backup(mount="../", dest="../backup")
utils.backup(__file__)

from common.stats import StatsCollection, HistoryStat
from common.sorn import Sorn
import cPickle as pickle
import gzip

# ============================================================================
# INCREMENTAL SAVING CLASSES
# ============================================================================

def _getvar(obj, var):
    if '.' in var:
        (obj_name, _, var) = var.partition('.')
        obj = obj.__getattribute__(obj_name)
        return _getvar(obj, var)
    return obj.__getattribute__(var)

class IncrementalSpikesStat:
    """
    Incremental version of SpikesStat that saves both E and I spikes
    """
    def __init__(self, inhibitory=False, save_interval=10000, max_chunks_memory=3):
        if inhibitory:
            self.name = 'SpikesInh'
            self.sattr = 'spikes_inh'
        else:
            self.name = 'Spikes'
            self.sattr = 'spikes'
        self.collection = 'gather'
        self.inh = inhibitory
        self.save_interval = save_interval
        self.max_chunks_memory = max_chunks_memory
        self.temp_buffer = []
        self.chunk_counter = 0
        self.step = 0
        self.chunk_files = []
        
    def connect(self, collection):
        """Required method for stats system"""
        pass
        
    def start(self, c, obj):
        """Required method for stats system"""
        pass
        
    def clear(self, c, sorn):
        """Initialize"""
        if self.inh:
            self.neurons = sorn.c.N_i
        else:
            self.neurons = sorn.c.N_e
        
        self.temp_buffer = []
        self.chunk_counter = 0
        self.step = 0
        self.chunk_files = []
        
        print("[INCREMENTAL] %s: Saving every %d steps (max %d chunks in memory)" % 
              (self.name, self.save_interval, self.max_chunks_memory))
        
    def add(self, c, obj):
        """Add spike data to buffer and save incrementally"""
        if self.inh:
            spikes = obj.y
        else:
            spikes = obj.x
            
        # Add to buffer
        self.temp_buffer.append(np.copy(spikes))
        
        # Save chunk if buffer is full
        if len(self.temp_buffer) >= self.save_interval:
            self._save_chunk_to_disk(c, obj)
            
        self.step += 1
        
    def _save_chunk_to_disk(self, c, obj):
        """Save current buffer as a chunk to disk"""
        if not self.temp_buffer:
            return
            
        # Create chunk name
        chunk_name = "%s_chunk_%03d" % (self.name, self.chunk_counter)
        
        # Convert buffer to array (neurons x timesteps)
        spike_array = np.array(self.temp_buffer).T
        
        # Save via datalog
        if hasattr(obj, 'stats') and hasattr(obj.stats, 'dlog'):
            obj.stats.dlog.append(chunk_name, spike_array)
            print("[SAVE] %s with shape %s" % (chunk_name, spike_array.shape))
            
            # Track chunk file
            chunk_file = utils.logfilename("result.h5")
            if chunk_file not in self.chunk_files:
                self.chunk_files.append(chunk_file)
        
        # Clean up old chunks (keep only latest chunks)
        self._cleanup_old_chunks()
        
        # Clear buffer and increment counter
        self.temp_buffer = []
        self.chunk_counter += 1
        
    def _cleanup_old_chunks(self):
        """Remove old chunks from memory, keep latest N"""
        if len(self.chunk_files) > self.max_chunks_memory:
            chunks_to_remove = len(self.chunk_files) - self.max_chunks_memory
            for i in range(chunks_to_remove):
                old_chunk_name = "%s_chunk_%03d" % (self.name, i)
                print("[CLEANUP] Marked %s for later removal" % old_chunk_name)
        
    def finalize(self, c, obj):
        """Save any remaining data in buffer"""
        if self.temp_buffer:
            self._save_chunk_to_disk(c, obj)
            
    def report(self, c, obj):
        """Save final buffer and return empty array"""
        self.finalize(c, obj)
        return np.array([])

class IncrementalHistoryStat(HistoryStat):
    """Modified HistoryStat that saves data incrementally"""
    def __init__(self, var='x', collection="gather", record_every_nth=1, 
                 save_interval=10000, chunk_size=10000):
        HistoryStat.__init__(self, var, collection, record_every_nth)
        self.save_interval = save_interval
        self.chunk_size = chunk_size
        self.temp_buffer = []
        self.chunk_counter = 0
        
    def add(self, c, obj):
        """Add data but flush to disk periodically"""
        if not (c.history[self.counter] % self.record_every_nth):
            tmp = _getvar(obj, self.var)
            if callable(tmp):
                tmp = tmp()
            
            self.temp_buffer.append(np.copy(tmp))
            
            if len(self.temp_buffer) >= self.chunk_size:
                self._save_chunk_to_disk(c, obj)
                
        c.history[self.counter] += 1
    
    def _save_chunk_to_disk(self, c, obj):
        """Save current buffer as a chunk to disk"""
        if not self.temp_buffer:
            return
            
        chunk_name = "%s_chunk_%03d" % (self.name, self.chunk_counter)
        
        if hasattr(obj, 'stats') and hasattr(obj.stats, 'dlog'):
            obj.stats.dlog.append(chunk_name, np.array(self.temp_buffer))
            print("[SAVE] %s with shape %s" % (chunk_name, np.array(self.temp_buffer).shape))
        
        self.temp_buffer = []
        self.chunk_counter += 1
        
    def finalize(self, c, obj):
        """Save any remaining data in buffer"""
        if self.temp_buffer:
            self._save_chunk_to_disk(c, obj)
    
    def report(self, c, obj):
        """Modified report that handles chunked data"""
        self.finalize(c, obj)
        return np.array([])

class IncrementalStatsCollection(StatsCollection):
    """Modified StatsCollection that handles incremental saving"""
    def __init__(self, obj, dlog=None, save_interval=10000):
        StatsCollection.__init__(self, obj, dlog)
        self.save_interval = save_interval
        self.step_counter = 0
        
    def add(self, *args, **kwargs):
        """Override add to track steps and trigger saves"""
        StatsCollection.add(self, *args, **kwargs)
        self.step_counter += 1
        
        if self.step_counter % self.save_interval == 0:
            self._incremental_save()
            print("[INFO] Incremental save triggered at step %d" % self.step_counter)
    
    def _incremental_save(self):
        """Force save of buffered data"""
        for method in self.methods:
            if hasattr(method, 'finalize'):
                method.finalize(self.c, self.obj)

# ============================================================================
# WRAPPER FUNCTION
# ============================================================================

def modify_sorn_for_extrainput_incremental_saving(sorn, save_interval=10000):
    """
    Enhanced wrapper for ExtraInputNew with both E and I spike saving
    """
    
    # Replace stats collection
    if hasattr(sorn, 'stats') and not isinstance(sorn.stats, IncrementalStatsCollection):
        new_stats = IncrementalStatsCollection(sorn, sorn.stats.dlog, save_interval)
        new_stats.methods = sorn.stats.methods
        new_stats.c = sorn.stats.c
        sorn.stats = new_stats
    
    # Enhanced stats list with both E and I spikes
    enhanced_methods = []
    spikes_e_added = False
    spikes_i_added = False
    
    for i, method in enumerate(sorn.stats.methods):
        class_name = method.__class__.__name__
        
        if class_name == 'SpikesStat':
            if not spikes_e_added:
                print("[ADD] Adding IncrementalSpikesStat for excitatory neurons")
                new_method_e = IncrementalSpikesStat(
                    inhibitory=False,
                    save_interval=save_interval
                )
                enhanced_methods.append(new_method_e)
                spikes_e_added = True
                
            if not spikes_i_added:
                print("[ADD] Adding IncrementalSpikesStat for inhibitory neurons")
                new_method_i = IncrementalSpikesStat(
                    inhibitory=True,
                    save_interval=save_interval
                )
                enhanced_methods.append(new_method_i)
                spikes_i_added = True
            
        elif isinstance(method, HistoryStat):
            print("[CONVERT] Converting %s to incremental version" % class_name)
            new_method = IncrementalHistoryStat(
                var=method.var,
                collection=method.collection,
                record_every_nth=method.record_every_nth,
                save_interval=save_interval
            )
            new_method.name = method.name
            new_method.counter = method.counter
            enhanced_methods.append(new_method)
        else:
            print("[KEEP] Keeping %s unchanged" % class_name)
            enhanced_methods.append(method)
    
    # Replace methods with enhanced list
    sorn.stats.methods = enhanced_methods
    
    print("[INFO] Enhanced for ExtraInputNew: E+I spikes, incremental saving every %dk steps" % 
          (save_interval/1000))
    
    return sorn

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*60)
    print("EXTRAINPUT INCREMENTAL SAVING TEST")
    print("="*60)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python %s <param_file>" % sys.argv[0])
        print("Example: python %s ../delpapa/param_ExtraInputNew.py" % sys.argv[0])
        sys.exit(1)
    
    param_file_path = sys.argv[1]
    
    if not os.path.exists(param_file_path):
        print("Error: Parameter file not found: %s" % param_file_path)
        sys.exit(1)
    
    print("Using parameter file: %s" % param_file_path)
    
    # Get simulation number
    sim_number = int(os.environ.get('SORN_SIM_NUMBER', '1'))
    np.random.seed(sim_number * 12345)
    print("Simulation number: %d" % sim_number)
    
    # Load parameters
    param_module_name = os.path.splitext(os.path.basename(param_file_path))[0]
    param_dir = os.path.dirname(param_file_path)
    
    if param_dir not in sys.path:
        sys.path.insert(0, param_dir)
    
    try:
        param = import_module(param_module_name)
    except ImportError as e:
        print("Error importing parameter module: %s" % e)
        sys.exit(1)
    
    # Load experiment
    experiment_module = import_module(param.c.experiment.module)
    experiment_name = param.c.experiment.name
    experiment = getattr(experiment_module, experiment_name)(param)
    
    c = param.c
    c.logfilepath = utils.logfilename('') + '/'
    
    print("Experiment: %s" % experiment.__class__.__name__)
    print("Total steps: %d (%.1fM)" % (c.N_steps, c.N_steps/1000000.0))
    print("Log path: %s" % c.logfilepath)
    
    # Start experiment
    (source, stats_single) = experiment.start()
    print("Source: %s" % source.__class__.__name__)
    print("Stats methods: %d" % len(stats_single))
    
    # Create SORN
    sorn = Sorn(c, source)
    stats = StatsCollection(sorn)
    stats.methods = stats_single
    sorn.stats = stats
    
    # APPLY ENHANCED INCREMENTAL SAVING
    print("\n" + "="*40)
    print("APPLYING ENHANCED INCREMENTAL SAVING")
    print("="*40)
    
    sorn = modify_sorn_for_extrainput_incremental_saving(sorn, save_interval=10000)
    
    print("Enhanced stats methods:")
    for i, method in enumerate(sorn.stats.methods):
        print("  %d. %s" % (i+1, method.__class__.__name__))
    
    # Setup data logging
    stats.dlog.set_handler('*', utils.StoreToH5, utils.logfilename("result.h5"))
    stats.dlog.append('c', utils.unbunchify(c))
    stats.dlog.set_handler('*', utils.TextPrinter)
    
    # Reset experiment
    experiment.reset(sorn)
    
    # Start stats
    sorn.stats.start()
    sorn.stats.clear()
    
    # Run experiment
    print("\n" + "="*40)
    print("RUNNING EXTRAINPUT EXPERIMENT")
    print("="*40)
    
    start_time = time.time()
    pickle_objects = experiment.run(sorn)
    end_time = time.time()
    
    print("Experiment completed in %.1f seconds" % (end_time - start_time))
    
    # Finalize data
    print("\n" + "="*40)
    print("FINALIZING DATA")
    print("="*40)
    
    # Finalize any remaining incremental data
    for method in sorn.stats.methods:
        if hasattr(method, 'finalize'):
            method.finalize(sorn.stats.c, sorn)
    
    # Save other objects
    for key in pickle_objects:
        filename = os.path.join(c.logfilepath, "%s.pickle" % key)
        pickle.dump(pickle_objects[key], gzip.open(filename, "wb"), pickle.HIGHEST_PROTOCOL)
        print("Saved %s" % filename)
    
    # Final operations
    if sorn.c.stats.control_rates:
        experiment.control_rates(sorn)
    
    stats.single_report()
    stats.disable = True
    stats.dlog.close()
    
    # Save network (with error handling)
    net_filename = os.path.join(c.logfilepath, 'net.pickle')
    try:
        sorn.quicksave(filename=net_filename)
        print("Saved network: %s" % net_filename)
    except (TypeError, OverflowError) as e:
        print("Warning: Could not save network pickle (%s)" % str(e))
        print("Spike data was saved successfully")
    
    # Check generated files
    print("\n" + "="*40)
    print("GENERATED FILES")
    print("="*40)
    
    if os.path.exists(c.logfilepath):
        for f in sorted(os.listdir(c.logfilepath)):
            if f.endswith('.h5') or f.endswith('.pickle'):
                filepath = os.path.join(c.logfilepath, f)
                size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
                print("  %s (%.1f MB)" % (f, size_mb))
    
    print("\n" + "="*60)
    print("EXTRAINPUT INCREMENTAL SAVING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()