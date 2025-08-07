#!/usr/bin/env python
"""
checkpoint_saver.py - Simple checkpoint saving module for SORN
Add this to your existing SORN simulation with minimal changes
Place this file in the common folder
"""

from __future__ import division, print_function
import os
import time
from datetime import datetime

# Handle Python 2/3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle

class CheckpointSaver:
    """Simple checkpoint saver that can be added to any SORN simulation"""
    
    def __init__(self, base_dir="../backup", experiment_name="sorn_run", interval=100000):
        """
        Initialize checkpoint saver
        
        Args:
            base_dir: Directory to save checkpoints
            experiment_name: Name for this experiment run
            interval: Save every N timesteps
        """
        self.interval = interval
        self.counter = 0
        self.last_save = 0
        self.total_steps = 0
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, "{0}_{1}".format(experiment_name, timestamp))
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        print("[CHECKPOINT] Save directory: {0}".format(self.save_dir))
        print("[CHECKPOINT] Will save every {0} steps".format(interval))
        
    def check_and_save(self, sorn, current_step=None, phase=""):
        """
        Check if we should save and do it if needed
        
        Args:
            sorn: The SORN instance
            current_step: Current timestep (if None, uses internal counter)
            phase: Optional phase name
        """
        if current_step is None:
            current_step = self.total_steps
            
        if current_step - self.last_save >= self.interval:
            self.save_now(sorn, current_step, phase)
            
    def save_now(self, sorn, step=None, phase=""):
        """Force save a checkpoint now"""
        if step is None:
            step = self.total_steps
            
        # Prepare data
        data = {
            'timestep': step,
            'phase': phase,
            'checkpoint_num': self.counter
        }
        
        # Save network state
        try:
            # Weight matrices
            if hasattr(sorn, 'W_ee'):
                data['W_ee'] = sorn.W_ee.W if hasattr(sorn.W_ee, 'W') else sorn.W_ee
            if hasattr(sorn, 'W_ei'):
                data['W_ei'] = sorn.W_ei.W if hasattr(sorn.W_ei, 'W') else sorn.W_ei
            if hasattr(sorn, 'W_ie'):
                data['W_ie'] = sorn.W_ie.W if hasattr(sorn.W_ie, 'W') else sorn.W_ie
            if hasattr(sorn, 'W_eu'):
                data['W_eu'] = sorn.W_eu.W if hasattr(sorn.W_eu, 'W') else sorn.W_eu
                
            # Thresholds
            if hasattr(sorn, 'T_e'):
                data['T_e'] = sorn.T_e
            if hasattr(sorn, 'T_i'):
                data['T_i'] = sorn.T_i
                
            # Current activity
            if hasattr(sorn, 'X'):
                data['X'] = sorn.X
            if hasattr(sorn, 'Y'):
                data['Y'] = sorn.Y
                
        except Exception as e:
            print("[WARNING] Could not save full network state: {0}".format(e))
        
        # Save to file
        filename = "checkpoint_{0:04d}_step_{1:08d}.pkl".format(self.counter, step)
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=2)
            
        file_size = os.path.getsize(filepath) / (1024.0 * 1024.0)
        print("[SAVED] Checkpoint {0} at step {1} ({2:.1f} MB) - {3}".format(
            self.counter, step, file_size, phase if phase else "running"))
        
        # Update counters
        self.counter += 1
        self.last_save = step
        
        # Update status file
        status_file = os.path.join(self.save_dir, "status.txt")
        with open(status_file, 'w') as f:
            f.write("Last checkpoint: {0}\n".format(filename))
            f.write("Step: {0}\n".format(step))
            f.write("Phase: {0}\n".format(phase))
            f.write("Time: {0}\n".format(datetime.now().isoformat()))
            
        return filepath
    
    def update_steps(self, steps):
        """Update internal step counter"""
        self.total_steps += steps
        self.check_and_save(None, self.total_steps)

# Global checkpoint saver instance
_global_saver = None

def init_checkpoint_saver(base_dir="../backup", experiment_name="sorn_run", interval=100000):
    """Initialize global checkpoint saver"""
    global _global_saver
    _global_saver = CheckpointSaver(base_dir, experiment_name, interval)
    return _global_saver

def save_checkpoint(sorn, step=None, phase=""):
    """Save checkpoint using global saver"""
    global _global_saver
    if _global_saver is not None:
        _global_saver.save_now(sorn, step, phase)

def check_checkpoint(sorn, step=None, phase=""):
    """Check and save if needed using global saver"""
    global _global_saver
    if _global_saver is not None:
        _global_saver.check_and_save(sorn, step, phase)

# ============================================
# MONKEY PATCH METHOD - Add to existing code
# ============================================

def add_checkpoint_to_sorn(sorn_instance, save_dir="../backup", interval=100000):
    """
    Monkey-patch checkpoint saving into an existing SORN instance
    
    Usage:
        sorn = Sorn(c, source)
        add_checkpoint_to_sorn(sorn)
    """
    # Initialize saver
    experiment_name = "sorn_experiment"
    saver = CheckpointSaver(save_dir, experiment_name, interval)
    
    # Store original simulation method
    original_simulation = sorn_instance.simulation
    
    # Create wrapped version
    def simulation_with_checkpoints(steps):
        """Modified simulation that saves checkpoints"""
        batch_size = 1000
        steps_done = 0
        
        while steps_done < steps:
            current_batch = min(batch_size, steps - steps_done)
            
            # Run original simulation
            original_simulation(current_batch)
            
            steps_done += current_batch
            saver.total_steps += current_batch
            
            # Check for checkpoint
            saver.check_and_save(sorn_instance, saver.total_steps)
            
            # Progress
            if steps_done % 10000 == 0:
                print("[PROGRESS] {0}/{1} steps completed".format(steps_done, steps))
    
    # Replace method
    sorn_instance.simulation = simulation_with_checkpoints
    sorn_instance._checkpoint_saver = saver
    
    print("[CHECKPOINT] Added checkpoint saving to SORN instance")
    return sorn_instance

# ============================================
# EXAMPLE USAGE IN YOUR EXISTING CODE
# ============================================
"""
Example 1: Minimal changes to test_single.py
-----------------------------------------
# At the top of your test_single.py, add:
from checkpoint_saver import init_checkpoint_saver, save_checkpoint

# After loading parameters:
param_name = os.path.basename(param_file).replace('.py', '')
saver = init_checkpoint_saver(experiment_name=param_name)

# In your experiment run loop, add periodic saves:
# (wherever you have access to the sorn object and know the current step)
save_checkpoint(sorn, current_step, "phase_name")


Example 2: Using monkey-patch method
------------------------------------
# After creating your SORN instance:
from checkpoint_saver import add_checkpoint_to_sorn

sorn = Sorn(c, source)
sorn = add_checkpoint_to_sorn(sorn, save_dir="../backup", interval=100000)
# Now simulation will automatically save checkpoints


Example 3: Manual control in experiment
---------------------------------------
from checkpoint_saver import CheckpointSaver

class Experiment_test(AbstractExperiment):
    def __init__(self, params):
        super(Experiment_test, self).__init__(params)
        self.saver = CheckpointSaver()
        
    def run(self, sorn):
        # Your phases...
        sorn.simulation(c.steps_transient)
        self.saver.save_now(sorn, c.steps_transient, "transient_complete")
        
        sorn.simulation(c.steps_plastic)
        self.saver.save_now(sorn, c.steps_transient + c.steps_plastic, "plastic_complete")
"""