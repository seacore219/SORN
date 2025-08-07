#!/usr/bin/env python
"""
simple_checkpoint_wrapper.py - Minimal wrapper to add checkpoints to SORN
Place in common folder and run like:
    python simple_checkpoint_wrapper.py ../delpapa/param_ExtraInputNew.py
"""

from __future__ import division, print_function
import os
import sys
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, "../")

# Import pickle
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Import the EXACT same way as test_single.py does
import utils
from sorn import Sorn
from stats import StatsCollection

# Get parameter file
if len(sys.argv) < 2:
    print("Usage: python simple_checkpoint_wrapper.py <param_file>")
    sys.exit(1)

param_file = sys.argv[1]
print("Loading parameters from:", param_file)

# Load parameters using the exact same method as test_single.py
from importlib import import_module
param = import_module(utils.param_file())

# Get experiment the same way
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module, experiment_name)(param)

c = param.c

# Create checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = "../backup/checkpoints_{0}".format(timestamp)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print("Checkpoint directory:", checkpoint_dir)

# Initialize experiment
(source, stats_single) = experiment.start()

# Create SORN
print("Creating SORN with N_e={0}, N_i={1}".format(c.N_e, c.N_i))
sorn = Sorn(c, source)

# Create stats
stats = StatsCollection(sorn)
stats.methods = stats_single
sorn.stats = stats

# Checkpoint saving function
checkpoint_counter = [0]
total_steps = [0]

def save_checkpoint(phase=""):
    """Save a checkpoint"""
    checkpoint_file = os.path.join(checkpoint_dir, 
        "checkpoint_{0:04d}_step_{1:08d}.pkl".format(checkpoint_counter[0], total_steps[0]))
    
    data = {
        'step': total_steps[0],
        'phase': phase,
        'W_ee': sorn.W_ee.W if hasattr(sorn.W_ee, 'W') else None,
        'W_ei': sorn.W_ei.W if hasattr(sorn.W_ei, 'W') else None,
        'W_ie': sorn.W_ie.W if hasattr(sorn.W_ie, 'W') else None,
        'T_e': sorn.T_e,
        'T_i': sorn.T_i
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f, protocol=2)
    
    print("[CHECKPOINT {0}] Saved at step {1} - {2}".format(
        checkpoint_counter[0], total_steps[0], phase))
    checkpoint_counter[0] += 1

# Wrap the simulation method
original_simulation = sorn.simulation

def simulation_with_checkpoints(steps):
    """Run simulation with periodic checkpoints"""
    global total_steps
    
    batch_size = 1000
    checkpoint_interval = 100000
    
    steps_done = 0
    while steps_done < steps:
        batch = min(batch_size, steps - steps_done)
        original_simulation(batch)
        steps_done += batch
        total_steps[0] += batch
        
        # Check if we should save
        if total_steps[0] % checkpoint_interval == 0:
            save_checkpoint("running")
        
        # Progress
        if steps_done % 10000 == 0:
            print("Progress: {0}/{1} steps".format(steps_done, steps))

# Replace simulation method
sorn.simulation = simulation_with_checkpoints

# Reset experiment
experiment.reset(sorn)

# Start stats
sorn.stats.start()
sorn.stats.clear()

print("\nStarting experiment...")
start_time = time.time()

# Run experiment
try:
    results = experiment.run(sorn)
    save_checkpoint("final")
    print("\nExperiment completed!")
except Exception as e:
    print("Error:", e)
    save_checkpoint("error")
    raise

elapsed = time.time() - start_time
print("Total time: {0:.1f} minutes".format(elapsed/60.0))
print("Total steps: {0}".format(total_steps[0]))
print("Checkpoints saved in: {0}".format(checkpoint_dir))
