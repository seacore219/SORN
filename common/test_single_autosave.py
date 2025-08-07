from __future__ import division  # has to be imported in every file
#~ import matplotlib  # if from ssh
#~ matplotlib.use('Agg')

import warnings
import tables
from importlib import import_module
import imp
import ipdb  # prettier debugger
import os
import sys
sys.path.insert(1, "../")

import sys
if 'win_unicode_console' in sys.modules:
    win_unicode_console = sys.modules['win_unicode_console']
    win_unicode_console.disable()

import utils

# This assumes that you are in the folder "common"
utils.initialise_backup(mount="../", dest="../backup")
utils.backup(__file__)

from utils.backup import dest_directory
from common.stats import StatsCollection
from common.sorn import Sorn
import cPickle as pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Get simulation number from environment variable
sim_number = int(os.environ.get('SORN_SIM_NUMBER', '1'))
print "[INFO] Running simulation number:", sim_number

# Set unique random seed for this simulation
np.random.seed(sim_number * 12345)
print "[INFO] Set random seed to:", sim_number * 12345

def load_param_file(filename):
    return imp.load_source('parameters', filename)

# Start debugging mode when an error is raised
def debugger(type,flag):
    print 'In debugger!'
    pass
np.seterrcall(debugger)
np.seterr(all='call')

# ========================================
# PERIODIC H5 SAVING AND ROTATION
# ========================================
SAVE_INTERVAL = 100000  # Save H5 every 100k timesteps
save_counter = [0]
total_steps = [0]

def rotate_h5_file(c):
    """Save current H5 and mark for deletion"""
    global save_counter
    
    # Close current H5 file
    try:
        import tables
        tables.file._open_files.close_all()
    except:
        pass
    
    # Current H5 file
    current_h5 = utils.logfilename("result.h5")
    
    if os.path.exists(current_h5):
        # Rename to chunk file
        chunk_h5 = os.path.join(
            c.logfilepath,
            "result_sim%02d_chunk%04d.h5" % (sim_number, save_counter[0])
        )
        
        # Move the file
        shutil.move(current_h5, chunk_h5)
        
        # Mark for deletion after sync
        marker_file = chunk_h5 + ".ready_to_delete"
        open(marker_file, 'a').close()
        
        file_size = os.path.getsize(chunk_h5) / (1024.0 * 1024.0)
        print "[ROTATE] Sim %d - Saved H5 chunk %d (%.1f MB) - ready for sync & delete" % (
            sim_number, save_counter[0], file_size
        )
        
        save_counter[0] += 1
        
        # Start new H5 file
        return True
    
    return False

# ========================================

# Parameters are read from the second command line argument
param = load_param_file(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module,experiment_name)(param)

c = param.c
c.logfilepath = utils.logfilename('')+'/'

(source,stats_single) = experiment.start()

sorn = Sorn(c,source)

# Create a StatsCollection and fill it with methods for all statistics
stats = StatsCollection(sorn)
stats.methods = stats_single
sorn.stats = stats

# ========================================
# WRAP SIMULATION FOR PERIODIC SAVING
# ========================================
original_simulation = sorn.simulation

def simulation_with_rotation(steps):
    """Modified simulation that rotates H5 files periodically"""
    global total_steps
    
    batch_size = 1000
    steps_done = 0
    
    while steps_done < steps:
        current_batch = min(batch_size, steps - steps_done)
        
        # Run original simulation
        original_simulation(current_batch)
        
        steps_done += current_batch
        total_steps[0] += current_batch
        
        # Rotate H5 file every SAVE_INTERVAL steps
        if total_steps[0] % SAVE_INTERVAL == 0:
            print "[INFO] Rotating H5 file at step %d" % total_steps[0]
            
            # Save and close current H5
            if hasattr(sorn.stats, 'dlog'):
                sorn.stats.dlog.close()
            
            # Rotate the file
            rotate_h5_file(c)
            
            # Reopen new H5
            sorn.stats.dlog.set_handler('*', utils.StoreToH5, utils.logfilename("result.h5"))
            
            print "[INFO] New H5 file started"
        
        # Progress update
        # Progress update every 10k timesteps
        if total_steps[0] % 10000 == 0 and total_steps[0] > 0:
            print "[Sim %d] Timesteps completed: %d" % (sim_number, total_steps[0])

# Replace simulation method
sorn.simulation = simulation_with_rotation
print "[INFO] H5 rotation enabled for simulation %d" % sim_number

# ========================================

# Datalog is used to store all results and parameters
stats.dlog.set_handler('*',utils.StoreToH5,  utils.logfilename("result.h5"))
stats.dlog.append('c', utils.unbunchify(c))
stats.dlog.set_handler('*',utils.TextPrinter)

# Final experimental preparations
experiment.reset(sorn)

# Start stats
sorn.stats.start()
sorn.stats.clear()

# Run experiment once
print "[INFO] Starting experiment..."
pickle_objects = experiment.run(sorn)

# Save final H5 chunk
print "[INFO] Saving final H5 chunk"
if hasattr(sorn.stats, 'dlog'):
    sorn.stats.dlog.close()
rotate_h5_file(c)

# Save sources etc (these are small)
for key in pickle_objects:
    filename = os.path.join(c.logfilepath,"%s.pickle"%key)
    topickle = pickle_objects[key]
    pickle.dump(topickle, gzip.open(filename,"wb"), pickle.HIGHEST_PROTOCOL)

# Control: Firing-rate model
if sorn.c.stats.control_rates:
    experiment.control_rates(sorn)

# Report stats and close
stats.single_report()
stats.disable = True

# Save network final state
sorn.quicksave(filename=os.path.join(c.logfilepath,'net.pickle'))

# Create manifest
manifest_file = os.path.join(c.logfilepath, "manifest_sim%02d.txt" % sim_number)
with open(manifest_file, 'w') as f:
    f.write("Simulation %d complete\n" % sim_number)
    f.write("Total H5 chunks: %d\n" % save_counter[0])
    f.write("Total timesteps: %d\n" % total_steps[0])

print "[INFO] Simulation %d complete - %d H5 chunks created" % (sim_number, save_counter[0])

# Note: Skip plotting since we have multiple H5 files