from __future__ import division
import os
import sys
sys.path.insert(1, "../")

import utils
utils.initialise_backup(mount="../", dest="../backup")
utils.backup(__file__)

from importlib import import_module
from common.stats import StatsCollection
from common.sorn import Sorn
import cPickle as pickle
import gzip
import numpy as np
import tables

# Get simulation number
sim_number = int(os.environ.get('SORN_SIM_NUMBER', '1'))
np.random.seed(sim_number * 12345)

# Silence debug errors
def debugger(type,flag):
    pass
np.seterrcall(debugger)
np.seterr(all='call')

# ========================================
# CHUNKED H5 SAVING
# ========================================
CHUNK_SIZE = 10000  # Save every 10k timesteps
chunk_counter = [0]
total_steps = [0]
h5_chunks = []

def save_h5_chunk(c):
    """Save current H5 as a chunk and start new one"""
    global chunk_counter, h5_chunks
    
    # Close current H5
    tables.file._open_files.close_all()
    
    current_h5 = utils.logfilename("result.h5")
    if os.path.exists(current_h5):
        # Rename to chunk
        chunk_name = "chunk_%03d.h5" % chunk_counter[0]
        chunk_path = os.path.join(c.logfilepath, chunk_name)
        os.rename(current_h5, chunk_path)
        h5_chunks.append(chunk_path)
        chunk_counter[0] += 1
        
        # Delete old chunks (keep last 5)
        if len(h5_chunks) > 10:
            for old_chunk in h5_chunks[:5]:
                if os.path.exists(old_chunk):
                    os.remove(old_chunk)
                    # Mark for remote deletion
                    open(old_chunk + ".deleted", 'w').close()
            h5_chunks = h5_chunks[5:]

# ========================================

# Load parameters - using import_module like original
param = import_module(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module, experiment_name)(param)

c = param.c
c.logfilepath = utils.logfilename('') + '/'

# Start experiment
(source, stats_single) = experiment.start()

# Create SORN
sorn = Sorn(c, source)
stats = StatsCollection(sorn)
stats.methods = stats_single
sorn.stats = stats

# Wrap simulation for chunked saving
original_simulation = sorn.simulation

def chunked_simulation(steps):
    """Run simulation with chunked H5 saving"""
    global total_steps
    
    remaining = steps
    while remaining > 0:
        batch = min(CHUNK_SIZE, remaining)
        original_simulation(batch)
        total_steps[0] += batch
        remaining -= batch
        
        # Save chunk every 10k steps
        if total_steps[0] % CHUNK_SIZE == 0:
            # Close and rotate H5
            sorn.stats.dlog.close()
            save_h5_chunk(c)
            # Reopen
            sorn.stats.dlog.set_handler('*', utils.StoreToH5, utils.logfilename("result.h5"))

sorn.simulation = chunked_simulation

# Setup stats
stats.dlog.set_handler('*', utils.StoreToH5, utils.logfilename("result.h5"))
stats.dlog.append('c', utils.unbunchify(c))
stats.dlog.set_handler('*', utils.TextPrinter)

# Run experiment
experiment.reset(sorn)
sorn.stats.start()
sorn.stats.clear()

# Run quietly - suppress percentage spam
c.display = False
pickle_objects = experiment.run(sorn)

# Save final chunk
sorn.stats.dlog.close()
save_h5_chunk(c)

# Save other files
for key in pickle_objects:
    filename = os.path.join(c.logfilepath, "%s.pickle" % key)
    pickle.dump(pickle_objects[key], gzip.open(filename, "wb"), pickle.HIGHEST_PROTOCOL)

if sorn.c.stats.control_rates:
    experiment.control_rates(sorn)

stats.single_report()
stats.disable = True
sorn.quicksave(filename=os.path.join(c.logfilepath, 'net.pickle'))

print "[Sim %d] Complete - %d chunks saved" % (sim_number, chunk_counter[0])