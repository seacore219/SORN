from __future__ import division  # has to be imported in every file
#~ import matplotlib  # if from ssh
#~ matplotlib.use('Agg')

import warnings
import tables  # Import tables to catch its specific warning
from importlib import import_module
import imp  # add this for Python 2.7 file importing
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
import os

# Get simulation number from environment variable
sim_number = int(os.environ.get('SORN_SIM_NUMBER', '1'))
print "[INFO] Running simulation number:", sim_number

# Set unique random seed for this simulation
# This ensures each parallel simulation gets different random patterns
np.random.seed(sim_number * 12345)  # Multiply by large prime for good separation
print "[INFO] Set random seed to:", sim_number * 12345

# Replace import_module with imp.load_source where needed
# For example, if you need to import a parameter file:
def load_param_file(filename):
    return imp.load_source('parameters', filename)

# Start debugging mode when an error is raised
def debugger(type,flag):
    print 'In debugger!'
    pass
    # import ipdb
    # ipdb.set_trace()
np.seterrcall(debugger)
np.seterr(all='call')

##### To control the random seed
#~ np.random.seed(1)

# Parameters are read from the second command line argument
# param = import_module(utils.param_file())
param = load_param_file(utils.param_file())
experiment_module = import_module(param.c.experiment.module)
experiment_name = param.c.experiment.name
experiment = getattr(experiment_module,experiment_name)(param)
print("[DEBUG] experiment instantiated:", experiment.__class__)

c = param.c
c.logfilepath = utils.logfilename('')+'/'

(source,stats_single) = experiment.start()
print("[DEBUG] experiment.start() returned successfully")
print("[DEBUG] Input source class:", type(source))
print("[DEBUG] Source module:", source.__class__.__module__)
print("[DEBUG] Does it have generate_connection_e?", hasattr(source, "generate_connection_e"))
print("[DEBUG] Is it RandomSource?", source.__class__.__name__ == "RandomSource")
print("[CHECK] generate_connection_e is defined in:", source.__class__.__name__)
print("[CHECK] generate_connection_e is actually:", source.generate_connection_e)
print("[CHECK] generate_connection_e comes from:", source.generate_connection_e.__func__.__code__.co_filename)
print("[CHECK] at line:", source.generate_connection_e.__func__.__code__.co_firstlineno)


sorn = Sorn(c,source)
print("hi")
# Create a StatsCollection and fill it with methods for all statistics
# that should be tracked (and later plotted)
stats = StatsCollection(sorn)
stats.methods = stats_single
sorn.stats = stats

# Datalog is used to store all results and parameters
stats.dlog.set_handler('*',utils.StoreToH5,  utils.logfilename("result.h5"))
stats.dlog.append('c', utils.unbunchify(c))
stats.dlog.set_handler('*',utils.TextPrinter)

print("[DEBUG] experiment class is:", experiment.__class__)
print("[DEBUG] does experiment have reset?", hasattr(experiment, "reset"))
print("[DEBUG] reset comes from:", experiment.reset.__func__.__code__.co_filename)
print("[DEBUG] Calling experiment.reset(sorn)")
experiment.reset(sorn)
print("[DEBUG] experiment.reset(sorn) completed")

# Start stats
sorn.stats.start()
sorn.stats.clear()

##### To control the random seed after initialization!
# np.random.seed(3)

# Run experiment once
print("[DEBUG] About to call experiment.run(sorn)")
pickle_objects = experiment.run(sorn)
print("[DEBUG] Returned from experiment.run(sorn)")

# Save sources etc
for key in pickle_objects:
    filename = os.path.join(c.logfilepath,"%s.pickle"%key)
    topickle = pickle_objects[key]
    try:
        # Try normal pickle first
        pickle.dump(topickle, gzip.open(filename,"wb"), protocol=2)
    except OverflowError:
        # If too large, save in chunks
        save_in_chunks(topickle, filename)

# Control: Firing-rate model: Substitute spikes by drawing random spikes
# according to firing rate for each inputindex
if sorn.c.stats.control_rates:
    experiment.control_rates(sorn)

# Report stats and close
stats.single_report()
stats.disable = True
stats.dlog.close()
try:
    net_filename = os.path.join(c.logfilepath,'net.pickle')
    sorn.quicksave(filename=net_filename)
except OverflowError:
    save_in_chunks(sorn, net_filename)

print("[INFO] Experiment completed... now plotting")

# Plot data collected by stats
#~ dest_directory = os.path.join(dest_directory,'common')
experiment.plot_single(dest_directory,
                       os.path.join('common','result.h5'))

# Display figures
# plt.show()
