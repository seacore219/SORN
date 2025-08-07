from __future__ import division
from __future__ import print_function
import warnings
import tables
from importlib import import_module
import imp
import ipdb
import os
import sys
import subprocess
import multiprocessing
import time
import argparse
import shutil
import datetime
import utils
from stats import StatsCollection
from stats import AbstractStat
from synapses import create_matrix
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from delpapa.plot import plot_results as plot_results_single
import random
import numpy as np
from os.path import *
from os.path import isfile
from time import strftime
from abc import ABCMeta, abstractmethod
from optparse import OptionParser
from scipy import linalg
import scipy.sparse as sp
import scipy.stats as st
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import scipy.sparse.linalg as spla
import itertools
import synapses
from scipy.optimize import curve_fit
import sources
from scipy import stats
from scipy import signal
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.stats import pearsonr
import platform