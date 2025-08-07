from __future__ import division
from __future__ import print_function
import argparse
import csv
import datetime
import gc
import glob
import gzip
import io
import itertools
import json
import math
import multiprocessing
import multiprocessing as mp
import os
import os.path as op
# import cPickle as pickle
import platform
import random
import random as randomstr
import re
import shutil
import subprocess
import sys
import time
import unittest
import warnings
import imp
from abc import ABCMeta, abstractmethod
from copy import deepcopy as cdc
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from functools import partial
from importlib import import_module
from os.path import *
from os.path import isfile
from pathlib import Path
from tempfile import TemporaryFile
from time import strftime
from typing import Dict, Tuple, Optional
from typing import List, Any
from urllib.parse import urljoin
import numpy as np
import scipy
import scipy.io as sio
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.stats as st
import scipy.stats as statistics
import scipy.stats as stats
from scipy import linalg
from scipy import signal
from scipy import stats
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.io import savemat
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import welch
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.special import erf
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
import h5py
import tables
# import ipdb
# import mrestimator as mre
# import powerlaw as pl
import psutil
import requests
# from autotable import AutoTable
from setuptools import setup