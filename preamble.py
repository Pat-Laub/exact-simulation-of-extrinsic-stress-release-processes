# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:16:18 2020

@author: Pat and Pierre-O
"""

# This tells Python to just use one thread for internal
# math operations. When it tries to be smart and do multithreading
# it usually just stuffs everything up and slows us down.
import os
for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[env] = "1"

import matplotlib.pyplot as plt # For plotting
import numpy as np # For fast math

# New way to get pseudorandom numbers
from numpy.random import SeedSequence, default_rng, Generator, PCG64

import dill

import pandas as pd
import seaborn as sns

import subprocess
import shutil

def crop_fig(filename):
    cmd = f"pdf-crop-margins {filename} -o cropped.pdf"
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    shutil.move("cropped.pdf", filename)

def save_cropped(filename):
    plt.savefig(filename.replace("pdf", "pgf"), pad_inches=0)
    plt.savefig(filename, pad_inches=0)
    crop_fig(filename)
    plt.show()

plt.rcParams['figure.figsize'] = (5.0, 2.0)
plt.rcParams['figure.dpi'] = 350
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.family'] = "serif"
# plt.rcParams["pgf.preamble"] = [r"\usepackage{amssymb}"] # To get blackboard fonts

# This is the default color scheme (with orange skipped).
colors = [
    "tab:blue", # "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

# Set the default color scheme so that this orange is skipped.
import cycler
plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)

# Re-add orange to colors list.
colors.insert(1, "tab:orange")

from ttictoc import tic, toc