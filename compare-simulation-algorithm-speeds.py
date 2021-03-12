# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext lab_black

# %run -i ./preamble.py

from simulate import *
from inverse_moments import *

# +
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
import timeit
from tqdm.notebook import trange, tqdm
# -

plt.rcParams["figure.figsize"] = (3.0, 3.0)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 12

lambda0 = 1.0
beta = 0.25
lambdaS = 3
lambdaE = 10.0
rho = 1.25
mat = 100

# ## Look at choosing the correct step size for the thinning alg

# +
# %%time 
rnd.seed(1)
R = 10**3
steps = np.linspace(1.25, 4, 10)
timesSteps = np.zeros(len(steps))
for i, s in enumerate(steps):
    start = time()
    for r in range(R):
        simulate_thinning(lambda0, beta, lambdaS, lambdaE, rho, mat, s)
    timesSteps[i] = time() - start
    
plt.plot(steps, timesSteps);
steps[np.argmin(timesSteps)]
# -

step = steps[np.argmin(timesSteps)]
step

# ## Create the tables for the paper

repeat = 3

# %%time
RExps = [2,3,4,5]
timesExact = []
for RExp in RExps:
    R = 10**RExp
    print(f"R = 10^{RExp}", end="")
    setup = "import simulate"
    rnd.seed(1)
    code = f"simulate.simulate_exact({lambda0}, {beta}, {lambdaS}, {lambdaE}, {rho}, {mat}, {R})"
    bestRun = np.min(timeit.repeat(code, setup, number=1, repeat=repeat))
    timesExact.append(bestRun)
    print(f", Best time of {repeat} = {bestRun} secs", flush=True)

# %%time
RExps = [2,3,4,5]
timesThin = []
for RExp in RExps:
    R = 10**RExp
    print(f"R = 10^{RExp}", end="")
    setup = "import simulate"
    rnd.seed(1)
    code = f"[simulate.simulate_thinning({lambda0}, {beta}, {lambdaS}, {lambdaE}, {rho}, {mat}, {step}) for r in range({R})]"
    bestRun = np.min(timeit.repeat(code, setup, number=1, repeat=repeat))
    timesThin.append(bestRun)
    print(f", Best time of {repeat} = {bestRun} secs", flush=True)
