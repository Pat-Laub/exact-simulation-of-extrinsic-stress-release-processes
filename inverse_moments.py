from simulate import *

from scipy.integrate import odeint
from scipy.interpolate import interp1d

from tqdm import trange, tqdm

import numpy as np

def lambdas_thinning(lambda0, mu, lambdaS, lambdaE, rho, mat, step, tGrid, R):
    lambdas = np.zeros((R, len(tGrid)))

    for r in trange(R):
        ts, logLambdas = simulate_thinning(lambda0, mu, lambdaS, lambdaE, rho, mat, step)[:2]
        logInt = interp1d(ts, logLambdas, assume_sorted=True)
        lambdas[r,:] = np.exp(logInt(tGrid))

    return lambdas

def lambdas_exact(lambda0, mu, lambdaS, lambdaE, rho, mat, tGrid, R):
    lambdas = np.zeros((R, len(tGrid)))
    tsAll, logLambdasAll = simulate_exact(lambda0, mu, lambdaS, lambdaE, rho, mat, R)

    for r in trange(R):
        ts, logLambdas = tsAll[r,:], logLambdasAll[r,:]
        ts = ts[~np.isnan(ts)]; logLambdas = logLambdas[~np.isnan(logLambdas)]
        logInt = interp1d(ts, logLambdas, assume_sorted=True)
        lambdas[r,:] = np.exp(logInt(tGrid))

    return lambdas

def Z_moments_thinning(lambda0, mu, lambdaS, lambdaE, rho, mat, step, tGrid, R, ns):
    lambdas = lambdas_thinning(lambda0, mu, lambdaS, lambdaE, rho, mat, step, tGrid, R)
    if type(ns) != list:
        ns = [ns]

    momentEsts = []
    momentCIs = []
    for n in ns:
        nthMomentRVs = np.power(lambdas, -float(n))
        momentEsts.append(nthMomentRVs.mean(axis=0))
        momentCIs.append(1.96 * nthMomentRVs.std(axis=0) / np.sqrt(R))

    if len(ns) == 1:
        return momentEsts[0], momentCIs[0]

    return momentEsts, momentCIs

def Z_moments_exact(lambda0, mu, lambdaS, lambdaE, rho, mat, tGrid, R, ns):
    lambdas = lambdas_exact(lambda0, mu, lambdaS, lambdaE, rho, mat, tGrid, R)

    if type(ns) != list:
        ns = [ns]

    momentEsts = []
    momentCIs = []
    for n in ns:
        nthMomentRVs = np.power(lambdas, -float(n))
        momentEsts.append(nthMomentRVs.mean(axis=0))
        momentCIs.append(1.96 * nthMomentRVs.std(axis=0) / np.sqrt(R))

    if len(ns) == 1:
        return momentEsts[0], momentCIs[0]

    return momentEsts, momentCIs

# Theoretical E[1/lambda^n]
def Z_moment(lambda0, lambdaS, lambdaE, rho, mu, t, n):
    # MGF_X at (1 or 2) minus 1
    m1S = lambdaS/(lambdaS-1)-1
    m1E = lambdaE/(lambdaE-1)-1
    m2S = lambdaS/(lambdaS-2)-1
    m2E = lambdaE/(lambdaE-2)-1

    psi1 = mu - rho*m1E

    ePsi1Inv = np.exp(-psi1*t)

    psi = mu - rho*m1E
    a = m1S/psi
    b = 1/lambda0 - m1S/psi

    if n == 1 or n==1.0:
        return a + b*np.exp(-psi*t)

    if n == 2 or n == 2.0:
        if not (lambda0 == 1 or lambda0==1.0):
            raise RuntimeException("Can't get analytic form of 2nd inverse moment with general lambda0")

        psi2 = 2*mu - rho*m2E
        ePsi2Inv = np.exp(-psi2*t)
        psiDiff = psi2-psi1

        term1 = (ePsi1Inv - ePsi2Inv) / psiDiff
        term2 = (1 / psi2) - (ePsi1Inv / psiDiff)
        term3 = ePsi2Inv * ((1 / psi2) - (1 / psiDiff))

        return ePsi2Inv + m2S * (term1 + (m1S/psi1) * (term2 - term3))

def Z_moment_2_ode(lambda0, lambdaS, lambdaE, rho, mu, t):
     # MGF_X at (1 or 2) minus 1
    m1S = lambdaS/(lambdaS-1)-1
    m1E = lambdaE/(lambdaE-1)-1
    m2S = lambdaS/(lambdaS-2)-1
    m2E = lambdaE/(lambdaE-2)-1

    psi = mu - rho*m1E
    a = m1S/psi
    b = 1/lambda0 - m1S/psi

    u = lambda t: a + b*np.exp(-psi*t)
    dv = lambda v, t: m2S*u(t) + (rho*m2E - 2*mu)*v

    return odeint(dv, lambda0**(-2), t).reshape(-1)