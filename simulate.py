import numpy as np
import numpy.random as rnd
import os

def intensities(lambda0, beta, selfTs, extTs, Xs, Ys, ts):
    """ Calculate the value of the intensity process at given time values.

    Assumption: The process is simulated up till some time >= max(ts)
    """
    events = list(zip(selfTs, Xs)) + list(zip(extTs, Ys)) + list(zip(ts, [None]*len(ts)))
    events.sort()

    t = 0
    logLambda = np.log(lambda0)
    intensities = np.zeros(len(ts))
    i = 0
    for t2, jump in events:
        logLambda += beta*(t2-t)
        t = t2

        # Either adjust for a jump, or record the current intensity value
        if jump:
             logLambda -= jump
        else:
            intensities[i] = logLambda
            i += 1

    return np.exp(intensities)

def find_max_intensity(logLambda, beta, extTs, Ys, ind, t, step):
    """ Calculate the maximum log-intensity over the period [t, t+step]
    when we assume only external arrivals occurred (i.e. no self-arrivals)
    """
    numArrivals = 0
    maxLL = -np.inf
    prevTime = t
    for i in range(ind, len(extTs)):
        if extTs[i] > t+step:
            break

        numArrivals += 1

        # Calculate the new log intensity before & after this external arrival
        logLambdaMinus = logLambda + beta*(extTs[i] - prevTime)
        prevTime = extTs[i]
        logLambda = logLambdaMinus - Ys[i]

        # Update the maximum log intensity over this period
        if logLambdaMinus > maxLL:
            maxLL = logLambdaMinus

    # Calculate the log intensity at final time t+step
    logLambda += beta*(t+step - prevTime)
    if logLambda > maxLL:
        maxLL = logLambda

    return maxLL

def exponential_generator(prealloc):
    while True:
        #print("Generating a bunch of exponentials\n", flush=True)
        exps = -np.log(rnd.rand(prealloc))
        for exp in exps:
            yield exp

def simulate_thinning(lambda0, beta, lambdaS, lambdaE, rho, maxT, step, prealloc=10**5, giveUp=10**7):
    # Arrival times and jumps
    selfTs = []; Xs = []

    # External jumps (independent of the rest, so simulate them first)
    rvCost = 1
    numExtJumps = rnd.poisson(maxT*rho)

    rvCost += numExtJumps*2
    extTs = np.sort(rnd.rand(numExtJumps))*maxT
    Ys = rnd.exponential(1/lambdaE, numExtJumps)

    expGen = exponential_generator(prealloc)

    t = 0.0
    ind = 0
    logLambda = np.log(lambda0)

    ts = [0]
    logLambdas = [logLambda]

    for _ in range(giveUp):
        # Calculate the maximum intensity over [t, t+step] time period.
        maxLL = find_max_intensity(logLambda, beta, extTs, Ys, ind, t, step)

        # Simulate an arrival time with rate = maximum intensity
        tau = next(expGen) / np.exp(maxLL)
        rvCost += 1

        # Update log-intensity for the external arrivals which
        # come before time t + min(tau, step)
        prevTime = t
        minTauStep = tau if tau < step else step
        tNext = t + minTauStep

        while ind < numExtJumps:
            if extTs[ind] > tNext:
                break

            logLambda += beta * (extTs[ind] - prevTime)
            ts.append(extTs[ind]); logLambdas.append(logLambda)
            logLambda += - Ys[ind]
            ts.append(extTs[ind]); logLambdas.append(logLambda)
            prevTime = extTs[ind]
            ind += 1

        # Update log-intensity for the last period when no arrivals
        # of any type occur
        t = tNext if tNext < maxT else maxT
        logLambda += beta * (t - prevTime)

        if t == maxT:
            ts.append(maxT); logLambdas.append(logLambda)
            return (np.array(ts), np.array(logLambdas), np.array(selfTs),
                    extTs, np.array(Xs), Ys, rvCost)

        # If the inter-arrival time proposed was larger than our step size
        # we say no self-arrivals occurred and skip ahead
        if tau > step:
            continue

        # If a small inter-arrival time is proposed, we flip a coin and
        # say it is a self-arrival with probability lambda(t) / max lambda
        if logLambda > maxLL:
            raise Exception("Maximum isn't a max!")

        rvCost += 1
        if rnd.rand() < np.exp(logLambda - maxLL):
            # Record self-arrival time
            selfTs.append(t)

            # Record arrival jump
            rvCost += 1
            X = next(expGen) / lambdaS
            Xs.append(X)

            ts.append(t); logLambdas.append(logLambda)
            logLambda -= X
            ts.append(t); logLambdas.append(logLambda)

    raise Exception("Ran out of simulation budget")

def simulate_thinning_pessimist(lambda0, beta, lambdaS, lambdaE, rho, maxT, step, prealloc=10**5, giveUp=10**7):
    # Arrival times and jumps
    selfTs = []; Xs = []

    # External jumps (independent of the rest, so simulate them first)
    rvCost = 1
    numExtJumps = rnd.poisson(maxT*rho)

    rvCost += numExtJumps*2
    extTs = np.sort(rnd.rand(numExtJumps))*maxT
    Ys = rnd.exponential(1/lambdaE, numExtJumps)

    expGen = exponential_generator(prealloc)

    t = 0.0
    ind = 0
    logLambda = np.log(lambda0)

    ts = [0]
    logLambdas = [logLambda]

    for _ in range(giveUp):
        # Calculate the maximum intensity over [t, t+step] time period.
        maxLL = logLambda + beta * step

        # Simulate an arrival time with rate = maximum intensity
        tau = next(expGen) / np.exp(maxLL)
        rvCost += 1

        # Update log-intensity for the external arrivals which
        # come before time t + min(tau, step)
        prevTime = t
        minTauStep = tau if tau < step else step
        tNext = t + minTauStep

        while ind < numExtJumps:
            if extTs[ind] > tNext:
                break

            logLambda += beta * (extTs[ind] - prevTime)
            ts.append(extTs[ind]); logLambdas.append(logLambda)
            logLambda += - Ys[ind]
            ts.append(extTs[ind]); logLambdas.append(logLambda)
            prevTime = extTs[ind]
            ind += 1

        # Update log-intensity for the last period when no arrivals
        # of any type occur
        t = tNext if tNext < maxT else maxT
        logLambda += beta * (t - prevTime)

        if t == maxT:
            ts.append(maxT); logLambdas.append(logLambda)
            return ts, logLambdas, np.array(selfTs), extTs, np.array(Xs), Ys, rvCost

        # If the inter-arrival time proposed was larger than our step size
        # we say no self-arrivals occurred and skip ahead
        if tau > step:
            continue

        # If a small inter-arrival time is proposed, we flip a coin and
        # say it is a self-arrival with probability lambda(t) / max lambda
        if logLambda > maxLL:
            raise Exception("Maximum isn't a max!")

        rvCost += 1
        if rnd.rand() < np.exp(logLambda - maxLL):
            # Record self-arrival time
            selfTs.append(t)

            # Record arrival jump
            rvCost += 1
            X = next(expGen) / lambdaS
            Xs.append(X)

            ts.append(t); logLambdas.append(logLambda)
            logLambda -= X
            ts.append(t); logLambdas.append(logLambda)

    raise Exception("Ran out of simulation budget")

def simulate_exact_serially(lambda0, beta, lambdaS, lambdaE, rho, maxT, prealloc=10**5, giveUp=10**7):
    # Arrival times and jumps
    selfTs = []; Xs = []
    extTs = []; Ys = []

    t = 0.0
    ts = [0.0]
    logLambda = np.log(lambda0)
    logLambdas = [logLambda]

    expGen = exponential_generator(prealloc)

    # Simulation of one path
    for _ in range(giveUp):
        E = next(expGen) / rho
        S = 1/beta * np.log(1 + beta/np.exp(logLambda) * next(expGen))

        # Jump times
        isSelfJump = S < E
        iit = min(S, E)

        if t + iit > maxT:
            logLambda += beta * (maxT-t)
            ts.append(maxT); logLambdas.append(logLambda)
            return ts, logLambdas, np.array(selfTs), np.array(extTs), np.array(Xs), np.array(Ys)
            #return ts, logLambdas, selfTs, extTs, Xs, Ys

        t += iit

        # Update the intensity function
        logLambda += beta * iit
        ts.append(t); logLambdas.append(logLambda)

        if isSelfJump:
            selfTs.append(t)
            X = next(expGen) / lambdaS
            Xs.append(X)
            logLambda -= X
        else:
            extTs.append(t)
            Y = next(expGen) / lambdaE
            Ys.append(Y)
            logLambda -= Y

        ts.append(t); logLambdas.append(logLambda)

    raise Exception("Needed to specify more jumps")

def simulate_exact(lambda0, beta, lambdaS, lambdaE, rho, maxT, R, batch=10**3, giveUp=10**7):
    ind = 0
    t = np.zeros(R)
    ts = np.NaN * np.ones((batch, R))
    ts[ind,:] = 0.0

    logLambda = np.log(lambda0) * np.ones(R)
    logLambdas = np.NaN * np.ones((batch, R))
    logLambdas[ind,:] = logLambda
    ind += 1

    going = t < maxT
    numAlive = R

    for _ in range(giveUp):

        if not (ind < ts.shape[0] - 2):
            print("Extending batch", flush=True)
            ts = np.concatenate([ts, np.NaN * np.ones((batch, R))])
            logLambdas = np.concatenate([logLambdas, np.NaN * np.ones((batch, R))])

        E = -np.log(rnd.rand(numAlive)) / rho
        S = 1/beta * np.log(1 - beta/np.exp(logLambda[going]) * np.log(rnd.rand(numAlive)))

        # Jump times
        iit = np.zeros(R)
        iit[going] = np.minimum(S, E)

        # Is self jump
        sLessThanE = np.full(R, False)
        sLessThanE[going] = S < E

        finishing = going & (t + iit > maxT)

        ts[ind,finishing] = maxT
        logLambdas[ind,finishing] = logLambda[finishing] + beta * (maxT-t[finishing])

        t += iit
        going = t < maxT
        numAlive = np.sum(going)

        if numAlive == 0:
            return ts.T, logLambdas.T

        # Update the intensity function
        logLambda += beta * iit
        ts[ind,going] = t[going]
        logLambdas[ind,going] = logLambda[going]
        ind += 1

        isSelfJump = going & sLessThanE
        isExtJump = going & ~sLessThanE

        logLambda[isSelfJump] -= -np.log(rnd.rand(np.sum(isSelfJump)))/lambdaS
        logLambda[isExtJump] -= -np.log(rnd.rand(np.sum(isExtJump)))/lambdaE

        ts[ind,going] = t[going]
        logLambdas[ind,going] = logLambda[going]
        ind += 1

    raise Exception("Needed to specify more jumps")
