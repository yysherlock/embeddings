# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

import glob
import random
import numpy as np
import os.path as op
from six.moves import xrange
import pickle

def load_saved_params(params_dir):
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob(params_dir+"/saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
            cnWords = pickle.load(f)
        return st, params, state, cnWords
    else:
        return st, None, None

def save_params(iter, params, params_dir, cnWords):
    with open(params_dir+"/saved_params_%d.npy" % iter, "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)
        pickle.dump(cnWords,f)


def sgd(f, x0, params_dir, step, iterations, cnWords, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    # Implement the stochastic gradient descent method in this
    # function.

    # Inputs:
    # - f: the function to optimize, it should take a single
    #     argument and yield two outputs, a cost and the gradient
    #     with respect to the arguments
    # - x0: the initial point to start SGD from
    # - step: the step size for SGD
    # - iterations: total iterations to run SGD for
    # - postprocessing: postprocessing function for the parameters
    #     if necessary. In the case of word2vec we will need to
    #     normalize the word vectors to have unit length.
    # - PRINT_EVERY: specifies every how many iterations to output

    # Output:
    # - x: the parameter value after SGD finishes

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.

        cost = None
        cost, grad = f(x)
        x -= step * grad
        x = postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d: %f" % (iter, expcost))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x, params_dir, cnWords)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
