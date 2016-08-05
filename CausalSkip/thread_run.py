import sys
import os
import random
import numpy as np
import configparser
from data_utils import *
from SkipModel import *
from sgd import *

if len(sys.argv) < 3:
    print("Please enter 2 parameters, 1) section name in configfile, 2) dimension of wordvectors")
    print("e.g. `python3 run.py COPA 100`")
    sys.exit(1)

dimVectors = int(sys.argv[2])

random.seed(314)
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)

for section in config.sections():
    if not section==sys.argv[1]: continue

    dataset = Causal(configPath, section)
    causenWords = len(dataset.causeprior)
    effectnWords = len(dataset.effectprior)

    datasets_dir = config.get(section, "datasets_dir")

    # context size
    C = 5
    params_dir = datasets_dir + "/dim=" + str(dimVectors)
    if not os.path.exists(params_dir): os.system("mkdir " + params_dir)
    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)
    wordVectors = np.concatenate(((np.random.rand(causenWords, dimVectors) - .5) / \
        dimVectors, np.zeros((effectnWords, dimVectors))), axis=0)
    wordVectors0 = sgd(
        lambda vec: word2vec_sgd_wrapper(cskipgram, vec, dataset, C,
        	negSamplingCostAndGradient),
        wordVectors, params_dir, 0.3, 100000, None, True, PRINT_EVERY=100)
    print("sanity check: cost at convergence should be around or below 10")
