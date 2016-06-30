import random
import numpy as np
import configparser
from data_utils import *
from SkipModel import *
from sgd import *

"""
Causal embeddings training
"""

random.seed(314)
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)

for section in config.sections():
    if not section=="CausalNet": continue
    dataset = Causal(configPath, section)
    causenWords = len(dataset.cause_prior)
    effectnWords = len(dataset.effect_prior)
    for dimVectors in [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]:
        # context size
        C = 5

        # Reset the random seed to make sure that everyone gets the same results
        random.seed(31415)
        np.random.seed(9265)
        wordVectors = np.concatenate(((np.random.rand(causenWords, dimVectors) - .5) / \
        	dimVectors, np.zeros((effectnWords, dimVectors))), axis=0)
        wordVectors0 = sgd(
            lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
            	negSamplingCostAndGradient),
            wordVectors, 0.3, 1000, None, True, PRINT_EVERY=10)
        print "sanity check: cost at convergence should be around or below 10"
