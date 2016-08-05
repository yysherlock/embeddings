import sys
import os
import random
import pickle
import numpy as np
from six.moves import xrange
import configparser
from data_utils import *
from SkipModel import *
from sgd import *

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def init_wordvectors(dataset,useGlove = False):
    wordVectors = np.concatenate(((np.random.rand(causenWords, dimVectors) - .5) / \
        dimVectors, np.zeros((effectnWords, dimVectors))), axis=0)
    N,D = wordVectors.shape

    if useGlove and os.path.exists(glove_selected_wordvectors_file) and os.path.exists(glove_selected_words_file):
        print('useGlove to initialize word vectors')
        with open(glove_selected_wordvectors_file,'rb') as f1, open(glove_selected_words_file,'rb') as f2:
            glove_selected_wordvectors = pickle.load(f1)
            glove_selected_words = pickle.load(f2)
            if not D == glove_selected_wordvectors.shape[1]:
                print('not match')
                sys.exit(1)
        average_vector = np.mean(glove_selected_wordvectors,axis=0)

        for i,word in enumerate(dataset.tokens):
            word = word.split('_')[0]
            idx = glove_selected_words.index(word)
            if idx == -1:
                wordVectors[i] = average_vectors
            else:
                wordVectors[i] = glove_selected_wordvectors[idx]

    return wordVectors


if len(sys.argv) < 3:
    print("Please enter 2 parameters, 1) section name in configfile, 2) dimension of wordvectors")
    print("e.g. `python3 run.py COPA 100`")
    sys.exit(1)

dimVectors = int(sys.argv[2])

random.seed(314)
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)

glove_selected_wordvectors_file = "glove_selected_" + sys.argv[1].lower() + ".npy"
glove_selected_words_file = "glove_selected_" + sys.argv[1].lower() + "_words.npy"

for section in config.sections():
    if not section==sys.argv[1]: continue

    dataset = Causal(configPath, section)
    causenWords = len(dataset.causeprior)
    effectnWords = len(dataset.effectprior)

    datasets_dir = config.get(section, "datasets_dir")

    # context size
    C = 5
    params_dir = datasets_dir + "/GloveInit/dim=" + str(dimVectors)
    if not os.path.exists(params_dir): os.system("mkdir " + params_dir)
    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    wordVectors = init_wordvectors(dataset, True)

    wordVectors0 = sgd(
        lambda vec: word2vec_sgd_wrapper(cskipgram, vec, dataset, C,
        	negSamplingCostAndGradient),
        wordVectors, params_dir, 0.3, 100000, None, True, PRINT_EVERY=100)
    print("sanity check: cost at convergence should be around or below 10")
