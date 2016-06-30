import numpy as np
import configparser
import random
from data_utils import *
from model_utils import softmax, sigmoid, sigmoid_grad, gradcheck_naive

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    x = x.T
    x = x / np.sqrt(np.sum(np.square(x),axis=0)) # N x D
    x = x.T
    return x

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, target_type,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    W,D = outputVectors.shape

    UK = np.zeros((K+1, D))
    indices = [target]
    tablesize = W
    for i in xrange(K):
        k = dataset.sampleTokenIdx(target_type)
        while k == target:
            k = dataset.sampleTokenIdx(target_type)
        indices.append(k)
    for i,ix in enumerate(indices):
        UK[i] = outputVectors[ix]

    u_o = outputVectors[target] # (D,)
    cost = - np.log(sigmoid(np.dot(u_o, predicted))) - np.sum(np.log(sigmoid(-np.dot(UK[1:], predicted))))
    gradPred = (sigmoid(np.dot(u_o,predicted))-1) * u_o + np.dot(UK[1:].T,sigmoid(np.dot(UK[1:], predicted))) # dJ/dV_c, (D,)

    y = np.zeros(K+1); y[0] = 1.0 #
    grad = np.zeros(outputVectors.shape)
    gradK = np.outer(sigmoid(np.dot(UK, predicted)) - y, predicted)
    for i,ix in enumerate(indices):
        grad[ix] += gradK[i]

    return cost, gradPred, grad


def cskipgram(center_type, target_type, currentWord, C, contextWords, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    tokens = dataset.tokens

    iW,D = inputVectors.shape
    oW,D = outputVectors.shape

    inoffset = getattr(dataset, center_type+'_offset')
    outoffset = getattr(dataset, target_type+'_offset')

    cost = 0.0
    gradIn = np.zeros((iW,D))
    gradOut = np.zeros((oW,D))
    center = tokens[currentWord] - inoffset
    predicted = inputVectors[center]

    for i,contextWord in enumerate(contextWords):
        target = tokens[contextWord] - outoffset
        inc_cost, inc_gradPred, inc_gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset, target_type)
        cost += inc_cost
        gradIn[center] += inc_gradPred
        gradOut += inc_gradOut

    return cost, gradIn, gradOut

def word2vec_sgd_wrapper(word2vecModel, wordVectors, dataset, C, word2vecCostAndGradient = negSamplingCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = len(dataset.causeprior)
    causeVectors = wordVectors[:N,:]
    effectVectors = wordVectors[N:,:]

    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        center_type, centerword, context = dataset.getRandomContext(C1)

        if center_type == "cause":
            target_type = "effect"
            inputVectors = causeVectors
            outputVectors = effectVectors
        else:
            target_type = "cause"
            inputVectors = effectVectors
            outputVectors = causeVectors

        if word2vecModel == cskipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(center_type, target_type, centerword, C1, context, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom

        M = inputVectors.shape[0]
        grad[:M, :] += gin / batchsize / denom
        grad[M:, :] += gout / batchsize / denom

    return cost, grad


def test_model():
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    configPath = 'bi-config.ini'
    config.read(configPath)

    random.seed(31415)
    np.random.seed(9265)

    dataset = Causal(configPath, "CausalNet")
    dimVectors = 10

    causenWords = len(dataset.causeprior)
    effectnWords = len(dataset.effectprior)

    wordVectors = np.concatenate(((np.random.rand(causenWords, dimVectors) - .5) / \
        dimVectors, np.zeros((effectnWords, dimVectors))), axis=0)

    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cskipgram, vec, dataset, 5, negSamplingCostAndGradient), wordVectors)

if __name__=="__main__":
    test_model()
