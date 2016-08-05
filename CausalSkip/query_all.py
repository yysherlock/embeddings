import os
import sys
import heapq
import configparser
import pickle
import numpy as np
from sgd import load_saved_params

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def get_cosine_similarity(v1, v2):
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)))

def knn(word, wordvec, vectors, offset, k=5):
    results = []
    h = []
    for i,vec in enumerate(vectors):
        if wordlist[offset + i] == word: continue
        dist = get_cosine_similarity(wordvec,vec)
        if len(h) < k:
            heapq.heappush(h, (dist,i))
        else: heapq.heappushpop(h, (dist,i))

    h = sorted(h, reverse=True)

    for dist,i in h:
        results.append(wordlist[i+offset] + ':' + str(dist))

    return results

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)
section = "COPA"
datasets_dir = config.get(section, "datasets_dir")
tokens = loadObj(config.get(section, "tokens"))
wordlist = loadObj(config.get(section, "id2word_list"))
causeprior = loadObj(config.get(section, "cause_prior"))
N = len(causeprior)
cause_offset, effect_offset = 0, N

while True:
    dim = input('Please enter the dimension of word vectors you want to try, e.g. 200,\nEnter `q` to quit. Please enter your dimension: ')
    if dim == 'q' or dim == 'quit':
        sys.exit(0)
    params_path = datasets_dir + '/dim=' + dim
    if not os.path.exists(params_path):
        print('dimension not exists!')
    else:
        # load params
        _, wordVectors, _, _ = load_saved_params(params_path)
        causeVectors, effectVectors = wordVectors[:N,:], wordVectors[N:,:]

        break

k = 20
with open('query_results_dim='+dim+'_top' + str(k) +'.txt','w') as outf:
    for word in wordlist:
        word_type = word.split('_')[1]

        if word_type == 'c':
            homo_vectors, hete_vectors = causeVectors, effectVectors
            homo_offset, hete_offset = cause_offset, effect_offset
            wordvec = causeVectors[tokens[word] - cause_offset]

        if word_type == 'e':
            homo_vectors, hete_vectors = effectVectors, causeVectors
            homo_offset, hete_offset = effect_offset, cause_offset
            wordvec = effectVectors[tokens[word] - effect_offset]

        homo_topk = knn(word, wordvec, homo_vectors, homo_offset, k)
        hete_topk = knn(word, wordvec, hete_vectors, hete_offset, k)

        outf.write('word: '+word + '\n')
        outf.write('Topk homo words: ')
        outf.write(str(homo_topk))
        outf.write('\n')
        outf.write('Topk hete words: ')
        outf.write(str(hete_topk))
        outf.write('\n')
        outf.flush()
