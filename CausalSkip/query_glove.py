import os
import sys
import heapq
import configparser
import pickle
import numpy as np

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def get_cosine_similarity(v1, v2):
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)))

def knn(word, wordvec, vectors, k=5):
    results = []
    h = []
    for i,vec in enumerate(vectors):
        if glove_copa_words[i] == word: continue
        dist = get_cosine_similarity(wordvec,vec)
        if len(h) < k:
            heapq.heappush(h, (dist,i))
        else: heapq.heappushpop(h, (dist,i))

    h = sorted(h, reverse=True)

    for dist,i in h:
        results.append(glove_copa_words[i] + ':' + str(dist))

    return results

glove_file = 'glove.840B.300d.txt'
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)
section = "COPA"
wordlist = loadObj(config.get(section, "id2word_list"))
glove_copa_words = []


if not os.path.exists('query_glove_copa_results.txt') or \
not os.path.exists('glove_copa_words.npy'):

    # generate glove vectors
    N, D = 0,300
    words = [word.split('_')[0] for word in wordlist]
    with open('glove.840B.300d.txt') as f, open('glove_copa_words.npy','wb') as outf:
        for line in f:
            word = line.strip().split()[0]
            if word in words:
                glove_copa_words.append(word)
                N += 1
        pickle.dump(glove_copa_words, outf)

    glove_copa_vectors = np.zeros((N,D))

    with open('glove.840B.300d.txt') as f, open('query_glove_copa_results.npy','wb') as outf:
        i = 0
        for line in f:
            content = line.strip().split()
            word, vec = content[0], content[-300:]
            if word not in glove_copa_words: continue
            glove_copa_vectors[i] = np.array(vec)
            i += 1

        pickle.dump(glove_copa_vectors, outf)

with open('glove_copa_words.npy','rb') as f: glove_copa_words = pickle.load(f)

k = 20
with open('query_glove_copa_results.npy','rb') as f, open('query_glove_copa_results.txt','w') as outf:
    wordVectors = pickle.load(f)
    for i,word in enumerate(glove_copa_words):
        topk = knn(word, wordVectors[i], wordVectors, k)
        outf.write('word: '+word + '\n')
        outf.write('Topk words: ')
        outf.write(str(topk))
        outf.write('\n')
