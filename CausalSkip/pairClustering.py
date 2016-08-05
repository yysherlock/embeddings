import heapq
import pickle
import configparser
import sklearn.cluster
import numpy as np
from sgd import load_saved_params

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def get_cosine_similarity(v1, v2):
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)))

def knn(word, wordvec, vectors, offset, k=10000):
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

dim = '300'
params_path = datasets_dir + '/GloveInit/dim=' + dim
#params_path = '../oldRndWrongCausalSkip/GloveInit/dim=300'
#params_path = '../oldRndWrongCausalSkip/dim=300'

_, wordVectors, _, _ = load_saved_params(params_path)
causeVectors, effectVectors = wordVectors[:N,:], wordVectors[N:,:]
# Select top 10000 causal pairs, X: pairVectors
causeWords, effectWords = wordlist[:N], wordlist[N:]
h = []
K = 10000
for cword in causeWords:
    for eword in effectWords:
        cvec = wordVectors[tokens[cword]]
        evec = wordVectors[tokens[eword]]
        dist = get_cosine_similarity(cvec,evec)
        if len(h) < K:
            heapq.heappush(h, (dist,cword,eword))
        else: heapq.heappushpop(h, (dist,cword,eword))

h = sorted(h, reverse=True)
pairlist = []
X = np.empty((K, 2*wordVectors.shape[1]))

for i,t in enumerate(h):
    dist,cword,eword = t
    pairlist.append(cword + ":" + eword)
    pairVec = np.concatenate((wordVectors[tokens[cword]],wordVectors[tokens[eword]]), axis=0)
    X[i] = pairVec

# cluster those pairs
k_means = sklearn.cluster.KMeans(n_clusters=500, max_iter=100000)
k_means.fit(X)

print(len(k_means.labels_))

d = {}
for i,label in enumerate(k_means.labels_):
    d.setdefault(label,[])
    d[label].append(pairlist[i])

with open('kmeans_paircluster500.txt','w') as outf:
    for k,v in d.items():
        outf.write(str(k)+': '+str(v)+'\n')
        outf.flush()
