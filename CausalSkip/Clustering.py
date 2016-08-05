import pickle
import configparser
import sklearn.cluster
import numpy as np
from sgd import load_saved_params

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)
section = "COPA"
datasets_dir = config.get(section, "datasets_dir")
wordlist = loadObj(config.get(section, "id2word_list"))
dim = '300'
params_path = datasets_dir + '/GloveInit/dim=' + dim
#params_path = '../oldRndWrongCausalSkip/GloveInit/dim=300'
#params_path = '../oldRndWrongCausalSkip/dim=300'

_, X, _, _ = load_saved_params(params_path) # X: wordVectors
k_means = sklearn.cluster.KMeans(n_clusters=500, max_iter=100000)
k_means.fit(X)

print(len(k_means.labels_))

d = {}
for i,label in enumerate(k_means.labels_):
    d.setdefault(label,[])
    d[label].append(wordlist[i])

with open('kmeans_gloveinit_cluster500.txt','w') as outf:
    for k,v in d.items():
        outf.write(str(k)+': '+str(v)+'\n')
        outf.flush()
