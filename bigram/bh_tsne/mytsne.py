import pickle
import numpy as np
import bhtsne

def get_data(biembedsfn): # list of list
    with open(biembedsfn,'rb') as f:
        X = pickle.load(f)
    N,d = len(X), X[0].shape[0]
    data = []
    for i in range(N):
        sample = [ X[i][j,0] for j in range(d) ]
        data.append(sample)
    return data

def tsne(biembedsfn):
    data = get_data(biembedsfn)
    bhtsne.bh_tsne(data,no_dims=2)

def tranferdata(biembedsfn,outfn):
    with open(biembedsfn,'rb') as f, open(outfn,'w') as outf:
        X = pickle.load(f)
        N,d = len(X), X[0].shape[0]
        for i in range(N):
            outf.write('\t'.join( [ str(X[i][j,0]) for j in range(d)]) + '\n')

dir_path = '../copa/'
biembedsfn = dir_path + 'biembeds.pickle'
outfn = dir_path + 'biembeds.txt'
tranferdata(biembedsfn, outfn)

