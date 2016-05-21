import numpy as np
import pickle
from six.moves import xrange
import matplotlib.pyplot as plt

def idxmapping(idx,rowsize,colsize):
    assert idx < rowsize * colsize
    row = idx / colsize
    col = idx % colsize
    return row, col

with open('../word.list','rb') as f:
    wordlist = pickle.load(f)

with open('../biembeds.pickle','rb') as f:
    biembeds = pickle.load(f) # a list of embeddings, each embedding is 50 x 1

#print(biembeds[0].T)
la = np.linalg
N = len(biembeds); d = biembeds[0].shape[0]
X = np.empty([N,d])

# generate selected_list
wordmap = {}
for i in xrange(len(wordlist)):
    wordmap[wordlist[i]] = i
selected_words = ['happy_c','enjoy_e','fire_c','fire_e','death_e','death_c','flood_c','flood_e','damage_c','damage_e','stress_e','smoke_c','smoke_e','hospital_c','hospital_e']
selected_list = [wordmap[word] for word in selected_words]

for i in xrange(N): #
    X[i] = biembeds[i].T # 1 x d

print(X)

print('min:', X[idxmapping(np.argmin(X), N, d)])
print('max:', X[idxmapping(np.argmax(X), N, d)])

U, s, Vh = la.svd(X, full_matrices=False)
U_ = U[:,0:2]
print('min:', U_[idxmapping(np.argmin(U_), N, 2)])
print('max:', U_[idxmapping(np.argmax(U_), N, 2)])

#for i in xrange(N):
for i in selected_list:
    plt.text(U[i,0], U[i,1], wordlist[i])

plt.axis([-0.04, 0.04, -0.05, 0.04])
plt.show()
