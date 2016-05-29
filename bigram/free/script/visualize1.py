import numpy as np
import matplotlib.pyplot as plt
from six.moves import xrange
import os
import pickle

normalizedfn = '../normalized_reduced.txt'

with open('../word.list','rb') as f:
    wordlist = pickle.load(f)

if not os.path.exists(normalizedfn):
    d = np.loadtxt('../reduced.txt')
    d -= d.min(axis = 0); d /= d.max(axis = 0)
    np.savetxt(normalizedfn, d , fmt='%.8f', delimiter='\t')

d = np.loadtxt(normalizedfn)
v = d.shape[0]
#print(d.shape) # V x 2

# generate selected_list
wordmap = {}
for i in xrange(len(wordlist)):
    wordmap[wordlist[i]] = i
#selected_words = ['smoke_e','fire_c','fire_e','death_e','death_c','flood_c','flood_e','damage_c','damage_e','stress_e','smoke_c','smoke_e','disease_c','disease_e']
selected_words = ['kill_c','guilty_e','happy_e', 'gift_c', 'fire_c', 'property_e', 'flood_c', 'damage_e']
selected_list = [wordmap[word] for word in selected_words]

#for i in xrange(v):
for i in selected_list:
    plt.text(d[i,0], d[i,1], wordlist[i])

plt.show()
