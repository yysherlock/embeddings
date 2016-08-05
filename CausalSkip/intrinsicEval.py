import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import configparser
import pickle
import matplotlib.pyplot as plt
from six.moves import xrange
import glob
from sgd import load_saved_params, sgd

def loadObj(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)

for section in config.sections():
    if not section=="COPA": continue
    datasets_dir = config.get(section, "datasets_dir")
    tokens = loadObj(config.get(section, "tokens"))
    wordlist = loadObj(config.get(section, "id2word_list"))
    causeprior = loadObj(config.get(section, "cause_prior"))

    N = len(causeprior)

    for f in glob.glob(datasets_dir+"/GloveInit/dim=*"):
        # Load the causal vectors we trained earlier
        if not os.listdir(f): continue
        print(f)
        _, wordVectors, _, _ = load_saved_params(f)
        causeVectors, effectVectors = wordVectors[:N,:], wordVectors[N:,:]

        visualizeWords = ['kill_c','guilty_e','happy_e', 'gift_c', 'fire_c',
        'property_e', 'flood_c', 'damage_e', 'war_c', 'hide_c', 'surprise_e',
        'coupon_c', 'discount_e','click_e','death_e','child_e','add_e',
        'information_e','email_e','contact_e','information_c','like_c','find_c',
        'like_c','look_c',
        'intention_c','interrupt_c','intersection_c']

        #visualizeWords = wordlist
        #visualizeWords = wordlist[0:N]
        """
        indices = list(np.random.choice(np.arange(0,len(wordlist)),size=20,replace=False))
        visualizeWords = [ wordlist[idx] for idx in indices ]
        """
        visualizeIdx = [tokens[word] for word in visualizeWords]
        visualizeVecs = wordVectors[visualizeIdx, :]
        temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
        covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
        U,S,V = np.linalg.svd(covariance)
        coord = temp.dot(U[:,0:2])

        for i in xrange(len(visualizeWords)):
            plt.text(coord[i,0], coord[i,1], visualizeWords[i],
            	bbox=dict(facecolor='green', alpha=0.1))

        plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
        plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

        plt.savefig(f+'/causal_vectors.png')
        #plt.show()
        plt.clf()
