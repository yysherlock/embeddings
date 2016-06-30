import numpy as np
import matplotlib.pyplot as pyplot
import glob
from sgd import load_saved_params, sgd

def loadObj(self, filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
configPath = 'bi-config.ini'
config.read(configPath)

for section in config.sections():
    if not section=="CausalNet": continue
    datasets_dir = config.get(section, "datasets_dir")
    tokens = loadObj(config.get(section, "tokens"))
    wordlist = loadObj(config.get(section, "wordlist"))
    for f in glob.glob(datasets_dir+"/dim=*"):
        # Load the causal vectors we trained earlier
        _, wordVectors, _, N = load_saved_params(f)
        causeVectors, effectVectors = wordVectors[:N,:], wordVectors[N:,:]

        #visualizeWords = ['kill_c','guilty_e','happy_e', 'gift_c', 'fire_c',
        #'property_e', 'flood_c', 'damage_e', 'war_c', 'hide_c', 'surprise_e',
        #'coupon_c', 'discount_e']

        visualizeWords = wordlist

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
        plt.show()
