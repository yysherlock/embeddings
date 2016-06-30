
import os
import random
import pickle
import configparser
import unittest
import numpy as np
from six.moves import xrange

class Causal(object):
    """ process causal pairs data for bigram model,
    treat pairs as bigrams """

    def __init__(self, configfilePath, sectionflag):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(configfilePath)
        self.corpusfn = self.config.get(sectionflag, 'corpus')
        self.idwordlistfn = self.config.get(sectionflag, 'id2word_list')
        self.tokensfn = self.config.get(sectionflag, 'tokens')
        self.causepriorfn = self.config.get(sectionflag, 'cause_prior')
        self.effectpriorfn = self.config.get(sectionflag, 'effect_prior')
        self.causedictfn = self.config.get(sectionflag, 'cause_dict') # {cause1:{effect1:p1,effect2:p2,...}, ...}
        self.effectdictfn = self.config.get(sectionflag, 'effect_dict')
        self.tokensfn = self.config.get(sectionflag, 'tokens')

        self.idwordlist = [] # [word1, word2, ...]
        self.tokens = {} # {w1:idx1, w2:idx2, ...}
        self.causeprior = []
        self.effectprior = []
        self.causedict = {}
        self.effectdict = {}

        if not self.loadedOrNot():
            self.loadDatasets()

        if not self.idwordlist:
            self.idwordlist = self.loadObj(self.idwordlistfn)
        if not self.causeprior:
            self.causeprior = self.loadObj(self.causepriorfn)
        if not self.effectprior:
            self.effectprior = self.loadObj(self.effectpriorfn)
        if not self.causedict:
            self.causedict = self.loadObj(self.causedictfn)
        if not self.effectdict:
            self.effectdict = self.loadObj(self.effectdictfn)
        if not self.tokens:
            self.tokens = self.loadObj(self.tokensfn)

        self.volcab_size = len(self.idwordlist)

        self.cause_offset = 0
        self.effect_offset = len(self.causeprior)

    def loadObj(self, filename):
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        return obj

    def loadedOrNot(self):
        return os.path.exists(self.corpusfn) and os.path.exists(self.idwordlistfn) and \
        os.path.exists(self.causepriorfn) and os.path.exists(self.effectpriorfn) and \
        os.path.exists(self.causedictfn) and os.path.exists(self.effectdictfn) and \
        os.path.exists(self.tokensfn)

    def loadDatasets(self):
        """ Read and shuffle bigrams into list, map words into ids,
        [ pair1_idx, pair2_idx, pair3_idx, ...] """

        curId = 0; curIdx = 0
        volcab = set()

        # construct indexOfpair map, and idOfword map
        with open(self.corpusfn) as f, open(self.idwordlistfn,'wb') as idwordlistf, \
                open(self.causepriorfn,'wb') as causepriorf, open(self.effectpriorfn,'wb') as effectpriorf, \
                open(self.causedictfn,'wb') as causedictf, open(self.effectdictfn,'wb') as effectdictf, \
                open(self.tokensfn, 'wb') as tokensf:

            causeId, effectId, tot = 0,0,0
            causewords, effectwords = [], []
            causetokens, effecttokens = {}, {}

            for line in f:

                if not curIdx % 10000:
                    print("line:", curIdx)

                cause,effect,freq = line.strip().split('\t')

                try:
                    freq = int(freq)
                except:
                    freq = max(10, int(np.log2(1 + float(freq)*100)))

                cause += '_c'; effect += '_e'

                if cause not in volcab:
                    volcab.add(cause)
                    causewords.append(cause)
                    causetokens[cause] = causeId
                    causeId += 1
                    self.causeprior.append(freq)

                else:
                    self.causeprior[causetokens[cause]] += freq

                self.causedict.setdefault(cause,{})
                self.causedict[cause][effect] = freq

                if effect not in volcab:
                    volcab.add(effect)
                    effectwords.append(effect)
                    effecttokens[effect] = effectId
                    effectId += 1
                    self.effectprior.append(freq)

                else:
                    self.effectprior[effecttokens[effect]] += freq

                self.effectdict.setdefault(effect,{})
                self.effectdict[effect][cause] = freq

                tot += freq
                curIdx += 1 # line number is the pair number

            tot = float(tot)
            # combine tokens
            self.effect_offset = len(causetokens)
            for k in effecttokens.keys(): effecttokens[k] += self.effect_offset
            self.tokens = causetokens
            self.tokens.update(effecttokens)
            pickle.dump(self.tokens, tokensf)

            # combine idwordlist
            self.idwordlist = causewords + effectwords
            pickle.dump(self.idwordlist, idwordlistf)

            for cause in self.causedict.keys():
                causetot = self.causeprior[self.tokens[cause]]
                for effect in self.causedict[cause]:
                    self.causedict[cause][effect] /= causetot
            for effect in self.effectdict.keys():
                effecttot = self.effectprior[self.tokens[effect] - self.effect_offset]
                for cause in self.effectdict[effect]:
                    self.effectdict[effect][cause] /= effecttot
            pickle.dump(self.causedict, causedictf)
            pickle.dump(self.effectdict, effectdictf)

            self.causeprior = list(np.array(self.causeprior) / tot)
            tmp = np.array(self.causeprior)**0.75
            self.causeprior = list(tmp/np.sum(tmp))

            self.effectprior = list(np.array(self.effectprior) / tot)
            tmp = np.array(self.effectprior)**0.75
            self.effectprior = list(tmp/np.sum(tmp))

            pickle.dump(self.causeprior, causepriorf)
            pickle.dump(self.effectprior, effectpriorf)

    def getRandomContext(self, C=5, lambd=0.5):

        # sample a cause or effect word as center word
        center_type = np.random.choice(np.array(["cause","effect"]), p=[1.0 - lambd, lambd])
        if center_type=="cause": target_type = "effect"
        else: target_type = "cause"

        offset = getattr(self, center_type+"_offset")
        center_prior = getattr(self, center_type+"prior")
        centerWord = self.idwordlist[np.random.choice(np.arange(0, len(center_prior)), p=center_prior) + offset]
        context_distribution = getattr(self, center_type+"dict")[centerWord]

        context_candidates, context_prior = zip(*context_distribution.items())

        # select 2C context words for the center word
        context = np.random.choice(context_candidates, size = 2*C, p = context_prior)

        if len(context) > 0:
            return center_type, target_type, centerWord, context
        else:
            return self.getRandomContext(C)

    def sampleTokenIdx(self, target_type):
        prior = getattr(self,target_type+'prior')
        return np.random.choice(np.arange(0,len(prior)), p=prior)


def main():
    # generate datasets for each corpus
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    configPath = 'bi-config.ini'
    config.read(configPath)

    for section in config.sections():
        if section=="CausalNet":
            print("datasets:",section)
            processor = Causal(configPath, section)



if __name__ == "__main__":
    main()
