
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
        self.idxpairmapfn = self.config.get(sectionflag, 'idx2pair_map')
        self.pairlistfn = self.config.get(sectionflag, 'pairidx_list')
        self.causepriorfn = self.config.get(sectionflag, 'cause_prior')
        self.effectpriorfn = self.config.get(sectionflag, 'effect_prior')
        self.causedictfn = self.config.get(sectionflag, 'cause_dict') # {cause1:{effect1:p1,effect2:p2,...}, ...}
        self.effectdictfn = self.config.get(sectionflag, 'effect_dict')

        self.dataidxs = []
        self.idwordlist = [] # [word1, word2, ...]
        self.tokens = {} # {w1:idx1, w2:idx2, ...}
        self.idxpairmap = {} # {pairidx:[causewordId, effectwordId]}
        self.causeprior = []
        self.effectprior = []
        self.causedict = {}
        self.effectdict = {}

        if not self.loadedOrNot():
            self.loadDatasets()

        if not self.dataidxs:
            self.dataidxs = self.loadObj(self.pairlistfn)
        if not self.idwordlist:
            self.idwordlist = self.loadObj(self.idwordlistfn)
        if not self.idxpairmap:
            self.idxpairmap = self.loadObj(self.idxpairmapfn)
        if not self.causeprior:
            self.causeprior = self.loadObj(self.causepriorfn)
        if not self.effectprior:
            self.effectprior = self.loadObj(self.effectpriorfn)
        if not self.causedict:
            self.causedict = self.loadObj(self.causedictfn)
        if not self.effectdict:
            self.effectdict = self.loadObj(self.effectdictfn)

        self.volcab_size = len(self.idwordlist)

    def loadObj(self, filename):
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        return obj

    def loadedOrNot(self):
        return os.path.exists(self.pairlistfn) and os.path.exists(self.corpusfn) and \
        os.path.exists(self.idwordlistfn) and os.path.exists(self.idxpairmapfn) and \
        os.path.exists(self.causepriorfn) and os.path.exists(self.effectpriorfn) and \
        os.path.exists(self.causedictfn) and os.path.exists(self.effectdictfn)

    def loadDatasets(self):
        """ Read and shuffle bigrams into list, map words into ids,
        [ pair1_idx, pair2_idx, pair3_idx, ...] """

        curId = 0; curIdx = 0
        volcab = set()

        # construct indexOfpair map, and idOfword map
        with open(self.corpusfn) as f, open(self.idwordlistfn,'wb') as idwordlistf, \
                open(self.idxpairmapfn,'wb') as idxOfpairf, open(self.pairlistfn,'wb') as dataf, \
                open(self.causepriorfn,'wb') as causepriorf, open(self.effectpriorfn,'wb') as effectpriorf, \
                open(self.causedictfn,'wb') as causedictf, open(self.effectdictfn,'wb') as effectdictf:

            causeId, effectId, tot = 0,0,0
            
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
                    #idOfwordf.write(str(curId) + ' ' + cause + '\n')
                    self.idwordlist.append(cause)
                    causeId = curId
                    self.tokens[cause] = curId
                    self.causeprior.append(freq)
                    self.effectprior.append(0)
                    curId += 1
                else:
                    self.causeprior[self.tokens[cause]] += freq
                
                self.causedict.setdefault(cause,{})
                self.causedict[cause][effect] = freq

                if effect not in volcab:
                    volcab.add(effect)
                    #idOfwordf.write(str(curId) + ' ' + effect + '\n')
                    self.idwordlist.append(effect)
                    effectId = curId
                    self.tokens[effect] = curId
                    self.effectprior.append(freq)
                    self.causeprior.append(0)
                    curId += 1
                else:
                    self.effectprior[self.tokens[effect]] += freq
                
                self.effectdict.setdefault(effect,{})
                self.effectdict[effect][cause] = freq

                tot += freq
                for i in range(freq): self.dataidxs.append(curIdx)
                self.idxpairmap[curIdx] = [causeId,effectId]
                #idxOfpairf.write(str(curIdx) + ' ' + str(causeId) + ',' + str(effectId) + '\n')
                curIdx += 1 # line number is the pair number

            tot = float(tot)
            for cause in self.causedict.keys():
                causetot = self.causeprior[self.tokens[cause]]
                for effect in self.causedict[cause]:
                    self.causedict[cause][effect] /= causetot
            for effect in self.effectdict.keys():
                effecttot = self.effectprior[self.tokens[effect]]
                for cause in self.effectdict[effect]:
                    self.effectdict[effect][cause] /= effecttot

            self.causeprior = list(np.array(self.causeprior) / tot)
            self.effectprior = list(np.array(self.effectprior) / tot)

            pickle.dump(self.causeprior, causepriorf)
            pickle.dump(self.effectprior, effectpriorf)

            pickle.dump(self.causedict, causedictf)
            pickle.dump(self.effectdict, effectdictf)

            pickle.dump(self.idxpairmap, idxOfpairf)
            pickle.dump(self.idwordlist, idwordlistf)

            random.shuffle(self.dataidxs)
            pickle.dump(self.dataidxs, dataf)

    def transform_data(self,pairidxs,causality): # pairidxs: a list of pairidx
        # transform context/outside word as input
        # transform target/center word as label
        batch_size = len(pairidxs)
        volcab_size = len(self.idwordlist)
        input_data = np.zeros([batch_size, self.volcab_size]) # N x V
        label_data = np.zeros([batch_size, self.volcab_size]) # N x V

        contextIdx, targetIdx = 0, 1 # cause word as context, effect word as target, w_e | w_c, sufficiency causality
        if causality=='nec': contextIdx, targetIdx = targetIdx, contextIdx

        for i in xrange(batch_size):
            context_word_id = self.idxpairmap[pairidxs[i]][contextIdx]
            target_word_id = self.idxpairmap[pairidxs[i]][targetIdx]
            input_data[i,context_word_id] = 1.0
            label_data[i,target_word_id] = 1.0

        return (input_data, label_data)

    def getRandomContext(self, lambd=0.5, C=5):

        # sample a cause or effect word as center word
        center = np.random.choice(np.array(["cause","effect"]), [1 - lambd, lambd])
        center_prior = getattr(self, center+"prior")
        centerWord = self.tokens[np.random.choice(np.arange(0,len(center_prior)), center_prior)]
        context_distribution = getattr(self, center+"dict")[centerWord]

        context_candidates, context_prior = zip(*context_distribution.items())

        # select 2C context words for the center word
        context = np.random.choice(context_candidates, size = 2*C, p = context_prior)

        if len(context) > 0:
            return centerWord, context
        else:
            return self.getRandomContext(C)


class TestCausalNet(unittest.TestCase):

    def setUp(self):
        #self.processor = Processor('bi-config.ini','DEBUG')
        self.processor = Processor('bi-config.ini', 'COPA')
    def test_loadData(self):
        obj = self.processor
        self.assertEqual(obj.loadedOrNot(), True)


def main():
    # generate datasets for each corpus
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    configPath = 'bi-config.ini'
    config.read(configPath)

    for section in config.sections():
        print("datasets:",section)
        processor = Causal(configPath, section)



if __name__ == "__main__":
    #unittest.main()
    main()
