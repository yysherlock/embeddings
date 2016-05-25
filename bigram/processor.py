"""
Todo:
    1. Multi-threads: import threading
    2. Other models: Can we borrow something from skip-gram and CBOW models?
"""
import os
import random
import pickle
import configparser
import unittest
import numpy as np
from six.moves import xrange

class Processor(object):
    """ process causal pairs data for bigram model,
    treat pairs as bigrams """

    def __init__(self, configfilePath, sectionflag):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(configfilePath)
        self.corpusfn = self.config.get(sectionflag, 'bigram_corpus')
        self.idwordlistfn = self.config.get(sectionflag, 'id2word_list')
        self.idxpairmapfn = self.config.get(sectionflag, 'idx2pair_map')
        self.pairlistfn = self.config.get(sectionflag, 'pairidx_list')
        self.biembeddingsfn = self.config.get(sectionflag, 'bi_embeddings')

        self.dataidxs = []
        self.idwordlist = [] # [word1, word2, ...]
        self.idxpairmap = {} # {pairidx:[causewordId, effectwordId]}

        if not self.loadedOrNot():
            self.loadBigramData()

        if not self.dataidxs:
            self.dataidxs = self.loadObj(self.pairlistfn)
        if not self.idwordlist:
            self.idwordlist = self.loadObj(self.idwordlistfn)
        if not self.idxpairmap:
            self.idxpairmap = self.loadObj(self.idxpairmapfn)

        self.volcab_size = len(self.idwordlist)

    def loadObj(self, filename):
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        return obj

    def loadedOrNot(self):
        return os.path.exists(self.pairlistfn) and os.path.exists(self.corpusfn) and os.path.exists(self.idwordlistfn) and os.path.exists(self.idxpairmapfn)

    def loadBigramData(self):
        """ Read and shuffle bigrams into list, map words into ids,
        [ pair1_idx, pair2_idx, pair3_idx, ...] """

        curId = 0; curIdx = 0
        volcab = set()

        # construct indexOfpair map, and idOfword map
        with open(self.corpusfn) as f, open(self.idwordlistfn,'wb') as idwordlistf, \
                open(self.idxpairmapfn,'wb') as idxOfpairf, open(self.pairlistfn,'wb') as dataf:

            causeId, effectId = 0,0
            for line in f:
                cause,effect,freq = line.strip().split('\t')
                cause += '_c'; effect += '_e'
                if cause not in volcab:
                    volcab.add(cause)
                    #idOfwordf.write(str(curId) + ' ' + cause + '\n')
                    self.idwordlist.append(cause)
                    causeId = curId
                    curId += 1
                if effect not in volcab:
                    volcab.add(effect)
                    #idOfwordf.write(str(curId) + ' ' + effect + '\n')
                    self.idwordlist.append(effect)
                    effectId = curId
                    curId += 1

                for i in range(int(freq)): self.dataidxs.append(curIdx)
                self.idxpairmap[curIdx] = [causeId,effectId]
                #idxOfpairf.write(str(curIdx) + ' ' + str(causeId) + ',' + str(effectId) + '\n')
                curIdx += 1 # line number is the pair number

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


class TestProcessor(unittest.TestCase):

    def setUp(self):
        #self.processor = Processor('bi-config.ini','DEBUG')
        self.processor = Processor('bi-config.ini', 'COPA')
    def test_loadBigramData(self):
        obj = self.processor
        self.assertEqual(obj.loadedOrNot(), True)
    def test_transform_data(self):
        pairidxs = [1,2]
        input_data, label_data = self.processor.transform_data(pairidxs,'nec')
        print(input_data)
        print(label_data)

if __name__ == "__main__":
    unittest.main()
