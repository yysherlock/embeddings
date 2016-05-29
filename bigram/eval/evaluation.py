from __future__ import division
import sys,os
import random
import pickle
import numpy as np

def outputConceptnetPairs(inputfn, outf, wordset = None):
    with open(inputfn) as f:
        for line in f:
            cause,effect = line.strip().split('->')
            if wordset:
                if cause not in wordset or effect not in wordset: continue
            if ' ' not in cause and ' ' not in effect:
                outf.write(cause+'\t'+effect+'\n')

def generateConceptNetDataset(conceptnetfn):
    """ # generateConceptNetDataset('conceptnet.txt') """
    with open(conceptnetfn,'w') as outf:
        outputConceptnetPairs('RQ1_positive.txt', outf, wordset)
        outf.write('-------\n')
        outputConceptnetPairs('RQ1_negative.txt', outf, wordset)

def outputCopaBch(inputfn, outputfn, thred=0.0, wordset = None):
    """
    #outputCopaBch('bch.txt','pos_bch_copa.txt', thred=5.0, wordset=wordset)
    #outputCopaBch('bch_neg.txt','neg_bch_copa.txt', wordset=wordset)
    """
    with open(inputfn) as f, open(outputfn,'w') as outf:
        for line in f:
            cause, effect, score = line.strip().split()
            if float(score) < thred: continue
            if wordset:
                if cause not in wordset or effect not in wordset: continue
            outf.write(line)

def getbchData(bch_result_fn, bch_pos_fn, bch_neg_fn, outputfn):
    """ getbchData('bch_result.txt','pos_bch_copa.txt','neg_bch_copa.txt','clean_bch_dataset.txt') """
    with open(bch_result_fn) as f, open(outputfn,'w') as outf:
        pairs = ['\t'.join(line.strip().split()[:2]) for line in f.readlines()]
        with open(bch_pos_fn) as f1:
            for line in f1:
                cause,effect,score = line.strip().split()
                if cause+'\t'+effect in pairs:
                    outf.write(line)
        with open(bch_neg_fn) as f1:
            for line in f1:
                cause,effect,score = line.strip().split()
                if cause+'\t'+effect in pairs:
                    outf.write(line)

def generateCounterPairs(counter_pair_fn):
    """
    generateCounterPairs('counter_pair.txt')
    """
    if os.path.exists(counter_pair_fn):
        return

    org = [] # [[pair1,score1], ...]
    result = []

    with open('clean_bch_dataset.txt') as f1:
        for l1 in f1:
            cause,effect,score = l1.strip().split()
            org.append([cause+'\t'+effect,float(score)])

    cnt = 0
    size = len(org)
    poslen = 379
    compairs = set({})
    outf = open(counter_pair_fn,'w')
    while cnt < 300:
        t1,t2 = random.randint(0,poslen), random.randint(poslen,size-1)
        if str((t1,t2)) in compairs: continue
        cnt += 1
        outf.write(str(t1)+'\t'+str(t2)+'\n')
        compairs.add(str((t1,t2))); compairs.add(str((t2,t1)))
    outf.close()

def splitor_conceptnet(evalfn, wordidmap, embeddings):
    """ Evaluation 1:
    see average cs of pos pairs and that of neg pairs in conceptnet.txt
    >>> splitor_conceptnet('conceptnet.txt',wordidmap,embeddings)
    """
    dim = embeddings[0].shape[0]
    pos_sum, neg_sum = 0.0,0.0
    pos_cnt, neg_cnt = 0,0
    splitFlag = False

    with open(evalfn) as f:
        for line in f:
            try:
                cause,effect = line.strip().split('\t')
                cause += '_c'
                effect += '_e'
                v_c = embeddings[wordidmap[cause]].reshape(dim,)
                v_e = embeddings[wordidmap[effect]].reshape(dim,)
                print(line.strip()+'\t'+str(get_cosine_similarity(v_c,v_e)))
                if not splitFlag:
                    pos_sum += get_cosine_similarity(v_c,v_e)
                    pos_cnt += 1
                else:
                    neg_sum += get_cosine_similarity(v_c,v_e)
                    neg_cnt += 1
            except:
                print('-------')
                splitFlag = True
                #continue
    print('pos_avg similarity:',str(pos_sum/pos_cnt))
    print('neg_avg similarity:',str(neg_sum/neg_cnt))

def case_splitor(center_word, candidates, metric):
    dim = embeddings[0].shape[0]
    center_vec = embeddings[wordidmap[center_word]].reshape(dim,)
    s = 0.0
    print(center_word)
    for candidate in candidates:
        outside_vec = embeddings[wordidmap[candidate]].reshape(dim,)
        sim = metric(center_vec, outside_vec)
        s += sim
        print('\t'+candidate,str(sim))
    return s

def get_cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)))

def get_euclidean_distance(v1,v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def bch_calculator(embeddings,evalfn,outf,metric = get_cosine_similarity):
    dim = embeddings[0].shape[0]
    with open(evalfn) as f:
        for line in f:
            cause,effect,score = line.strip().split()
            try:
                score = metric(embeddings[wordidmap[cause+'_c']].reshape(dim,),embeddings[wordidmap[effect+'_e']].reshape(dim,))
                outf.write(cause+'\t'+effect+'\t'+str(score)+'\n')
            except:
                continue

def generate_bch_result(bch_result_fn, embeddings=None, csdatafn=None):
    #bch_result_fn = 'bch_result.txt'
    #bch_result_fn = 'bch_result_bz=1000.txt'

    with open(bch_result_fn,'w') as outf:
        if not csdatafn: # embed model
            bch_calculator(embeddings,'pos_bch_copa.txt', outf)
            bch_calculator(embeddings,'neg_bch_copa.txt', outf)
        else: # KR 2016 model
            # scan CS score (KR 2016), generate cs_dataset
            org, csdic = {}, {}
            with open('clean_bch_dataset.txt') as f:
                for l in f:
                    cause,effect,score = l.strip().split()
                    org[cause + '\t' + effect] = float(score)

            with open(csdatafn) as f:
                for line in f:
                    cause,effect,score = line.strip().split()
                    k = cause + '\t' + effect
                    if k in org:
                        csdic[k] = float(score)
            with open('clean_bch_dataset.txt') as f:
                for l in f:
                    cause,effect,score = l.strip().split()
                    k = cause + '\t' + effect
                    if k in csdic:
                        outf.write(k+'\t'+str(csdic[k])+'\n')
                    else:
                        outf.write(k+'\t'+str(0.0)+'\n')


def ModelEvaluation(bch_dataeset_fn, bch_result_fn):
    """
    bch_dataeset_fn: labeled dataset
    bch_result_fn: result of our model
    """
    csdic = {}
    with open(bch_result_fn) as f:
        for line in f:
            cause,effect,score = line.strip().split()
            k = cause + '\t' + effect
            csdic[k] = float(score)

    orglist = []
    with open(bch_dataeset_fn) as f:
        for l in f:
            cause,effect,score = l.strip().split()
            orglist.append([cause+'\t'+effect,float(score)])

    with open('counter_pair.txt') as f:
        tot,correct = 0,0
        for line in f:
            pidx1, pidx2 = line.strip().split()
            tot += 1
            try:
                if (orglist[int(pidx1)][1] - orglist[int(pidx2)][1]) * (csdic[orglist[int(pidx1)][0]] - csdic[orglist[int(pidx2)][0]]) > 0.0:
                    correct += 1
                    print('true',orglist[int(pidx1)],orglist[int(pidx2)],str(orglist[int(pidx1)][1] - orglist[int(pidx2)][1]),str(csdic[orglist[int(pidx1)][0]] - csdic[orglist[int(pidx2)][0]]))
                else: print('false',orglist[int(pidx1)],orglist[int(pidx2)],str(orglist[int(pidx1)][1] - orglist[int(pidx2)][1]),str(csdic[orglist[int(pidx1)][0]] - csdic[orglist[int(pidx2)][0]]))
            except Exception as e:
                print('false')
        print('acc:',correct / tot)



wordset = set()
with open('../copa/copaword.txt') as f:
    for line in f:
        wordset.add(line.strip())

with open('../copa/biembeds0.pickle','rb') as f:
    embeddings0 = pickle.load(f)
    embeddings0 -= np.mean(embeddings0)
#embedfn = '../copa/biembeds0.pickle'
with open('../copa/biembeds_bz=1000.pickle','rb') as f:
    embeddings1 = pickle.load(f)
    embeddings1 -= np.mean(embeddings1)


wordidmap = {}
with open('../copa/word.list','rb') as f:
    wordlist = pickle.load(f)
    for i,w in enumerate(wordlist):
        wordidmap[w] = i

generateCounterPairs('counter_pair.txt')
generate_bch_result('bch_result.txt',embeddings0)
generate_bch_result('bch_result_bz=1000.txt',embeddings1)
generate_bch_result('cs_result.txt', csdatafn='/home/luozhiyi/data/cs-0.9.txt')
generate_bch_result('cs_result1.txt', csdatafn='/home/luozhiyi/data/cs-1.0.txt')
generate_bch_result('freq_result.txt', csdatafn='/home/luozhiyi/data/causal/CausalNet.txt')
#bch_result_fn = 'bch_result.txt'
#bch_result_fn = 'bch_result_bz=1000.txt'
bch_dataeset_fn = 'clean_bch_dataset.txt'
ModelEvaluation(bch_dataeset_fn,'bch_result.txt') # 0.5533
ModelEvaluation(bch_dataeset_fn,'bch_result_bz=1000.txt') # 0.4666
ModelEvaluation(bch_dataeset_fn,'cs_result.txt') # 0.6333
ModelEvaluation(bch_dataeset_fn,'cs_result1.txt') # 0.6233
ModelEvaluation(bch_dataeset_fn,'freq_result.txt') # 0.4966
