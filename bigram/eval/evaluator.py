import os
import sys
import pickle
import numpy as np

def get_cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)) * np.sqrt(np.dot(v2,v2)))

def splitor_conceptnet(evalfn, wordidmap, embeddings):
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

def get_euclidean_distance(v1,v2):
    return np.sqrt(np.sum((v1 - v2)**2))

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

def bch_calculator(evalfn,outf,metric = get_cosine_similarity):
    dim = embeddings[0].shape[0]
    with open(evalfn) as f:
        for line in f:
            cause,effect,score = line.strip().split()
            try:
                score = metric(embeddings[wordidmap[cause+'_c']].reshape(dim,),embeddings[wordidmap[effect+'_e']].reshape(dim,))
                outf.write(cause+'\t'+effect+'\t'+str(score)+'\n')
            except:
                continue

def ModelEvaluation(bch_dataeset_fn, bch_result_fn):
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
                if (orglist[int(pidx1)][1] - orglist[int(pidx2)][1]) * (csdic[orglist[int(pidx1)][0]] - csdic[orglist[int(pidx2)][0]]) >= 0.0:
                    correct += 1
                    print('true',orglist[int(pidx1)],orglist[int(pidx2)],str(orglist[int(pidx1)][1] - orglist[int(pidx2)][1]),str(csdic[orglist[int(pidx1)][0]] - csdic[orglist[int(pidx2)][0]]))
                else: print('false')
            except Exception as e:
                print('false')
        print('acc:',correct / tot)

# Evaluation
embedfn = '../copa/biembeds_bz=1000.pickle'
#embedfn = '../copa/biembeds0.pickle'
with open(embedfn,'rb') as f:
    embeddings = pickle.load(f)
    embeddings -= np.mean(embeddings)

wordidmap = {}
with open('../copa/word.list','rb') as f:
    wordlist = pickle.load(f)
    for i,w in enumerate(wordlist):
        wordidmap[w] = i

#bch_result_fn = 'bch_result.txt' #'bch_result_bz=1000.txt'
bch_result_fn = 'bch_result_bz=1000.txt'
if not os.path.exists(bch_result_fn):
    with open(bch_result_fn,'w') as outf:
        bch_calculator('pos_bch_copa.txt', outf)
        bch_calculator('neg_bch_copa.txt', outf)

    with open(bch_result_fn) as f:
        dic = {}
        for line in f:
            cause,effect,score = line.strip().split()
            dic[cause+'\t'+effect] = float(score)
        dd = sorted(dic.items(),key = lambda d:-d[1])

        with open('sorted_'+bch_result_fn,'w') as outf:
            for k,v in dd:
                outf.write(k+'\t'+str(v)+'\n')

ModelEvaluation('clean_bch_dataset.txt', bch_result_fn) #bch_result_bz=1000.txt

"""
#splitor_conceptnet('conceptnet.txt',wordidmap,embeddings)
#case_splitor('stale_c',['throw_e','fire_e','disease_e','sun_e','grass_e','woman_e','wear_e','dislike_e','surprise_e'])
all_effects = [ word for word in wordlist if word.endswith('_e')]
all_causes = [ word for word in wordlist if word.endswith('_c')]
#print('average: ', str(case_splitor('stale_c',all_effects)/len(all_effects)))
print('average: ', str(case_splitor('death_e',all_causes, get_euclidean_distance)/len(all_causes)))
"""
