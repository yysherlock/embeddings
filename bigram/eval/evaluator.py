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

# Evaluation
with open('../copa/biembeds0.pickle','rb') as f:
    embeddings = pickle.load(f)
    embeddings -= np.mean(embeddings)

wordidmap = {}
with open('../copa/word.list','rb') as f:
    wordlist = pickle.load(f)
    for i,w in enumerate(wordlist):
        wordidmap[w] = i

#splitor_conceptnet('conceptnet.txt',wordidmap,embeddings)
#case_splitor('stale_c',['throw_e','fire_e','disease_e','sun_e','grass_e','woman_e','wear_e','dislike_e','surprise_e'])
all_effects = [ word for word in wordlist if word.endswith('_e')]
all_causes = [ word for word in wordlist if word.endswith('_c')]
#print('average: ', str(case_splitor('stale_c',all_effects)/len(all_effects)))
print('average: ', str(case_splitor('death_e',all_causes, get_euclidean_distance)/len(all_causes)))

