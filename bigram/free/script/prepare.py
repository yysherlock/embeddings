import os

wordset = set()
wordfn = '../freeword.txt'
if not os.path.exists(wordfn):
    # generate wordfn
    causalnetwords = set()
    with open('/home/luozhiyi/data/causal/CausalNet.txt') as f:
        for line in f:
            cause,effect,freq = line.strip().split()
            causalnetwords.add(cause)
            causalnetwords.add(effect)

    with open(wordfn,'w') as outf, open('../copaword.txt') as f1, open('../../eval/bch.txt') as f2:
        freewords = set()
        for line in f1:
            freewords.add(line.strip())
        for line in f2:
            cause,effect,score = line.strip().split()
            if cause in causalnetwords:
                freewords.add(cause)
            if effect in causalnetwords:
                freewords.add(effect)
        for word in freewords:
            outf.write(word+'\n')

with open(wordfn) as f:
    for line in f:
        wordset.add(line.strip())

def generate_copa_causalnet(causalnetfn, outputfn):
    with open(causalnetfn) as f, open(outputfn,'w') as outf:
        for line in f:
            cause,effect,freq = line.strip().split('\t')
            if cause in wordset and effect in wordset:
                outf.write(line)
generate_copa_causalnet('/home/luozhiyi/data/causal/CausalNet.txt','free_cn.txt')
