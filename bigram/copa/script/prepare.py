wordset = set()
with open('../copaword.txt') as f:
    for line in f:
        wordset.add(line.strip())

def generate_copa_causalnet(causalnetfn, outputfn):
    with open(causalnetfn) as f, open(outputfn,'w') as outf:
        for line in f:
            cause,effect,freq = line.strip().split('\t')
            if cause in wordset and effect in wordset:
                outf.write(line)
generate_copa_causalnet('/home/luozhiyi/data/causal/CausalNet.txt','copa_cn.txt')
