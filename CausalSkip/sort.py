
d = {}
with open('/home/zhiyi/projects/embeddings/CausalSkip/datasets/COPA/copa_cn.txt') as f:
    for line in f:
        cause, effect, freq = line.strip().split('\t')
        d.setdefault(cause,{})
        d[cause].setdefault(effect,0)
        d[cause][effect] += int(freq)

for k,v in d.items():
    d[k] = sorted(d[k], key=d[k].__getitem__,reverse=True)

revd={}
with open('/home/zhiyi/projects/embeddings/CausalSkip/datasets/COPA/copa_cn.txt') as f:
    for line in f:
        cause, effect, freq = line.strip().split('\t')
        revd.setdefault(effect,{})
        revd[effect].setdefault(cause,0)
        revd[effect][cause] += int(freq)

for k,v in revd.items():
    revd[k] = sorted(revd[k], key=revd[k].__getitem__,reverse=True)

with open('sorted_copa.txt','w') as outf:
    for k,v in d.items():
        outf.write(k+': '+str(v)+'\n')
        outf.flush()

    outf.write('\n')
    for k,v in revd.items():
        outf.write(k+': '+str(v)+'\n')
        outf.flush()
