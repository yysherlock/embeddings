def outputConceptnetPairs(inputfn, outf, wordset = None):
    with open(inputfn) as f:
        for line in f:
            cause,effect = line.strip().split('->')
            if wordset:
                if cause not in wordset or effect not in wordset: continue
            if ' ' not in cause and ' ' not in effect:
                outf.write(cause+'\t'+effect+'\n')

wordset = set()
with open('../copa/copaword.txt') as f:
    for line in f:
        wordset.add(line.strip())

with open('conceptnet.txt','w') as outf:
    outputConceptnetPairs('RQ1_positive.txt', outf, wordset)
    outf.write('-------\n')
    outputConceptnetPairs('RQ1_negative.txt', outf, wordset)

