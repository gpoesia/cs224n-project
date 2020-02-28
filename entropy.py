import json
import math
import itertools

with open('medium.json','r') as f:
    j = json.loads(f.read())

print(j['Java']['train'][0])
#def getTraining(lang)

def breakIntoSequences(line,length):
    s = list(filter(lambda c: ord(c)<2**7, line))
    return [tuple(s[i:i+length]) for i in range(0, len(s)-length)]

def lineToEntropy(line, length):
    sequences = breakIntoSequences(line, length)
    unique_prefixes = set(map(lambda x:x[:-1], sequences))
    prev_next_dict = {prefix:[] for prefix in unique_prefixes}
    for sequence in sequences:
        prev_next_dict[tuple(sequence[:-1])].append(sequence[-1])
    entropy = 0
    for prev in prev_next_dict:
        prob_prev = len(prev_next_dict[prev])/len(sequences)
        for unique_next in set(prev_next_dict[prev]):
            prob_given_prev = prev_next_dict[prev].count(unique_next)/ len(prev_next_dict[prev])
            prob = prob_given_prev * prob_prev
            entropy -= prob_prev*prob_given_prev*math.log2(prob_given_prev) #prob*math.log2(prob)
    return entropy

def average(lst): return sum(lst)/len(lst)

def naiveEntropy(lang):
    lines = j[lang]['train']
    def getLineNaiveEntropy(line):
        probs = [line.count(x)/float(len(line)) for x in set(''.join(line))]
        return -sum(map(lambda p: p*math.log2(p), probs))
    return average(list(map(getLineNaiveEntropy, lines)))

def calcEntropyGivenLength(lang, length):
    lines = j[lang]['train']
    return average(list(map(lambda line: lineToEntropy(line, length), lines)))

print(naiveEntropy('Java'))
print(calcEntropyGivenLength('Java',1))

#getLineNaiveEntropy((''.join(j['Java']['train']))[:100])

#>>> list(map(lambda n: calcEntropyGivenLength('Python',n), range(1,5)))
#[3.706962778894826, 0.7724645205142203, 0.10236192891270411, 0.03426463399819952]
#>>> list(map(lambda n: calcEntropyGivenLength('Haskell',n), range(1,5)))
#[3.675195333821533, 0.7988870412408351, 0.10757292736873808, 0.04639521774967454]
#>>> list(map(lambda n: calcEntropyGivenLength('Java',n), range(1,5)))
#[3.885628680955303, 0.8516079495611222, 0.10855849802518813, 0.04063378757033601]

#for n in range(1,10): print(n, lineToEntropy('\n'.join(j['Python']['train']),n))
#for n in range(1,10): print(n, lineToEntropy('\n'.join(j['Haskell']['train']),n))
#for n in range(1,10): print(n, lineToEntropy('\n'.join(j['Java']['train']),n))
