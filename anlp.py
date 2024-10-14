#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import random
import math
from collections import defaultdict


tri_counts = defaultdict(int) #counts of all trigrams in input
bi_counts = defaultdict(int)
vocabLength = [0]

def preprocess_line(line):
    # defaulting to start with a # character
    ans = "#"
    accepted_chars = "abcdefghijklmnopqrstuvwxyz. "
    for char in line:
        if char.lower() in accepted_chars:
           ans += char.lower()
        # if we're at the end of a sentence, include stop character
        elif char == '\n':
            ans += '#'
        elif char.isdigit():
           ans += '0'

    return ans


if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1]

# reading sample line by line
with open(infile) as f:
    for line in f:
        line = preprocess_line(line) 
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            
            tri_counts[trigram] += 1

def build_trigram_prob_from_LM(model):
    trigram_probs = defaultdict(int)
    bi_counts_from_model = defaultdict(int)
    with open(model) as f:
        for line in f:
            line = line[:-1]
            trigram = line[:3]
            prob = line[4:]
            trigram_probs[trigram] = float(prob)
            bi_counts_from_model[trigram[:2]] += 1
    return trigram_probs, bi_counts_from_model


def build_trigram(tri_counts):
    vocab  = set()
    for trigram, count in tri_counts.items():
        bigram = trigram[:2]
        bi_counts[bigram]+=count
        vocab.update(bigram)

    vocabLength[0] = len(vocab)
    trigram_probs = defaultdict(int)
    for trigram, count in tri_counts.items():
        bigram = trigram[:2]
        prob = count/bi_counts[bigram]
        trigram_probs[trigram] = prob

    return trigram_probs

def generate_from_LM(trigram_prob, length):
    output = ''
    while len(output) < length:
        cur = '#'
        output += '#'
        for _ in range(length - len(output)):
            possibilities = [(trigram, prob) for trigram, prob in trigram_prob.items() if trigram.startswith(cur)]
            if not possibilities:
                break

            possibilities, probs = zip(*possibilities)
            total_prob = sum(probs)
            normalized = [prob/total_prob for prob in probs]
            next = random.choices(possibilities, weights = normalized)[0]
            output+=next[2]
            cur = next[1:]
    
    return output

    
trigram_probs = build_trigram(tri_counts)
model_trigram_probs, model_bigram_counts = build_trigram_prob_from_LM('./model-br.en')


sentence = generate_from_LM(trigram_probs, 300)
print(sentence)
sentence_model= generate_from_LM(model_trigram_probs, 300)
print(sentence_model)


# solve division by 0 error
def perplexity(testfile, trigram_prob):
    sentence = ''
    with open(testfile) as f:
        for line in f:
            sentence+=preprocess_line(line)

        
    log_prob_sum = 0
    for i in range(len(sentence)-2):
        trigram = sentence[i:i+3]
        bigram = trigram[:2]
        if trigram in trigram_prob:
            prob = trigram_prob[trigram]
        else:
            prob = (tri_counts[trigram]+1)/(bi_counts[bigram]+vocabLength[0])
        log_prob_sum+=math.log2(prob)

    perplexity = 2**(-log_prob_sum/(len(sentence)-1))
    return perplexity


# trigram_hash, bigram_hash = build_trigram_prob_from_LM('./training2.en')
# print(perplexity('', trigram_hash))
print(perplexity('./test', trigram_probs))
