#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import random
import math
from collections import defaultdict


tri_counts=defaultdict(int) #counts of all trigrams in input


#this function currently does nothing.
def preprocess_line(line):
    ans = ""
    accepted_chars = "abcdefghijklmnopqrstuvwxyz. "
    for char in line:
        if char.lower() in accepted_chars:
           ans += char.lower()
        elif char.isdigit():
           ans += '0'

    return ans




#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 3:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file
model = sys.argv[2]

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    for line in f:
        line = preprocess_line(line) 
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1


model2 = []
with open(model) as f:
    for line in f:
        line = line[:-1]
        trigram = line[:3]
        prob = line[4:]
        model2.append((trigram, float(prob)))
#print(model2)


#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
#print("Trigram counts in ", infile, ", sorted alphabetically:")
#for trigram in sorted(tri_counts.keys()):
    #print(trigram, ": ", tri_counts[trigram])
#print("Trigram counts in ", infile, ", sorted numerically:")
#for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    #p#rint(tri_count[0], ": ", str(tri_count[1]))


def build_trigram(tri_counts):
    bigram_counts = defaultdict(int)
    vocab  = set()
    for trigram, count in tri_counts.items():
        bigram = trigram[:2]
        bigram_counts[bigram]+=count
        vocab.update(bigram)

    length = len(vocab)
    print(length)
    
    trigram_probs = []
    for trigram, count in tri_counts.items():
        bigram = trigram[:2]
        prob = (count+1)/(bigram_counts[bigram]+length)
        trigram_probs.append((trigram, prob))
    
    
    #print(trigram_probs)

    return trigram_probs

res = build_trigram(tri_counts)

# change start to random

def generate_from_LM(trigram_prob, start, length):
    output = start
    cur = start
    for _ in range(length-2):
        possibilities = [trigram for trigram, prob in trigram_prob if trigram.startswith(cur)]
        if not possibilities:
            break
        probs = [prob for trigram, prob in trigram_prob if trigram.startswith(cur)]
        total_prob = sum(probs)
        normalized = [prob/total_prob for prob in probs]
        next = random.choices(possibilities, weights = normalized)[0]
        output+=next[2]
        cur = next[1:]
    return output

sentence = generate_from_LM(res, 'th', 300)
print(sentence)
#print(generate_from_LM(model2, 'th', 300))


def perplexity(sentence, trigram_prob):
    log_prob_sum = 0
    for i in range(2, len(sentence)):
        bigram = sentence[i-2:i]
        nextCh = sentence[i]
        curTrigram = bigram+nextCh
        prob = next((prob for trig, prob in trigram_prob if trig==curTrigram), 0)
        log_prob_sum+=math.log2(prob) if prob > 0 else float('-inf')

    perplexity = 2**(-log_prob_sum/(len(sentence)-2))
    return perplexity

print(perplexity(sentence, res))
