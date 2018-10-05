import re
import sys
from random import random
from math import log
from collections import defaultdict
import json
import os

tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all bigrams in input

# Check for English Character
def isEngAlpha(character):
    if (character >= 'a' and character <= 'z'):
        return True
    elif (character >= 'A' and character <= 'Z'):
        return True
    else:
        return False

#Pre Processes the input line: by keeping only certain characters in the set {alpha[lowercase],space,isdidgit,.}
def preprocess_line(line):
    character_list = list()
    for character in line:
        if (isEngAlpha(character)): # character.isalpha(): True for other language characters also, hence use isEngAlpha()
            character_list.append(character.lower()) # keep lower characters
        elif (character.isspace() or character == "."): 
            character_list.append(character) # keep ' '
        elif (character.isdigit()):
            character_list.append('0') # convert digits {0-9} to 0

    line = "".join(character_list)      
    return line

def estimate_probs(bi_counts,tri_counts):
    tri_probs = defaultdict(float) # probabilties of the trigram
    for item in tri_counts:
        tri_probs[item] = tri_counts[item] / bi_counts[item[0:2]] 
    return tri_probs    

def save_model_to_file(model):
    with open('file.txt', 'w') as file:
        for k, v in model.items():
            file.write(str(k) + '\t'+ str(v) + '\n')

def generate_from_LM(model):
    pass

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
# if len(sys.argv) != 2:
#     print("Usage: ", sys.argv[0], "<training_file>")
#     sys.exit(1)
infile = '/Users/matt/Documents/Masters 2018-2019/Modules/Full/ANLP INFR11125/Assignments/AcceleratedNLP_1/Assignment-1/assignment1-data/training.en'

# infile = sys.argv[1] #get input argument: the training file

with open(infile) as f:
    i = 0
    for line in f:
        i+=1
        if i == 4:
# write question 5 and getting model from file
            print(line)
            line = preprocess_line(line) # Pre Processes the input line
            print(line)
            for j in range(len(line)-(3)):
                trigram = line[j:j+3]
                tri_counts[trigram] += 1

            for j in range(len(line)-(2)):
                bigram = line[j:j+2]
                bi_counts[bigram] += 1
                # bigram = line[j:j+2]
                # bi_counts[bigram] +=1
                
            # bigram = line[len(line)-3:]
            # bi_counts[bigram] +=1
            # print (bigram)
            break

tri_probs = estimate_probs(bi_counts,tri_counts)
save_model_to_file(tri_probs)
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
# print("Bigram counts in ", infile, ", sorted alphabetically:")
# for bigram in sorted(bi_counts.keys()):
#     print(bigram, ": ", bi_counts[bigram])
# print("Trigram counts in ", infile, ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#     print(tri_count[0], ": ", str(tri_count[1]))
# print("Bigram counts in ", infile, ", sorted numerically:")
# for bi_count in sorted(bi_counts.items(), key=lambda x:x[1], reverse = True):
#     print(bi_count[0], ": ", str(bi_count[1]))

