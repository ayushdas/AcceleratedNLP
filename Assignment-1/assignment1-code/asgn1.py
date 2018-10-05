import re
import sys
# from random import random
import random
from math import log
from collections import defaultdict
import json

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

# Prints the trigrams and probabilities that have the char1+char2 character history
def trigram_with_two_character_history(char1,char2,tri_probs):
    prefix = char1+char2
    print('Diplaying all the n-grams and probability with the two-character history '+prefix)
    for key in  tri_probs:
        if (key.startswith(prefix)):
            print ('n-gram','\t',key,'\t',tri_probs[key])
    return            

def generate_from_LM(num_of_chars,tri_probs):
    valid_char_list = [' ','.','0']
    for i in range(ord('a'),ord('z')+1):
        valid_char_list.append(chr(i))
    if(num_of_chars == 0):
        return
    elif (num_of_chars == 1):
        return (random.choice(valid_char_list))
    elif (num_of_chars == 2):
        return (random.choice(valid_char_list))+(random.choice(valid_char_list))    
    else:
        seq = ''
        Num_Of_Chars = num_of_chars
        num_of_iter = 0
        while(len(seq) != Num_Of_Chars and num_of_iter <= 1000):  
            two_char_seq = (random.choice(valid_char_list))+(random.choice(valid_char_list))
            print(two_char_seq)    
            seq = two_char_seq
            num_of_chars = Num_Of_Chars-2
            while (num_of_chars > 0):
                prob = 0
                trigram_key = ''
                # print ('1',' Two character sequence:',two_char_seq,' |Trigram Sequence:',trigram_key) 
                foundKey = False
                for key in tri_probs:              
                    if ((key.startswith(two_char_seq)) and tri_probs[key] > prob):
                        prob = tri_probs[key]
                        trigram_key = key
                        foundKey = True
                # print ('2',' Two character sequence:',two_char_seq,' |Trigram Sequence:',trigram_key, ' |Character Extracted:',trigram_key[2:3])       
                if (foundKey == True):
                    seq = seq + trigram_key[2:3]
                    two_char_seq = trigram_key[1:3]
                else :
                    print(two_char_seq,' Key not found!')
                    break    
                num_of_chars -= 1
            num_of_iter += 1                       
    return seq


#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

with open(infile) as f:
    i = 0
    for line in f:
        i+=1
        if i <= 10000:





            # print(line)
            line = preprocess_line(line) # Pre Processes the input line
            # print(line)
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




            # break

tri_probs = estimate_probs(bi_counts,tri_counts)
save_model_to_file(tri_probs)
# trigram_with_two_character_history('n','g',tri_probs)
sequence = generate_from_LM(300,tri_probs)
print(len(sequence))
print(sequence)
# print("Bigram counts in ", infile, ", sorted alphabetically:")
# for bigram in sorted(bi_counts.keys()):
#     print(bigram, ": ", bi_counts[bigram])
# print("Trigram counts in ", infile, ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#     print(tri_count[0], ": ", str(tri_count[1]))
# print("Bigram counts in ", infile, ", sorted numerically:")
# for bi_count in sorted(bi_counts.items(), key=lambda x:x[1], reverse = True):
#     print(bi_count[0], ": ", str(bi_count[1]))

