import re
import sys
import random
# from random import random
from math import log
from collections import defaultdict
import linecache

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

    line = "".join(character_list).rstrip('\n')      
    return line

def estimate_probs(bi_counts,tri_counts):
    tri_probs = defaultdict(float) # probabilties of the trigram
    for item in tri_counts:
        tri_probs[item] = tri_counts[item] / bi_counts[item[0:2]] 
    return tri_probs    

def save_model_to_file(model,name):
    with open(name, 'w') as file:
        for k, v in model.items():
            file.write(str(k) + '\t'+ str(v) + '\n')

def read_model_from_file(file):
    in_model = defaultdict(int)
    with open(file) as f:
        for line in f:
            (key, val) = line.split('\t')
            in_model[key] = val.rstrip('\n')
    return in_model

def generate_debug_filepath(model):
    return '/Users/matt/Documents/Masters 2018-2019/Modules/Full/ANLP INFR11125/Assignments/AcceleratedNLP_1/Assignment-1/assignment1-data/'+model

def terminal_input_filepath():
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)
    return sys.argv[1]

def find_bi_tri_grams(line):
    for j in range(len(line)-(2)):
        trigram = line[j:j+3]
        tri_counts[trigram] += 1
        bigram = line[j:j+2]
        bi_counts[bigram] += 1
      
    final_bigram = line[len(line)-2:]
    bi_counts[final_bigram] +=1

def testing_routine(line_num,infile):
    line = linecache.getline(infile, line_num).rstrip('\n')
    print ('LINE SELECTED:')
    print (line)
    processed_line = preprocess_line(line) # removes special characters and lowercases all letters
    print ('PROCESSED LINE:')
    print (processed_line)
    find_bi_tri_grams(processed_line)
    provided_model = read_model_from_file('../assignment1-models/model-br.en')
    
def complete_model(infile):
    with open(infile) as f:
        for line in f:
                processed_line = preprocess_line(line) # Pre Processes the input line
                find_bi_tri_grams(processed_line)

def bigram_viewer(alpha,num,infile):
    if alpha:
        print("Bigram counts in ", infile, ", sorted alphabetically:")
        for bigram in sorted(bi_counts.keys()):
            print(bigram, ": ", bi_counts[bigram])
    if num:
        print("Bigram counts in ", infile, ", sorted numerically:")
        for bi_count in sorted(bi_counts.items(), key=lambda x:x[1], reverse = True):
            print(bi_count[0], ": ", str(bi_count[1]))

def trigram_viewer(alpha,num,infile):
    if alpha:
        print("Trigram counts in ", infile, ", sorted alphabetically:")
        for trigram in sorted(tri_counts.keys()):
            print(trigram, ": ", tri_counts[trigram])
    if num:
        print("Trigram counts in ", infile, ", sorted numerically:")
        for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
            print(tri_count[0], ": ", str(tri_count[1]))

def perplexity_computation(file,model):

    with open(file) as f:
        total_prob = 0
        total_tris = 0
        for line in f:
            processed_line = preprocess_line(line)
            for j in range(len(processed_line)-(2)):
                total_tris +=1
                trigram = processed_line[j:j+3]
                trigram_prob = log(model[trigram],10)
                total_prob = trigram_prob + total_prob
        # perplexity = 
        # take antilog and do to power of 1/n?  still unsure here....

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

def main_routine():

    debugger = True
    testing = True
    alpha = False
    num = False
    line_num = 4
    model_lang = 'en'
    test_file = '../assignment1-data/test'

    if debugger:
        infile = generate_debug_filepath('training.'+model_lang)
    else:
        infile = terminal_input_filepath()

    if testing:
        testing_routine(line_num,infile)
        # save_model_to_file(bi_counts,'testline.txt')
    else:
        complete_model(infile)
        tri_probs = estimate_probs(bi_counts,tri_counts)
        save_model_to_file(tri_probs,'modelv1.txt')
        sequence = generate_from_LM(300,tri_probs)
        perplexity_computation(test_file,tri_probs)

    # bigram_viewer(alpha,num,infile)
    trigram_viewer(alpha,num,infile)

if __name__ == '__main__':
    main_routine()
# do tables and graphs for presentation
# Smoothing

