import re
import sys
import random
import os
# from random import random
from math import log10
from math import pow
from collections import defaultdict
import linecache
import numpy as np
import time

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
            # print (line)
            (key, val) = line.split('\t')
            in_model[key] = float(val.rstrip('\n')) ## Convert values to float
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
    # provided_model = read_model_from_file('../assignment1-models/model-br.en')
    
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
        trigram_prob = 0
        for line in f:
            processed_line = preprocess_line(line)
            # processed_line = line
            for j in range(len(processed_line)-(2)):
                total_tris +=1
                trigram = processed_line[j:j+3]
                # print (trigram)
                if trigram in model.keys():
                    trigram_prob = log10(model[trigram])
                else:
                    # print('unseen trigram: ' + trigram)
                    
                    trigram_prob = log10( 1 / ( bi_counts[trigram[:2]] + len(tri_counts)) )
                total_prob += trigram_prob 
        log_perplexity = total_prob*(-1/total_tris)
        perplexity = pow(10,log_perplexity)
    return perplexity

def init_dummy_model():
    distribution = dict([('##a', 0.2),
                    ('#aa', 0.2),
                    ('#ba', 0.15),
                    ('aaa', 0.4),
                    ('aba', 0.6),
                    ('baa', 0.25),
                    ('bba', 0.5),
                    ('##b', 0.8),
                    ('#ab', 0.7),
                    ('#bb', 0.75),
                    ('aab', 0.5),
                    ('abb', 0.3),
                    ('bab', 0.65),
                    ('bbb', 0.4),
                    ('###', 0.0),
                    ('#a#', 0.1),
                    ('#b#', 0.1),
                    ('aa#', 0.1),
                    ('ab#', 0.1),
                    ('ba#', 0.1),
                    ('bb#', 0.1)])
    return distribution

def trigram_with_two_character_history(char1,char2,tri_probs):
    prefix = char1+char2
    print('Diplaying all the n-grams and probability with the two-character history '+prefix)
    for key in  tri_probs:
        if (key.startswith(prefix)):
            print ('n-gram','\t',key,'\t',tri_probs[key])
                
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
            # print(two_char_seq)    
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
    


   
def random_generate_from_LM(num_of_chars,tri_probs):
    
    # z = tri_probs.keys()
    # t = tri_probs.values()
    # r = 5

    # x = list(filter(lambda s: s.startswith('ab'), tri_probs))
    # outed = [tri_probs[i] for i in x]
    # r =5
    valid_char_list = [' ','.','0','#']
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
            # print(two_char_seq)    
            seq = two_char_seq
            num_of_chars = Num_Of_Chars-2

            while (num_of_chars > 0):
            
                if two_char_seq == '.#':
                    output = '.##'
                else:

                    # t1 = time.time()
                    # trigrams = list(filter(lambda s: s.startswith(two_char_seq), tri_probs))
                    trigrams = [two_char_seq + i for i in valid_char_list]
                    # trigrams = [tri_probs[two_char_seq+i] for i in valid_char_list]
                    # print(time.time()-t1)
                    # if not trigrams:
                    #     print(two_char_seq,' Key not found!')
                    #     break    
                    
                    # distribution = [tri_probs[i] for i in trigrams]
                    distribution = [tri_probs[i] for i in trigrams]

                    bins = np.cumsum(distribution)
                    total = np.sum(distribution)
                    # tryt = np.digitize(total*np.random.random_sample(10),bins)
                    # outputx = [trigrams[i] for i in tryt]
                    output = trigrams[np.digitize(total*np.random.random_sample(), bins)]
                   
                seq = seq + output[-1]
                two_char_seq = output[1:3]
            
                # prob = 0
                # trigram_key = ''
                # foundKey = False
                # for key in tri_probs:              
                #     if ((key.startswith(two_char_seq)) and tri_probs[key] > prob):
                #         prob = tri_probs[key]
                #         trigram_key = key
                #         foundKey = True
                # print ('2',' Two character sequence:',two_char_seq,' |Trigram Sequence:',trigram_key, ' |Character Extracted:',trigram_key[2:3])       
            
                num_of_chars -= 1
            num_of_iter += 1  
    # print(seq)                     
    return seq

def main_routine():
    # Parameter selection
    debugger = True
    testing = False
    modelling = True
    alpha = False
    num = False
    line_num = 4
    model_lang = 'en'
    test_file = '../assignment1-data/test'
    model_file = '../assignment1-models/Empirical_Model_en'
    given_model_file = '../assignment1-models/model-br.en'
    selected_data = '../assignment1-data/training.'+ model_lang

    if debugger:
        infile = generate_debug_filepath('training.'+model_lang)
    else:
        infile = terminal_input_filepath()

    if modelling:
        # model_in = init_dummy_model()
        given_model = read_model_from_file(given_model_file)

        sequence = random_generate_from_LM(300,given_model)
        print (sequence)
        # model_in = read_model_from_file(model_file)
        # print ('Perplexity of '+ model_lang + ' file: ' + str(perplexity_computation(selected_data,model_in)))
    else:
        if testing:
            testing_routine(line_num,infile)
            # save_model_to_file(bi_counts,'testline.txt')
        else:
            print('Question 1:')
            print('----------------------')
            complete_model(infile)
            print('Preprocessing complete')
            tri_probs = estimate_probs(bi_counts,tri_counts)
            save_model_to_file(tri_probs,'../assignment1-models/Empirical_Model_' + model_lang)
            print('Question 3:')
            print('----------------------')
            print('All trigams starting with "ng":')
            trigram_with_two_character_history('n','g',tri_probs)
            print('Question 4:')
            print('----------------------')
            given_model = read_model_from_file(given_model_file)
            sequence = generate_from_LM(300,tri_probs)
            print('Our model:')
            print (sequence) #sequence 
            sequence = generate_from_LM(300,given_model)
            print ('Given model:')
            print (sequence)
            print('Question 5:')
            print('----------------------')
            print ('Perplexity of '+ model_lang + ' file: ' +str(perplexity_computation(selected_data,tri_probs)))

        # bigram_viewer(alpha,num,infile)
        # trigram_viewer(alpha,num,infile)

if __name__ == '__main__':
    print ('here')
    print (os.getcwd())
    main_routine()
# do tables and graphs for presentation
# Smoothing

