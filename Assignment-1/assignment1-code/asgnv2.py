import re
import sys
import random
# from random import random
from math import log10
from math import pow
from collections import defaultdict
import linecache
import itertools

tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all bigrams in input
uni_counts=defaultdict(int) #counts of all bigrams in input
total_count = defaultdict(int)
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
    for j in range(len(line)):
        unigram = line[j]
        uni_counts[unigram] += 1
        total_count['total'] += 1

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
                # else:
                #     print('unseen trigram: ' + trigram)
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


def generate_from_LM(num_of_chars,tri_probs,generate_from_smoothed_model,trigram_distribution):
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
        Num_Of_Chars = num_of_chars
        seq = ''
        if (generate_from_smoothed_model == False):             
            num_of_iter = 0
            while(len(seq) != Num_Of_Chars and num_of_iter <= 1000):  
                two_char_seq = (random.choice(valid_char_list))+(random.choice(valid_char_list))
                # two_char_seq = 'bt' #'gq'
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
                        # print(two_char_seq,' Key not found!')
                        break    
                    num_of_chars -= 1                    
                num_of_iter += 1
        else:
            two_char_seq = (random.choice(valid_char_list))+(random.choice(valid_char_list))
            # two_char_seq = 'bt'#'gq'
            seq = two_char_seq
            num_of_chars = Num_Of_Chars-2
            while (num_of_chars > 0):
                prob = 0
                trigram_key = ''
                # print ('1',' Two character sequence:',two_char_seq,' |Trigram Sequence:',trigram_key) 
                for key in trigram_distribution:              
                    if ((key.startswith(two_char_seq)) and trigram_distribution[key] > prob):
                        prob = trigram_distribution[key]
                        trigram_key = key
                # print ('2',' Two character sequence:',two_char_seq,' |Trigram Sequence:',trigram_key, ' |Character Extracted:',trigram_key[2:3])       
                seq = seq + trigram_key[2:3]
                two_char_seq = trigram_key[1:3]  
                num_of_chars -= 1         
    return seq

## Add 1 Smoothing
def create_smoothing_add_one():
    print('Applying 1+ smoothing to the values')
    smoothed_model = defaultdict(int)
    valid_char_list = [' ','.','0']
    for i in range(ord('a'),ord('z')+1):
        valid_char_list.append(chr(i))
    all_possible_trigrams = []
    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                all_possible_trigrams.append(char1+char2+char3)
    V = len(valid_char_list)
    for trigram in all_possible_trigrams:
        smoothed_model[trigram] = (tri_counts[trigram] + 1) / (bi_counts[trigram[0:2]] + V)
    save_model_to_file(smoothed_model , '../assignment1-models/Empirical_Model_Smoothed_en')
    print ('1+ smoothing completed')
    return

## Add alpha Smoothing
def create_smoothing_add_alpha(alpha):
    print('Applying add alpha smoothing to the values')
    smoothed_model = defaultdict(int)
    valid_char_list = [' ','.','0']
    for i in range(ord('a'),ord('z')+1):
        valid_char_list.append(chr(i))
    all_possible_trigrams = []
    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                all_possible_trigrams.append(char1+char2+char3)
    V = len(valid_char_list)
    for trigram in all_possible_trigrams:
        smoothed_model[trigram] = (tri_counts[trigram] + alpha) / (bi_counts[trigram[0:2]] + (alpha * V))
    save_model_to_file(smoothed_model , '../assignment1-models/Empirical_Model_Smoothed_en')
    print ('Add alpha smoothing completed')
    return   

## Interpolation Smoothing
def create_smoothing_by_interpolation():
    print('Applying interpolation smoothing to the values')
    smoothed_model = defaultdict(int)
    valid_char_list = [' ','.','0']
    for i in range(ord('a'),ord('z')+1):
        valid_char_list.append(chr(i))
    all_possible_trigrams = []
    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                all_possible_trigrams.append(char1+char2+char3)
    for trigram in all_possible_trigrams:
        trigram_value = 0
        bigram_value = 0
        unigram_value = 0
        if(bi_counts[trigram[0:2]] > 0):
            trigram_value = (tri_counts[trigram]) / (bi_counts[trigram[0:2]])
        if(uni_counts[trigram[0]] > 0):
            bigram_value = (bi_counts[trigram[0:2]]) / (uni_counts[trigram[0]])
        if(total_count['total'] > 0):
            unigram_value = uni_counts[trigram[0]] / total_count['total']      
        smoothed_model[trigram] = (0.6 * trigram_value) + (0.3 * bigram_value) + (0.1 * unigram_value)
    save_model_to_file(smoothed_model , '../assignment1-models/Empirical_Model_Smoothed_en')
    print ('Interpolation smoothing completed')
    return 


def main_routine():
    # Parameter selection
    debugger = False
    testing = False
    modelling = False
    alpha = False
    num = False
    line_num = 4
    model_lang = 'en'
    test_file = '../assignment1-data/test'
    # model_file = '../assignment1-models/Empirical_Model_en'
    given_model_file = '../assignment1-models/model-br.en'
    model_file = '../assignment1-models/Empirical_Model_Smoothed_en'
    
    if debugger:
        infile = generate_debug_filepath('training.'+model_lang)
    else:
        infile = terminal_input_filepath()

    if modelling:
        # model_in = init_dummy_model()        
        model_in = read_model_from_file(model_file)
        print ('Perplexity: ' + str(perplexity_computation('../assignment1-data/training.es',model_in)))
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
            # -------Smoothing Done in this section -------
            create_smoothing_add_one()
            # create_smoothing_add_alpha(0.8)
            # create_smoothing_by_interpolation()
            # --------Smoothing Section completed -------
            print('Question 3:')
            print('----------------------')
            print('All trigams starting with "ng":')
            trigram_with_two_character_history('n','g',tri_probs)
            print('Question 4:')
            print('----------------------')
            generate_from_smoothed_model = False # True: use the smoothed model; False: use the model without smoothing 
            smoothed_model = dict()
            given_model = read_model_from_file(given_model_file)
            if generate_from_smoothed_model == True:
                smoothed_model = read_model_from_file('../assignment1-models/Empirical_Model_Smoothed_en')
            sequence = generate_from_LM(300,tri_probs,generate_from_smoothed_model,smoothed_model)
            print('Our model:')
            print('Seed sequence used: ',sequence[0:2])
            print (sequence) #sequence 
            sequence = generate_from_LM(300,given_model,False,dict())
            print ('Given model:')
            print('Seed sequence used: ',sequence[0:2])
            print (sequence) #sequence 
            print('Question 5:')
            print('----------------------')
            print ('Perplexity of file: ' +str(perplexity_computation(test_file,tri_probs)))

        # bigram_viewer(alpha,num,infile)
        # trigram_viewer(alpha,num,infile)

if __name__ == '__main__':
    main_routine()
# do tables and graphs for presentation
# Smoothing

