import re
import sys
import random
import os
from math import log10
from math import pow
from collections import defaultdict
import linecache
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Check for English Characters
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
    line = '#'+line+'#' #adds sentence start/end markers
    return line

def estimate_probs(bi_counts,tri_counts):
    # computes probabilities for each observed trigram
    tri_probs = defaultdict(float) # probabilities of the trigram
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

def terminal_input_filepath():
    if len(sys.argv) != 2:
        print("Usage: ", sys.argv[0], "<training_file>")
        sys.exit(1)
    return sys.argv[1]

def find_bi_tri_grams(corp):
    # generates all bigrams and trigrams from given corpus
    tri_counts=defaultdict(int) #counts of all trigrams 
    bi_counts=defaultdict(int) #counts of all bigrams 

    for j in range(len(corp)-(2)):
        trigram = corp[j:j+3]
        tri_counts[trigram] += 1
        bigram = corp[j:j+2]
        bi_counts[bigram] += 1
      
    final_bigram = corp[len(corp)-2:]
    bi_counts[final_bigram] +=1
    return bi_counts,tri_counts

def testing_routine(line_num,infile):
    # one line test routine
    line = linecache.getline(infile, line_num).rstrip('\n')
    print ('LINE SELECTED:')
    print (line)
    processed_line = preprocess_line(line) # removes special characters and lowercases all letters
    print ('PROCESSED LINE:')
    print (processed_line)

    line = linecache.getline(infile, line_num+1).rstrip('\n')
    print ('LINE SELECTED:')
    print (line)
    processed_line = processed_line + preprocess_line(line) # removes special characters and lowercases all letters
    print ('PROCESSED LINE:')
    print (processed_line)
    
    return find_bi_tri_grams(processed_line)
     
def find_uni_bi_tri_grams(corp):
    # generates all bigrams and trigrams from given corpus
    tri_counts=defaultdict(int) #counts of all trigrams 
    bi_counts=defaultdict(int) #counts of all bigrams 
    uni_counts = defaultdict(int)
    total = 0
    for j in range(len(corp)-(2)):
        trigram = corp[j:j+3]
        tri_counts[trigram] += 1
        bigram = corp[j:j+2]
        bi_counts[bigram] += 1
    final_bigram = corp[len(corp)-2:]
    bi_counts[final_bigram] +=1


    for j in range(len(corp)): # extra code for unigram interpolation
        unigram = corp[j]
        uni_counts[unigram] += 1
        total += 1

    return bi_counts,tri_counts,uni_counts,total

def complete_model_with_uni(infile):
    with open(infile) as f:
        total_input = ''
        for line in f:
                if len(line) > 2:
                    processed_line = preprocess_line(line) 
                    total_input = total_input + processed_line #concatenates corpus together
        return find_uni_bi_tri_grams(total_input)
def complete_model(infile):
    with open(infile) as f:
        total_input = ''
        for line in f:
                if len(line) > 2:
                    processed_line = preprocess_line(line) 
                    total_input = total_input + processed_line #concatenates corpus together
        return find_bi_tri_grams(total_input)
def bigram_viewer(alpha,num,infile,bi_counts):
    if alpha:
        print("Bigram counts in ", infile, ", sorted alphabetically:")
        for bigram in sorted(bi_counts.keys()):
            print(bigram, ": ", bi_counts[bigram])
    if num:
        print("Bigram counts in ", infile, ", sorted numerically:")
        for bi_count in sorted(bi_counts.items(), key=lambda x:x[1], reverse = True):
            print(bi_count[0], ": ", str(bi_count[1]))

def trigram_viewer(alpha,num,infile,tri_counts):
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
        total_text = ''
        for line in f:
            if line == '\n':
                continue
            processed_line = preprocess_line(line)
            total_text = total_text + processed_line
        
        for j in range(len(total_text)-(2)):
            total_tris +=1
            trigram = total_text[j:j+3]
            
            if trigram[1:] != '##' and trigram in model.keys():
                trigram_prob = log10(model[trigram])
                total_prob += trigram_prob 
            else:
                total_tris -=1 #if the trigram ends with '##', this was 
                               # enforced by preprocessing.  Thus, disregard, 
                               # and reduce the tri count to prevent it from 
                               # affecting the final perplexity value.
        # print(total_tris)
        log_perplexity = total_prob*(-1/total_tris)
        perplexity = pow(10,log_perplexity)
    return perplexity

def dummy_perplexity(string,model):
    total_prob = 0
    total_tris = 0
    trigram_prob = 0
    for j in range(len(string)-(2)):
        total_tris +=1
        trigram = string[j:j+3]
        trigram_prob = log10(model[trigram])
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
                    
def generate_from_LM(num_of_chars,tri_probs,valid_char_list,non_sequence_marker_list):
    # function generating a random sequence of characters from a trigram model
    if(num_of_chars == 0):
        return ''
    elif (num_of_chars == 1):
        return (random.choice(valid_char_list))
    elif (num_of_chars == 2):
        return  '#'+ random.choice(valid_char_list)    
    else:
        seq = '' 
        two_char_seq = '#'+ random.choice(valid_char_list)   
        seq = two_char_seq
        num_of_chars -=1 # first '#' is not random, and so the first
                         # sequence only counts as one character.

        while (num_of_chars > 0):
            if two_char_seq[-1] == '#' and two_char_seq[0] != '#':
                output = two_char_seq[0] + '##'
            else:
                if two_char_seq == '##':#was two_char_seq[0] == '#'
                    trigrams = [two_char_seq + i for i in non_sequence_marker_list]                    
                else:
                    trigrams = [two_char_seq + i for i in valid_char_list]                    
                distribution = [tri_probs[i] for i in trigrams]
                bins = np.cumsum(distribution)
                total = np.sum(distribution)
                output = trigrams[np.digitize(total*np.random.random_sample(), bins)]
                num_of_chars -= 1 # character count only subtracted here, since when a
                                  # '##' is generated, the second hash is not random.
            seq = seq + output[-1]
            two_char_seq = output[1:3]
              
    return seq

def de_process(sequence):
    de_processed_line = str()
    eol = False
    for index,i in enumerate(sequence):
        if index == 0:
            continue
        if i == '#':
            if eol:
                eol = False
            else:
                de_processed_line += '\n'
                eol = True
        else:
            de_processed_line += i
    return de_processed_line
        
def valid_char_generator():
    valid_char_list = [' ','.','0','#']
    non_sequence_marker_list = [' ','.','0']
    for i in range(ord('a'),ord('z')+1):
        valid_char_list.append(chr(i))
        non_sequence_marker_list.append(chr(i))
        
    return valid_char_list,non_sequence_marker_list

def create_smoothing_add_one(bi_counts,tri_counts,valid_char_list):
    # Add 1 Smoothing
    print('Applying 1+ smoothing to the values')
    smoothed_model = defaultdict(int)
    all_possible_trigrams = []

    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                if char1=='#' and char2=='#' and char3=='#':
                    continue # can never be 3#s one after the other
                if char2=='#' and char1 != '#':
                    continue # can never be a # stuck between two other characters
                # if (char1 == '#' and char3 == '#'):
                #         continue # cannot have #a#, too short a sentence 
                all_possible_trigrams.append(char1+char2+char3)

    V = len(valid_char_list) 
    for trigram in all_possible_trigrams:
        if trigram[0:2] == '##': #takes care of special V case
            V_act = V-1
        else:
            V_act = V
        smoothed_model[trigram] = (tri_counts[trigram] + 1) / (bi_counts[trigram[0:2]] + V_act)
    print ('1+ smoothing completed')
    return smoothed_model

## Add alpha Smoothing
def create_smoothing_add_alpha(alpha,bi_counts,tri_counts,valid_char_list):
    print('Applying add alpha smoothing to the values')
    smoothed_model = defaultdict(int)
    all_possible_trigrams = []

    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                if char1=='#' and char2=='#' and char3=='#':
                    continue # can never be 3#s one after the other
                if char2=='#' and char1 != '#':
                    continue # can never be a # stuck between two other characters
                # if (char1 == '#' and char3 == '#'):
                #         continue # cannot have #a#, too short a sentence 
                all_possible_trigrams.append(char1+char2+char3)

    V = len(valid_char_list) 
    for trigram in all_possible_trigrams:
        if trigram[0:2] == '##': 
            V_act = V-1 # takes care of special V case
        else:
            V_act = V
        smoothed_model[trigram] = (tri_counts[trigram] + alpha) / (bi_counts[trigram[0:2]] + (alpha * V_act))
    print ('Add alpha smoothing completed')
    return  smoothed_model

## Interpolation Smoothing
def create_smoothing_by_interpolation(bi_counts,tri_counts,valid_char_list,uni_counts,total,l1,l2,l3):
    print('Applying interpolation smoothing to the values')
    smoothed_model = defaultdict(int)
    all_possible_trigrams = []

    for char1 in valid_char_list:
        for char2 in valid_char_list:
            for char3 in valid_char_list:
                if char1=='#' and char2=='#' and char3=='#':
                    continue # can never be 3#s one after the other
                if char2=='#' and char1 != '#':
                    continue # can never be a # stuck between two other characters
                all_possible_trigrams.append(char1+char2+char3)
    V = len(valid_char_list) 
    # V_act = V
    # total_t = 0
    # total_b = 0
    # total_c = 0
    for trigram in all_possible_trigrams:
        trigram_value = 0
        bigram_value = 0
        unigram_value = 0
        # meas = False
        if trigram[0:2] == '##': 
            V_act = V-1# takes care of special V case
            V_act_b = V - (bi_counts['##']+1)
            total_act = total - uni_counts['#']
            meas = True
        else:
            V_act = V
            total_act = total
            V_act_b = V
        # interpolation of smoothed values due to incomplete distributions
        trigram_value = (tri_counts[trigram] + 1) / (bi_counts[trigram[0:2]] + (1 * V_act))  
        bigram_value = (bi_counts[trigram[1:]] + 1) / (uni_counts[trigram[1]] + V_act_b)
        unigram_value = uni_counts[trigram[-1]] / total_act      
        # if meas:
        #     total_t += trigram_value
        #     total_b += bigram_value
        #     total_c += unigram_value
        smoothed_model[trigram] = (l1 * trigram_value) + (l2 * bigram_value) + (l3 * unigram_value)
    # print (total_t)
    # print (total_b)
    # print (total_c)
    print ('Interpolation smoothing completed')
    return smoothed_model

def model_checker(model,valid_char_list):
    
    all_bigrams = []
    for char1 in valid_char_list:
        for char2 in valid_char_list:
            if  char2=='#' and char1 != '#':
                continue
            else:
                all_bigrams.append(char1 + char2)

    for bigram in all_bigrams:
        all_tris = [bigram + i for i in valid_char_list]
        distribution = [model[i] for i in all_tris]
        if sum(distribution) < 0.999:
            print ('WARNING:')
            print (sum(distribution))
            print (bigram)
        # print (sum(distribution))

def main_routine():
    # answers question 1 - 5 completely
    # Parameter selection
    debugger = True # select when filepath is non-dynamic
    testing = False # select when modelling only specific line of an input file
    modelling = False # select when running program only to test perplexity of a test doc
    alphabet = False # select whether or not to display all trigrams/bigrams alphabetically
    num = False # select whether or not to display all trigrams/bigrams numerically
    line_num = 119 # line number for pinpointed model testing
    model_lang = 'en'# english(en), german(de) or spanish(es)
    input_lang = 'en'
    valid_char_list,non_sequence_marker_list = valid_char_generator()# generates all valid characters
    test_given_model = False #decides which model to test
    dummy_modelling = False # choose whether or not to model the dummy example
    alpha = 0.7
    l1 = 0.8
    l2 = 0.1
    l3 = 0.1

    # Input files
    test_file = '../assignment1-data/test' # final test data
    test_es = '../assignment1-data/friends_es/test.txt'
    test_de = '../assignment1-data/test_german/test.txt'
    model_file = '../assignment1-models/Empirical_Model_Smoothed_'+model_lang # generated model
    # model_file = '../assignment1-models/Empirical_Model_Interpolated_en'# generated model
    given_model_file = '../assignment1-models/model-br.en' # given model
    

    if debugger:
        infile =  '../assignment1-data/training.'+input_lang # hard-coded input
    else:
        infile = terminal_input_filepath() # user-provided input
        model_lang = infile[-2:]

    if modelling:
        # fast track to computing perplexity of given document
        if test_given_model:
            given_model = read_model_from_file(given_model_file)
            print ('Perplexity of '+ input_lang + ' file using given model: ' + 
                str(perplexity_computation(infile,given_model)))
        else:
            model_in = read_model_from_file(model_file)
            print ('Perplexity of '+ input_lang + ' file using generated model: ' + 
                str(perplexity_computation(test_es,model_in)))
        
        if dummy_modelling:
            dummy_string = '##abaab#'
            dummy_model = init_dummy_model()
            print ('Perplexity of Dummy input: ' 
                + str(dummy_perplexity(dummy_string,dummy_model)))
    else:
        if testing:
            bi_counts,tri_counts = testing_routine(line_num,infile)
            # save_model_to_file(bi_counts,'testline.txt')
        else:
            print('Preprocessing and modelling (Question 1 & 3)')
            print('----------------------')
            # bi_counts,tri_counts = complete_model(infile) #counts bi/trigrams from input file
            bi_counts,tri_counts,uni_counts,total = complete_model_with_uni(infile) #counts uni/bi/trigrams from input file

            print('Preprocessing complete')
            # tri_probs = estimate_probs(bi_counts,tri_counts) #trigram probabilities
            # print ('Trigram Computations Complete')
            # -------Smoothing Done in this section -------
            smoothed_model = create_smoothing_add_one(bi_counts,tri_counts,valid_char_list)
            smoothed_alpha_model = create_smoothing_add_alpha(alpha,bi_counts,tri_counts,valid_char_list)
            smoothed_inter_model = create_smoothing_by_interpolation(bi_counts,tri_counts,valid_char_list,uni_counts,total,l1,l2,l3)
            # --------Smoothing Section completed -------
            save_model_to_file(smoothed_model,'../assignment1-models/Empirical_Model_Smoothed_' + model_lang)
            save_model_to_file(smoothed_alpha_model,'../assignment1-models/Empirical_Model_Smoothed_'+str(alpha)+'_' + model_lang)
            save_model_to_file(smoothed_inter_model,'../assignment1-models/Empirical_Model_Interpolated_'+ model_lang)

            print ('Model(s) saved to file')
            trigram_with_two_character_history('n','g',smoothed_model)
            print('Question 4:')
            print('----------------------')
            given_model = read_model_from_file(given_model_file)
            print('Random sequence from generated model:')
            print ('--')
            sequence = generate_from_LM(300,smoothed_inter_model,
                        valid_char_list,non_sequence_marker_list)
            sequence = de_process(sequence)
            print (sequence) 
            print ('--')
            print ('Random sequence from given model:')
            print ('--')
            sequence = generate_from_LM(300,given_model,
                        valid_char_list,non_sequence_marker_list)
            sequence = de_process(sequence)
            print (sequence)
            print('Question 5:')
            print('----------------------')
            print ('Perplexity of test file under '+model_lang.upper()+' model: '
                 +str(perplexity_computation(test_file,smoothed_alpha_model)))
            model_checker(smoothed_alpha_model,valid_char_list)
        bigram_viewer(alphabet,num,infile,bi_counts)
        trigram_viewer(alphabet,num,infile,tri_counts)

def batch_modelling():
    # function which quickly reads files and computes perplexities
    debugger = True
    test_given_model = False
    # Input files
    model_lang = 'de'
    model_file = '../assignment1-models/Empirical_Model_Smoothed_'+model_lang # generated model
    given_model_file = '../assignment1-models/model-br.en' # given model

    if debugger:
        infolder =  '../assignment1-data/friends_es/'
    else:
        infolder = terminal_input_filepath() # user-provided input
   
    if test_given_model:
        model_use = read_model_from_file(given_model_file)
    else:
        model_use = read_model_from_file(model_file)

    all_perps = []
    for direc in os.listdir(infolder):
        # infile = infolder + 'episode-' +str(i) +  '.txt'
        infile = infolder + direc
        if os.path.getsize(infile) == 0:
            continue
        perp = perplexity_computation(infile,model_use)
        all_perps.append(perp)
    print (np.mean(all_perps))

def plot_histogram(vals,counts):
    # histogram plotting function
    x_pos = np.arange(len(counts)) 
    barlist = plt.bar(x_pos,counts)
    [barlist[i].set_color('r') for i in range(0,3)]
    [barlist[i].set_color('b') for i in range(3,6)]
    [barlist[i].set_color('g') for i in range(6,9)]
    barlist[9].set_color('k') 

    plt.xticks(x_pos, vals)
    plt.ylim([0,50])
    # plt.show()
    plt.savefig('../assignment1-models/bars.png',dpi = 800)

def random_generate_and_rebuild():
    # flawed function - not to be used
    given_model_file = '../assignment1-models/model-br.en' # given model
    valid_char_list,non_sequence_marker_list = valid_char_generator()# generates all valid characters

    given_model = read_model_from_file(given_model_file)

    infile =  '../assignment1-data/training.en'
    bi_counts,tri_counts = complete_model(infile) #counts bi/trigrams from input file
    s_model = create_smoothing_add_alpha(0.6,bi_counts,tri_counts,valid_char_list)

    sequence = generate_from_LM(4000,given_model,
                valid_char_list,non_sequence_marker_list)
    print('here')
    bi_counts,tri_counts = find_bi_tri_grams(sequence)

    means = []
    alphas = np.arange(0.00001,1.0,0.01)
    for alpha in alphas:
        
        re_done_model = create_smoothing_add_alpha(alpha,bi_counts,tri_counts,valid_char_list)
        
        full_sum = 0
        count = 0
        for k,v in re_done_model.items():
            count += 1
            # print (v)
            # z = (given_model[k])
            full_sum += (v - given_model[k])**2
        mean = full_sum/count
        means.append(mean)
        # print('Mean for alpha of',alpha,':' +str(mean))
    plt.plot(alphas,means)
    plt.xlabel('alphas')
    plt.ylabel('errors')
    plt.show()
    # plt.savefig('../assignment1-models/test.png')
    save_model_to_file(re_done_model,'../assignment1-models/given_model_repeat')

def language_comparison():
    # an attempt at comparing languages
    # Input files
    model_langs = ['en','es','de']
    alphas = ['0.1','0.5','0.8']

    a_labels = [[ j + '_' + i  for j in alphas] for i in model_langs]
    a_labels = [item for sublist in a_labels for item in sublist]
    models = ['../assignment1-models/Empirical_Model_Smoothed_'+i for i in model_langs]
    a_models = ['../assignment1-models/Empirical_Model_Smoothed_'+i for i in a_labels]
    given_model_file = '../assignment1-models/model-br.en' # given model
    a_models.append(given_model_file)
    a_labels.append('given')
    test_file = '../assignment1-data/test' 

    perps = []
    for model in a_models:
        model_use = read_model_from_file(model)
        perp = perplexity_computation(test_file,model_use)
        perps.append(perp)
    plot_histogram(a_labels,perps)

def bulk_perplexity_finder():
    # works out perplexities for Gutenberg dataset
 
    model_file = '../assignment1-models/Empirical_Model_Smoothed_en'

    infolder =  '../assignment1-data/Gutenberg/'
 
    model_use = read_model_from_file(model_file)

    p = re.compile('^[^_]+(?=_)')
    filecounter = 0
    for filepath in os.listdir(infolder):
        filecounter += 1

    with open('../assignment1-data/DataStore', 'w') as file:
        for direc in tqdm(os.listdir(infolder), total=filecounter, unit="files"):
            infile = infolder + direc
            if os.path.getsize(infile) == 0:
                continue
            author = p.match(direc)
            perp = perplexity_computation(infile,model_use)
            file.write(str(perp) + '\t'+ author.group(0) + '\n')

def check_bulk_perplexity():

    lim = 50
    with open('../assignment1-data/DataStore2') as f:
        in_model = defaultdict(int)
        in_totals = defaultdict(int)
        i = 0
        for line in f:
            i +=1
            (perplexity, name) = line.split('\t')
            in_model[name.rstrip('\n')] += float(perplexity)
            in_totals[name.rstrip('\n')] +=1
            if i == lim:
                break
        # print (in_totals['George Alfred Henty'])
            
        # print("Bigram counts in ", infile, ", sorted alphabetically:")
        for author in sorted(in_model.keys()):
            print(author, ": ", in_model[author]/in_totals[author])

def train_singular_model():
    # directly trains just one model 
    valid_char_list,non_sequence_marker_list = valid_char_generator()# generates all valid characters
    p = re.compile('^[^_]+(?=_)')
    infile = 'Ambrose Bierce___Cobwebs From an Empty Skull.txt'
    infile_f = 'episode-7.txt'

    author = p.match(infile)
    bi_counts,tri_counts = complete_model('../assignment1-data/friends/'+infile_f)
    # bi_counts,tri_counts = complete_model('../assignment1-data/Gutenberg/'+infile)

    smoothed_model = create_smoothing_add_one(bi_counts,tri_counts,valid_char_list)
    save_model_to_file(smoothed_model,'../assignment1-models/f_model')

    # save_model_to_file(smoothed_model,'../assignment1-models/' +author.group(0) +  ' model')

def compare_authors():
    # compare author complexities
    valid_char_list,non_sequence_marker_list = valid_char_generator()
    infolder =  '../assignment1-data/Gutenberg/'
    
    p = re.compile('.+?(?= model)')
    infile = 'Ambrose Bierce model'
    author = p.match(infile).group(0)
    author = 'Mark Twain'
    model_loc = '../assignment1-models/'+infile
    # model_loc = '../assignment1-models/model-br.en'
    model_loc = '../assignment1-models/Empirical_Model_Smoothed_es'

    model = read_model_from_file(model_loc)
    
    p2 = re.compile('^[^_]+(?=_)')

    perps = []
    # for direc in os.listdir(infolder):
    #     infile = infolder + direc
    #     if os.path.getsize(infile) == 0:
    #         continue
    #     in_author = p2.match(direc)
    #     # print (in_author.group(0))
    #     if in_author.group(0) ==  author:
    #         perp = perplexity_computation(infile,model)
    #         # print ('Perplexity of ' + in_author.group(0) + ' :' + str(perp))

    infolder =  '../assignment1-data/poetry/'
    
    for direc in os.listdir(infolder):
        infile = infolder + direc
        if os.path.getsize(infile) == 0:
            continue
        perp = perplexity_computation(infile,model)
        print (perp)
        perps.append(perp)
        # print ('Perplexity: ' + str(perp))
    # plt.hist(perps,1000)
    # plt.show()

def shear_providence():
    # an attempt at preprocessing providence dataset
    infolder =  '../assignment1-data/providence/'
    for direc in os.listdir(infolder):
        infile = infolder + direc
        cut_lines = []
        with open(infile) as f:
            for line in f:
                line = line.replace('*CHI', "")
                line = line.replace('*MOT', "")
                line = line.replace('%pho', "")
                line = line.replace('%sit', "")
                line = line.replace('%mod', "")

                cut_lines.append(line)
        with open('../assignment1-data/providence/cut'+direc, 'w') as file:
                for cut_line in cut_lines:
                    file.write(cut_line + '\n')

def alpha_checker():
    # checks out various Add Alpha models and computes perplexities of test documents

    model_langs = ['en','es','de']
    valid_char_list,non_sequence_marker_list = valid_char_generator()# generates all valid characters
    alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    # Input files
    test_file = '../assignment1-data/test' # final test data
    # test_file = '../assignment1-data/friends_es/test.txt'
    test_file = '../assignment1-data/test_german/test.txt'

    predir = '../assignment1-data/training.'
    infiles =  [predir+model_langs[0],predir+model_langs[1], predir+model_langs[2]]
    perplexities = np.zeros((3,9))
    for a_index,alpha in enumerate(alphas):
        for index,filez in enumerate(infiles):
            bi_counts,tri_counts,uni_counts,total = complete_model_with_uni(filez)
            smoothed_alpha_model = create_smoothing_add_alpha(alpha,bi_counts,tri_counts,valid_char_list)
            perplex = perplexity_computation(test_file,smoothed_alpha_model)
            perplexities[index,a_index] = perplex
            print ('Perplexity of test file under '+model_langs[index].upper()+' model with alpha = '+ str(alpha)+': '
                    +str(perplex))


    print (perplexities)
    plt.plot(alphas,perplexities[0,:],label = 'English Model Perplexity')
    plt.plot(alphas,perplexities[1,:],label = 'Spanish Model Perplexity')
    plt.plot(alphas,perplexities[2,:],label = 'German Model Perplexity')
    plt.xlabel('Alpha Value')
    plt.ylabel('German Test Document Perplexity')
    plt.legend()
    plt.grid()
    plt.savefig('../assignment1-models/alpha_changes_German.png',dpi = 500)

def interp_checker():
    # checks out various interpolation models and computes perplexities of test documents
    model_langs = ['en','es','de']
    valid_char_list,non_sequence_marker_list = valid_char_generator()# generates all valid characters
    lambda_in = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

    # Input files
    test_file = '../assignment1-data/test' # final test data
    test_file = '../assignment1-data/friends_es/test.txt'
    # test_file = '../assignment1-data/test_german/test.txt'

    predir = '../assignment1-data/training.'
    infiles =  [predir+model_langs[0],predir+model_langs[1], predir+model_langs[2]]
    i = 0
    for index_m,l1 in enumerate(lambda_in):
        for l2 in lambda_in[0:len(lambda_in)-index_m]:
            i +=1
    perplexities = np.zeros((3,i))
    i = 0
    for index_m,l1 in enumerate(lambda_in):
        for l2 in lambda_in[0:len(lambda_in)-index_m]:
            l3 = 1 - (l1 + l2)
            for index,filez in enumerate(infiles):
                bi_counts,tri_counts,uni_counts,total = complete_model_with_uni(filez)
                smoothed_inter_model = create_smoothing_by_interpolation(bi_counts,tri_counts,valid_char_list,uni_counts,total,l1,l2,l3)
                perplex = perplexity_computation(test_file,smoothed_inter_model)
                perplexities[index,i] = perplex 
            i +=1
            print (l1,l2,l3)

    print (perplexities)
    plt.plot(perplexities[0,:],label = 'English Model Perplexity')
    plt.plot(perplexities[1,:],label = 'Spanish Model Perplexity')
    plt.plot(perplexities[2,:],label = 'German Model Perplexity')
    plt.xlabel('Iteration')
    plt.ylabel('Spanish Test Document Perplexity')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('../assignment1-models/interp_changes_Spanish.png',dpi = 500)

def plot_datastore():
    with open('../assignment1-data/Datastore') as f:
        
        in_model = defaultdict(int)
        in_totals = defaultdict(int)
        model_file = '../assignment1-models/Empirical_Model_Smoothed_en' 
        model_use = read_model_from_file(model_file)

        for line in f:
            (perplexity, name) = line.split('\t')
            in_model[name.rstrip('\n')] += float(perplexity)
            in_totals[name.rstrip('\n')] +=1
        
        means = []
        perps_f = []
        for author in sorted(in_model.keys()):
            means.append(in_model[author]/in_totals[author])

        for direc in os.listdir('../assignment1-data/friends/'):
            infile = '../assignment1-data/friends/' + direc
            if os.path.getsize(infile) == 0:
                continue
            perp = perplexity_computation(infile,model_use)
            perps_f.append(perp)
        plt.plot(means, label = 'Gutenberg Authors')
        plt.plot(perps_f, label = 'Drama Episodes')
        plt.legend()
        plt.xlabel('Various Authors/Drama Episodes')
        plt.ylabel('Document Perplexity')
        plt.savefig('../assignment1-models/authors_vs_friends.png',dpi = 500)

if __name__ == '__main__':

    # main questions handler
    # main_routine()

    # Additional functions providing further analysis or batch-runs (in development)
    plot_datastore()
    # alpha_checker()
    # interp_checker()
    # document_classifier()
    # bulk_perplexity_finder()
    # check_bulk_perplexity()
    # batch_modelling()
    # language_comparison()
    # train_singular_model()
    # shear_providence()
    # compare_authors()
    # random_generate_and_rebuild()

