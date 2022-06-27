#Tomas Vera HW4 CS190I

import sys
import sklearn_crfsuite
import shutil
import re

from sklearn_crfsuite import metrics
import time


def process_text_file(filename):
    data = []
    file = open(filename, 'r')
    read_lines = file.readlines()

    curr_sentence =[]
    for line in read_lines: 
        word_info = line.split(' ')
        word_info[-1] = word_info[-1].strip()
        word_tuple = tuple(word_info)
        
        if word_info[0] == '':
            data.append(curr_sentence)
            curr_sentence = []
        else:
            curr_sentence.append(word_tuple)
    data.append(curr_sentence)
    return data

def convert_to_txt(data, infile, outfile):
    shutil.copy(infile, outfile)
    flat_data = []
    for sentence in data:
        for tag in sentence:
            flat_data.append(tag)
        flat_data.append('')
    
    with open(outfile, 'r') as f:
        file_lines = [''.join([x.strip(), " " + flat_data[i], '\n']) for i, x in enumerate(f.readlines())]
    
    with open(outfile, 'w') as f:
        f.writelines(file_lines) 

def process_sentence(sentence, i):
    word = sentence[i][0]
    features = {
        'bias': 1.0,
        'word': word,
        'lowercase': word.lower(),
        'prefix': word[:3],
        'suffix': word[-3:],
        'acronym': word.isupper(),
        'first_upper': word.istitle(),
        'digit': word.isdigit(),
        'has_digit': bool(re.findall('[0-9]+', word)),
    }
    if i > 0:
        word1 = sentence[i-1][0]
        features.update({
            '-1:lowercase': word1.lower(),
            '-1:acronym': word1.istitle(),
            '-1:first_upper': word1.isupper(),
            '-1:has_digit': bool(re.findall('[0-9]+', word1)),
        })
    else:
        features['BOS'] = True

    if i < len(sentence)-1:
        word1 = sentence[i+1][0]
        features.update({
            '+1:lowercase': word1.lower(),
            '+1:acronym': word1.istitle(),
            '+1:first_upper': word1.isupper(),
            '+1:has_digit': bool(re.findall('[0-9]+', word1)),
        })
    else:
        features['EOS'] = True

    return features

def get_features(sentence):
    return [process_sentence(sentence, i) for i in range(len(sentence))]

def get_labels(sentence):
    return [label for token, label in sentence]

def get_tokens(sentence):
    return [token for token, label in sentence]


#Load command line arguments

test_file_path = sys.argv[1]

prediction_file_path = sys.argv[2]

#Read datasets

#reading_time = time.time()

train_data = process_text_file('train_utf8.txt')

test_data = process_text_file(test_file_path)

#print("reading dataset time: %s seconds" % (time.time() - reading_time))


#Training model

#training_time = time.time()
    
X_train = [get_features(s) for s in train_data]
y_train = [get_labels(s) for s in train_data]

X_test = [get_features(s) for s in test_data]
#y_test = [get_labels(s) for s in test_data]

crf = sklearn_crfsuite.CRF(
    algorithm='l2sgd',
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=False,
)

crf.fit(X_train, y_train)

#print("training time: %s seconds" % (time.time() - training_time))


#Prediction accuracy metric
""" labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
 """

#Prediction using model

#testing_time = time.time()
y_pred = crf.predict(X_test)
#print("testing time: %s seconds" % (time.time() - testing_time))

#Write to output file

#writing_time = time.time()

convert_to_txt(y_pred, test_file_path, prediction_file_path)

#print("writing prediction time: %s seconds" % (time.time() - writing_time))