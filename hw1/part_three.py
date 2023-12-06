import random
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import hw3.part_one as part_one

testFile = "./data/test.csv"
trainFile = "./data/train.csv"

# PART 2 Basic Bit Vector

def get_freq(string, content):
    split = content.split(" ")
    res = 0
    for i in range(len(split)):
        if split[i] == string:
            res += 1
    return res

def shared_words(q, content):
    q_s = q.split(" ")
    c_s = content.split(" ")
    return set(q_s).intersection(set(c_s))

def tf_idf(q, vocabulary, descriptions, testType):
    # TF-IDF necessary variables
    k = 1.4
    M = len(descriptions)
    vals = []

    # precompute document freq
    q_s = q.split(" ")
    doc_freq = {}
    for w in q_s: 
        for d in descriptions:
            d_s = d.split(" ")
            if w in d_s:
                doc_freq[w] = doc_freq.get(w, 0) + 1

    for i in range(len(descriptions)):
        doc = descriptions[i]
        matched_words = shared_words(q, doc)
        f = 0
        
        for w in matched_words:
            f += get_freq(w, q) * (((k + 1) * get_freq(w, doc)) / (get_freq(w, doc) + k)) * np.log((M + 1) / doc_freq[w])
        
        vals.append([i, f])

    if (testType == "train"):
        print(vals)
    else:
        vals.sort(key=lambda x : x[1])
        # top 5
        print(vals[-5:])
        # lowest 5
        print(vals[:5])


vocabulary, clean_descriptions = part_one.parseData(trainFile)
vocabTopTen = vocabulary[:10]
vals = tf_idf("olympic gold athens", vocabTopTen, clean_descriptions, "train")
vals = tf_idf("reuters stocks friday", vocabTopTen, clean_descriptions, "train")
vals = tf_idf("investment market prices", vocabTopTen, clean_descriptions, "train")

vocabulary, clean_descriptions = part_one.parseData(testFile)
vals = tf_idf("olympic gold athens", vocabulary, clean_descriptions, "test")
vals = tf_idf("reuters stocks friday", vocabulary, clean_descriptions, "test")
vals = tf_idf("investment market prices", vocabulary, clean_descriptions, "test")

randWords = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randString = " ".join(randWords)
vals = tf_idf(randString, vocabulary, clean_descriptions, "test")
