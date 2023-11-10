import random
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import build_vocab

output_file = "./data/part-00000.txt"

def get_freq(string, content):
    split = content.split(" ")
    res = 0
    for i in range(len(split)):
        if split[i] == string:
            res += 1
    return res

def get_doc_count(w, docid, wordMap):
    for pair in wordMap[w]:
        if pair[0] == docid:
            return pair[1]
    return 0

def shared_words(q, docid, wordMap):
    q_s = q.split(" ")
    res = []
    
    for w in q_s:
        
        if w not in wordMap:
            continue
        for pair in wordMap[w]:
            if pair[0]== docid:
                res.append(w)
                continue
    return res

def tf_idf(q, vocabulary, wordMap, M):
    # TF-IDF necessary variables
    k = 1.4
    vals = []

    # precompute document freq (how many docs does w appear in)
    q_s = q.split(" ")
    doc_freq = {}
    for w in q_s: 
        doc_freq[w] = len(wordMap[w])
        if doc_freq == 0: doc_freq += 1

    for docid in range(M):
        matched_words = shared_words(q, str(docid), wordMap)
        score = 0
        
        for w in matched_words: 
            score += get_freq(w, q) * (((k + 1) * get_doc_count(w, str(docid), wordMap)) / (get_doc_count(w, str(docid), wordMap) + k)) * np.log((M + 1) / doc_freq[w])
        
        vals.append([docid, score])

    # print(vals)
    vals.sort(key=lambda x : x[1])
    # top 5
    print(vals[-5:])
    # lowest 5
    print(vals[:5])


vocabulary, wordMap, M = build_vocab.parseData(output_file)
vals = tf_idf("olympic gold athens", vocabulary, wordMap, M)
vals = tf_idf("reuters stocks friday", vocabulary, wordMap, M)
vals = tf_idf("investment market prices", vocabulary, wordMap, M)

randWords = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randString = " ".join(randWords)
vals = tf_idf(randString, vocabulary, wordMap, M)
