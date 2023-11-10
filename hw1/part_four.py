import random
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import part_one

testFile = "./data/test.csv"
trainFile = "./data/train.csv"

# PART 4 word2vec

def average_log_likelihood(word_list, model):
    log_likelihoods = []
    for word in word_list:
        emb = np.abs(model.wv[word])
        # print(np.log(emb))
        # try:
        #     word_vector = model.wv[word]
        #     print(word)
        #     log_likelihood = np.mean(np.log(word_vector))
        #     log_likelihoods.append(log_likelihood)
        # except:
        #     # Handle words not in the vocabulary
        #     pass
        log_likelihoods.append(np.log(emb))

    if log_likelihoods:
        # Calculate the average log likelihood
        avg_log_likelihood = np.mean(log_likelihoods)
        return avg_log_likelihood
    return None

def sentence_vec(sentence, model):
    words = sentence
    vector = np.copy(model.wv[words[0]])
    for word in words[1:]:
        vector += model.wv[word]
    vector /= len(words)
    return vector

def simVocab(vocabulary, collection, model):
    valMap = {}
    for v in vocabulary:
        emb = model.wv[v]
        total = 0
        for d in collection:
            sv = sentence_vec(d, model)
            sim = model.wv.cosine_similarities(emb, np.array([sv]))[0]
            total += sim
        valMap[v] = total
    l = list(valMap)
    l.sort(key = lambda x : x[1])
    return l

def simQuery(q, collection, model):
    docList = []
    qVec = sentence_vec(q, model)
    for i in range(len(collection)):
        d= collection[i]
        sv = sentence_vec(d, model)
        sim = model.wv.cosine_similarities(qVec, np.array([sv]))[0]
        docList.append([" ".join(collection[i]), sim])

    docList.sort(key = lambda x : x[1])
    return docList

def w2v(q, vocabulary, descriptions, testType):
    tokenized_docs = []
    for d in descriptions:
        tokenized_docs.append(d.split(" "))
    model = gensim.models.Word2Vec(tokenized_docs, min_count = 1, vector_size= 100, window=5, sg=1)
    
    if (testType == "train"):
        l = simVocab(vocabulary, tokenized_docs, model)
        print(l[-10:])
    else:
        vals = simQuery(q.split(" "), tokenized_docs, model)
        # top 5
        print(vals[-5:])
        # lowest 5
        print(vals[:5])


vocabulary, clean_descriptions = part_one.parseData(trainFile)
# get top ten
vals = w2v("", vocabulary, clean_descriptions, "train")

vocabulary, clean_descriptions = part_one.parseData(testFile)
vals = w2v("olympic gold athens", vocabulary, clean_descriptions, "test")
vals = w2v("reuters stocks friday", vocabulary, clean_descriptions, "test")
vals = w2v("investment market prices", vocabulary, clean_descriptions, "test")

randWords = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randString = " ".join(randWords)
vals = w2v(randString, vocabulary, clean_descriptions, "test")
