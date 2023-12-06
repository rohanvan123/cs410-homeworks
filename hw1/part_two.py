import random
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import hw3.part_one as part_one

testFile = "./data/test.csv"
trainFile = "./data/train.csv"

# PART 2 Basic Bit Vector

def build_vector(string, vocabulary):
    split = string.split(" ")
    point = [0] * len(vocabulary)
    for i in range(len(vocabulary)):
        if vocabulary[i] in split:
            point[i] = 1
    return np.array(point)

def basic_bit_vector(q, vocabulary, descriptions, testType):
    # create a tuple
    # print(vocabulary)
    qv = build_vector(q, vocabulary)

    vals = []
    for i in range(len(descriptions)):
        d = descriptions[i]
        dv = build_vector(d, vocabulary)
        vals.append((i, np.dot(qv, dv))) # (doc_id, rank)

    if (testType == "train"):
        print(vals)
    else:
        vals.sort(key=lambda x : x[1])
        # top 5
        print(vals[-5:])
        # lowest 5
        print(vals[:5])


vocabulary, clean_descriptions = part_one.parseData(trainFile)
# use top 10 vocab words
# for this I print out the docId
vocabTopTen = vocabulary[:10]
vals = basic_bit_vector("olympic gold athens", vocabTopTen, clean_descriptions, "train")
vals = basic_bit_vector("reuters stocks friday", vocabTopTen, clean_descriptions, "train")
vals = basic_bit_vector("investment market prices", vocabTopTen, clean_descriptions, "train")

vocabulary, clean_descriptions = part_one.parseData(testFile)
vals = basic_bit_vector("olympic gold athens", vocabulary, clean_descriptions, "test")
vals = basic_bit_vector("reuters stocks friday", vocabulary, clean_descriptions, "test")
vals = basic_bit_vector("investment market prices", vocabulary, clean_descriptions, "test")

# randomly generated query
randWords = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randString = " ".join(randWords)
vals = basic_bit_vector(randString, vocabulary, clean_descriptions, "test")
    