import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords

testFile = "./data/test.csv"
trainFile = "./data/train.csv"

# PART 1 Build Vocabulary

def parseData(fileName):
    df = pd.read_csv(fileName)
    descriptions = df.iloc[:, 1].head(12000)
    nltk.download('punkt')
    sw = stopwords.words("English")
    
    wordFreq = {}

    for i in range(len(descriptions)):
        word = descriptions[i]
        tokens = nltk.word_tokenize(word)
        cleaned_tokens = [word.lower() for word in tokens if word.isalpha() and word not in sw]

        for w in cleaned_tokens:
            wordFreq[w] = wordFreq.get(w, 0) + 1
    
    sorted_vocab = dict(sorted(wordFreq.items(), key=lambda item: item[1], reverse=True))
    vocabulary = list(sorted_vocab)[:200]
    return vocabulary

# PART 2 Basic Bit Vector 

def basic_bit_vector(q, vocabulary):
    # create a tuple
    print(vocabulary)
    point = [0] * 200
    for i in range(len(vocabulary)):
        if vocabulary[i] in q:
            print(vocabulary[i])
            point[i] = 1
    
    vector = np.array(point)



    
if __name__ == "__main__":
    vocabulary = parseData(trainFile)
    basic_bit_vector("olympic gold athens", vocabulary)
