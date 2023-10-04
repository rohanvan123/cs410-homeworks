import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords

testFile = "./data/test.csv"
trainFile = "./data/train.csv"

# PART 1 Build Vocabulary

def parseData(fileName):
    df = pd.read_csv(fileName)
    descriptions = df.iloc[:, 1].head(15000)
    # nltk.download('punkt')
    # sw = stopwords.words("English")
    with open('stopwords.txt', 'r') as f:
        sw = f.read()
    clean_descriptions = []
    wordFreq = {}

    for i in range(len(descriptions)):
        word = descriptions[i]
        tokens = nltk.word_tokenize(word)
        cleaned_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in sw]
        clean_descriptions.append(" ".join(cleaned_tokens))
        for w in cleaned_tokens:
            wordFreq[w] = wordFreq.get(w, 0) + 1
    
    sorted_vocab = dict(sorted(wordFreq.items(), key=lambda item: item[1], reverse=True))
    vocabulary = list(sorted_vocab)[:200]
    return vocabulary, clean_descriptions

vocabulary, clean_descriptions = parseData(trainFile)
# print(vocabulary)

    
