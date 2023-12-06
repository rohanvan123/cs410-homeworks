# Question 2
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import logging
import tensorflow as tf
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
testFile = "./data/test.csv"
trainFile = "./data/train.csv"

def parseData(fileName):
    df = pd.read_csv(fileName)
    descriptions = df.iloc[:, 1].head(15000)
    # nltk.download('punkt')
    # sw = stopwords.words("English")
    clean_descriptions = []
    wordFreq = {}

    for i in range(len(descriptions)):
        word = descriptions[i]
        tokens = nltk.word_tokenize(word)
        cleaned_tokens = [word.lower() for word in tokens if word.isalpha()]
        clean_descriptions.append(" ".join(cleaned_tokens))
        for w in cleaned_tokens:
            wordFreq[w] = wordFreq.get(w, 0) + 1
    
    sorted_vocab = dict(sorted(wordFreq.items(), key=lambda item: item[1], reverse=True))
    vocabulary = list(sorted_vocab)[:200]
    return vocabulary, clean_descriptions

def part_two(tokenized_docs, fileName):
    # create dictionary using docs
    test_data = pd.read_csv(fileName)
    documents = test_data.iloc[:, 2].values
    rel_docs = [list(gensim.utils.simple_preprocess(doc)) for doc in documents]

    D1 = corpora.Dictionary(rel_docs)
    doc_matrix = [D1.doc2bow(doc) for doc in rel_docs]

    # Create the object for LDA model
    lda1 = LdaModel

    # Train the LDA model using the document term matrix .
    ldamodel = lda1 ( doc_matrix , num_topics =10 , id2word = D1 , passes =100 )

    # Display the 5 most relevant terms for one of the topics
    five_most_relevant = ldamodel.show_topic(0, 5)  # Change the topic number as needed

    # Calculate the topic differences
    topic_diff_mat = ldamodel.diff(ldamodel, distance='kullback_leibler', num_words=50)

    return five_most_relevant, topic_diff_mat

vocabulary, clean_descriptions = parseData(testFile)
five_most_relevant, topic_diff_mat = part_two(clean_descriptions, testFile)
print(five_most_relevant)
print(topic_diff_mat)

