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
import logging 

tf.get_logger().setLevel(logging.ERROR)
testFile = "./data/test.csv"
trainFile = "./data/train.csv"
n_steps_in, n_steps_out = 3, 2

# helper functions

def root_mean_squared_error(y_t, y_p):
    return np.sqrt(mean_squared_error(y_t, y_p))

def plot_combined_histogram(rmse_tanh, rmse_relu, include_stopwords):
    plt.hist(rmse_tanh, color='green', alpha = .5, label='tanh')
    plt.hist(rmse_relu, color='purple', alpha = .5, label='relu')
    plt.xlabel('Root Mean Squared Error')
    plt.ylabel('Frequency')
    plt.title('RMSE of different activiation functions')
    
    plt.legend()

    # plt.show()
    if (include_stopwords):
        plt.savefig('./images/output_plot_sw.png')
    else:
        plt.savefig('./images/output_plot_nsw.png')

# 1, 2 build vocab and get data

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

#####################################################

def preprocess_documents_with_sw(clean_documents):
    processed_docs = []
    with open("stopwords.txt", 'r') as file:
        stop_words = [line.strip() for line in file]

        for doc in clean_documents:
            split_doc = doc.split(" ")
            clean_doc = [word for word in split_doc if word not in stop_words]

            # Joining the tokens back into a string
            processed_docs.append(" ".join(clean_doc))

        # tokenized_documents = [word_tokenize(doc.lower()) for doc in processed_docs]
    return processed_docs


def part3(clean_documents, include_stopwords):
    if (not include_stopwords):
        clean_documents = preprocess_documents_with_sw(clean_documents)

    tokenized_docs = [word_tokenize(doc.lower()) for doc in clean_documents]

    word2vec_model = Word2Vec(tokenized_docs, vector_size=100, window=10, min_count=1, workers=4)

    # convert documents into sequence vectors
    
    X, y = [], []
    for doc in clean_documents:
        tokens = doc.split(" ")
        vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        for i in range(len(vectors) - n_steps_in - n_steps_out + 1):
            seq_x = vectors[i:i + n_steps_in]
            seq_y = vectors[i + n_steps_in:i + n_steps_in + n_steps_out]
            X.append(seq_x)
            y.append(seq_y)
    

    # output the model and vectors
    return word2vec_model, np.array(X), np.array(y), tokenized_docs



def build_model(X, y):

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2)

    # train an lstm model, this is adapted from provided code
    def lstm_model(activation_function, X_test, y_test):
        # define model
        model = Sequential()
        model.add(LSTM(200, activation=activation_function, input_shape=(n_steps_in, 100)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(200, activation=activation_function, return_sequences=True))
        model.add(TimeDistributed(Dense(100)))
        model.compile(optimizer='adam', loss='mse')
        # Fit model
        model.fit(X, y, epochs=1, verbose=0)
        # Making predictions
        y_pred = model.predict(X_test, verbose=0)

        return model, [root_mean_squared_error(y_test[i], y_pred[i]) for i in range(len(y_test))]
    
    # get the different models
    model_tanh, tanh_rmse_data = lstm_model('tanh', X_test, y_test)
    model_relu, relu_rmse_data = lstm_model('relu', X_test, y_test)

    model, activation_func = None, None
    # get the model with the one of the smaller error mean
    if np.mean(tanh_rmse_data) < np.mean(relu_rmse_data):
        model = model_tanh
        activation_func = "tanh"
    else:
        model = model_relu
        activation_func = "relu"

    return model, activation_func, tanh_rmse_data, relu_rmse_data


# 3.4 pick longest doc and predict the last two words
def predict_last_two_words(tokenized_docs, w2v_model, best_model):
    # get largest doc
    longest_doc = []
    for doc in tokenized_docs:
        if len(doc) > len(longest_doc):
            longest_doc = doc
    print("longest doc: ", longest_doc)

    # prediction portion
    in_seq = longest_doc[-(n_steps_in + 2):-2]

    input_vectors = np.array([w2v_model.wv[word] for word in in_seq if word in w2v_model.wv])
    input_vectors = input_vectors.reshape((1, n_steps_in, 100))

    yhat = best_model.predict(input_vectors, verbose=0)
    last_two_words = [w2v_model.wv.most_similar([yhat[0][0]], topn=1)[0][0], w2v_model.wv.most_similar([yhat[0][0]], topn=1)[0][0]]
    return last_two_words

# 3.5 find similar words
def determine_related_words(tokenized_docs, w2v_model, best_model):
    tok_tr_docs, tok_test_docs = train_test_split(tokenized_docs, test_size=.2, random_state=42)
    test_X, actual= [], []

    for doc in tok_test_docs:
        vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv]
        for i in range(len(vectors) - n_steps_in):
            test_X.append(vectors[i:i + n_steps_in])
            actual.append(doc[i + n_steps_in])

    test_X =np.array(test_X)

    test_X = test_X.reshape((test_X.shape[0], n_steps_in, 100))

    # Make predictions
    pred_vects = best_model.predict(test_X, verbose=0)
    if len(pred_vects.shape) == 3:
        pred_vects = pred_vects.reshape(-1, 100)

    #finding predicted words again
    pred_words = []
    for i in range(0, len(pred_vects), 2):
        for vec in pred_vects[i:i+2]:
            w = w2v_model.wv.most_similar([vec], topn=1)[0][0]
            pred_words.append(w)
    
    return actual, pred_words

def get_f1_score(actual_words, predicted_words):
    true_pos = 0
    for i in range(len(min(actual_words))):
        if (actual_words[i] == predicted_words[i]):
            true_pos += 1

    false_neg = len(predicted_words) - true_pos
    false_pos = len(actual_words) - true_pos

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    if (precision + recall == 0):
        return precision, recall, 0
    
    f1_score = (2 * (precision * recall)) / (precision + recall)

    return precision, recall, f1_score


##################################

vocabulary, clean_descriptions = parseData(testFile)

def part_one(include_stopwords):
    if (include_stopwords):
        print("Including stop words: ")
    else:
        print("Not including stop words: ")
    w2v_model, X, y, tokenized_docs = part3(clean_descriptions, include_stopwords)
    best_model, activation_func, tanh_rmse_data, relu_rmse_data = build_model(X, y)
    plot_combined_histogram(tanh_rmse_data, relu_rmse_data, include_stopwords)
    print(f"best activation function is {activation_func}")
    pred_words = predict_last_two_words(tokenized_docs, w2v_model, best_model)
    print(f"Two predicted predicted_words from longest doc are {pred_words}")
    actual_words, predicted_words = determine_related_words(tokenized_docs, w2v_model, best_model)
    print(f"Actual: {actual_words[:20]}")
    print(f"Predicted: {predicted_words[:20]}")
    precision, recall, f1_score = get_f1_score(actual_words, predicted_words)
    print(f"Our ouput has an f1 score of {f1_score}, with precision of {precision} and recall of {recall}")
    print("##################################")

part_one(include_stopwords=False)
