import random
import numpy as np
import build_vocab

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

def jelinek_mercer_smoothing(tf, doc_length, cf, total_words, smoothing_param):
    return (1 - smoothing_param) * (tf / doc_length) + smoothing_param * (cf / total_words)

def probability_retrieval(query, vocabulary, wordMap, M, smoothing_param):
    scores = []

    for docid in range(5000):
        matched_words = shared_words(query, str(docid), wordMap)
        score = 0
        for w in matched_words:
            tf = get_freq(w, query)
            doc_length = len(wordMap.get(w, []))
            cf = sum(pair[1] for pair in wordMap.get(w, []))
            total_words = sum(pair[1] for sublist in wordMap.values() for pair in sublist)

            score += np.log(jelinek_mercer_smoothing(tf, doc_length, cf, total_words, smoothing_param))
        scores.append([docid, score])


    scores.sort(key=lambda x: x[1])
    # top 5
    print(scores[-5:])
    # lowest 5
    print(scores[:5])


output_file = "./data/part-00000.txt"
vocabulary, wordMap, M = build_vocab.parseData(output_file)


smoothing_param = 0.1

vocabulary, wordMap, M = build_vocab.parseData(output_file)
vals = probability_retrieval("olympic gold athens", vocabulary, wordMap, M, smoothing_param)
vals = probability_retrieval("reuters stocks friday", vocabulary, wordMap, M, smoothing_param)
vals = probability_retrieval("investment market prices", vocabulary, wordMap, M, smoothing_param)

randWords = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randString = " ".join(randWords)
vals = probability_retrieval(randString, vocabulary, wordMap, M, smoothing_param)
