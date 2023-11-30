import random
from build_vocab import parseData
from part_four import probability_retrieval
from part_three import tf_idf

output_file = "./data/corpus.txt"
vocabulary, wordMap, M = parseData(output_file)
randWords1 = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
randWords2 = [random.choice(vocabulary), random.choice(vocabulary), random.choice(vocabulary)]
vals = tf_idf(" ".join(randWords1), vocabulary, wordMap, M)
vals = tf_idf(" ".join(randWords2), vocabulary, wordMap, M)

smoothing_param = .1
cap = 10
vals = probability_retrieval(" ".join(randWords1), vocabulary, wordMap, M, smoothing_param, cap)
vals = probability_retrieval(" ".join(randWords2), vocabulary, wordMap, M, smoothing_param, cap)
