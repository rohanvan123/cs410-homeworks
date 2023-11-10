
import ast
from collections import defaultdict


output_file = "./data/part-00000.txt"

# Build Vocabulary

def parseData(fileName):

    wordFreq = {}
    wordMap = defaultdict(list)
    docsSeen = set()

    with open(fileName, 'r') as f:
      for line in f:
        words = line.split("\t")
        word, docs = words
        docs = docs.strip()

        # Use ast.literal_eval to safely evaluate the string as a Python literal
        docs = ast.literal_eval(docs)
        wordMap[word] = docs
        for d in docs:
           wordFreq[word] = wordFreq.get(word, 0) + d[1]
           docsSeen.add(d[0])
    
    wordList = list(wordFreq.items())
    wordList.sort(key = lambda x : x[1])
    wordList = wordList[::-1]
    vocabulary = [wordList[i][0] for i in range(len(wordList[:200]))]

    return vocabulary, wordMap, len(docsSeen)

