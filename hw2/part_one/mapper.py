# Code for mapper.py to be used on colab 

#Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
import sys
import io
import re
import nltk
import pandas as pd  

nltk.download('stopwords',quiet=True)
from nltk.corpus import stopwords
punctuations = '''0123456789!()-[]{};:'"\,<>./?@#$%^&*_~'''

stop_words = set(stopwords.words('english'))
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')

docid = 1 #actually line id
for line in input_stream:
  line = line.split(',', 3)[2]
  line = line.strip()
  line = re.sub(r'[^\w\s]', '',line)
  line = line.lower()
  line = re.sub(r'\s+', ' ', line)
  for x in line:
    if x in punctuations:
      line=line.replace(x, " ")

  words=line.split()
  wordFreq = {}
  for word in words:
    if word not in stop_words:
      wordFreq[word] = wordFreq.get(word, 0) + 1
  
  for word in wordFreq:
    print('%s %s %s' % (word, docid, wordFreq[word]))
  docid +=1 
