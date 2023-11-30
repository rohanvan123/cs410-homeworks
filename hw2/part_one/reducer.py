# Code for reducer.py to be used on colab

# Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
from operator import itemgetter
import sys

current_word = None
current_count = 0
current_word_counts = []

word = None


# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line=line.lower()
    
    
    # parse the input we got from mapper.py
    word, docid, count = line.split(' ', 2)
    
    try:
      count = int(count)
    except ValueError:
      #count was not a number, so silently
      #ignore/discard this line
      continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word :
        current_count += count
        current_word_counts.append((docid, count))
        
    else:
        if current_word:
            # write result to STDOUT
            print ('%s	%s' % (current_word, str(current_word_counts)))
        current_count = count
        current_word = word
        current_word_counts = []
        current_word_counts.append((docid, count))


# do not forget to output the last word if needed!
if current_word == word:
    print('%s	%s' % (current_word, str(current_word_counts)))
