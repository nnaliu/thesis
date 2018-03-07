import csv
import os
import gensim
from gensim.models import Word2Vec
import pandas as pd
import pickle
import preprocessor as p
import align
import pdb

seq = []
with open('cache/all_tweets.csv', 'r') as f:
	reader = csv.reader(f)
	seq.append(list(word for word in reader))

seq = seq[0]

model = Word2Vec(seq, size=300, window=5, min_count=1, workers=4)
print(model)
words = list(model.wv.vocab)
print("Num words:", len(words))
model.wv.save_word2vec_format('my_model.bin')

