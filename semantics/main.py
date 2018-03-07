import csv
import os
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
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

my_model = Word2Vec(seq, size=300, window=5, min_count=1, workers=4)
print(my_model)
words = list(my_model.wv.vocab)
print("Num words:", len(words))
my_model.wv.save_word2vec_format('my_model.bin')

glove_filename = '../deeplearning-annaliu/.vector_cache/glove.840B.300d.txt'
gmodel = KeyedVectors.load_word2vec_format(glove_filename, binary=True)
word2vec_output = 'glove.840B.300d.txt.word2vec'
glove2word2vec(glove_filename, word2vec_output)

gmodel = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)
my_model_aligned = align.smart_procrustes_align_gensim(gmodel, my_model)
my_model_aligned.wv.save_word2vec_format('my_model_aligned.bin')

