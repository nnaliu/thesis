import csv
import os
import numpy as np
import torch.nn as nn
import gensim
from gensim import matutils
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import pickle
import preprocessor as p
import align
import pdb

def import_model(input_file='cache/all_tweets.csv', my_model_filename='my_model.txt'):
    if os.path.exists(my_model_filename):
        my_model = KeyedVectors.load_word2vec_format(my_model_filename, binary=True)
        my_model.init_sims()
    else:
        seq = []
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            seq.append(list(word for word in reader))
        seq = seq[0]
        my_model = Word2Vec(seq, size=300, window=5, min_count=1, workers=4)
        print(my_model)
        words = list(my_model.wv.vocab)
        print("Num words:", len(words))
        # my_model.wv.save_word2vec_format(my_model_filename, binary=False)
    return my_model

def align_model(my_model, my_model_aligned_filename='my_model_aligned.bin'):
    if os.path.exists(my_model_aligned_filename):
        my_model_aligned = Word2Vec.load(my_model_aligned_filename)
    else:
        google_filename = 'GoogleNews-vectors-negative300.bin'
        print("Reading Google")
        gmodel = KeyedVectors.load_word2vec_format(google_filename, binary=True)
        my_model_aligned = align.smart_procrustes_align_gensim(gmodel, my_model)
        my_model_aligned.wv.save_word2vec_format(my_model_aligned_filename, binary=False)
    return my_model_aligned

def compare_word(word, gmodel, aligned_model):
    g_word = gmodel.get_vector(word)
    a_word = aligned_model.get_vector(word)
    output = np.dot(matutils.unitvec(g_word), matutils.unitvec(a_word))
    return output

# my_model = import_model()
# my_model_aligned = align_model(my_model)

my_model_aligned_filename='my_model_aligned.bin'
my_model_aligned = Word2Vec.load(my_model_aligned_filename)
my_model_aligned_txt = my_model_aligned.wv.save_word2vec_format('my_model_aligned.txt', binary=False)

# google: 0.33103912924302953
# the: 0.5419154203028927
# is: 0.530239081501294
# jew: 0.2026357102372878
# that: 0.5108400733421291
# man: 0.5964687428268138
# muslim: 0.43858249536031085
# candy: 0.5549777415600545
# aboriginal: 0.3755228368294336
# skype: 0.5048145927328979
# african: 0.29049011280501214
# guinea: 0.2655043914639797

# common_english_words = ['the','of','and','a','to','in','is','you','that','it','he','was','for','on','are','as','with','his','they','I']

# print("Reading Google")
# google_filename = 'GoogleNews-vectors-negative300.bin'
# gmodel = KeyedVectors.load_word2vec_format(google_filename, binary=True)

# print("Reading own model")
# my_model_aligned_filename = 'my_model_aligned.bin'
# aligned_model = KeyedVectors.load_word2vec_format(my_model_aligned_filename, binary=True)

# for word in common_english_words:
#     similarity = compare_word(word, gmodel, aligned_model)
#     print(word + "     Similarity: " + str(similarity))
