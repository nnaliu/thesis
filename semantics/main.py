import csv
import os
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import pickle
import preprocessor as p
import align
import pdb

def import_model(input_file='cache/all_tweets.csv', my_model_filename='my_model.bin'):
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
		my_model.wv.save_word2vec_format(my_model_filename, binary=True)
	return my_model

def align_model(my_model, my_model_aligned_filename='my_model_aligned.bin'):
	if os.path.exists(my_model_aligned_filename):
		my_model_aligned = Word2Vec.load(my_model_aligned_filename)
	else:
		google_filename = 'GoogleNews-vectors-negative300.bin'
		print("Reading Google")
		gmodel = KeyedVectors.load_word2vec_format(google_filename, binary=True)
		my_model_aligned = align.smart_procrustes_align_gensim(gmodel, my_model)
		my_model_aligned.wv.save_word2vec_format(my_model_aligned_filename, binary=True)
	return my_model_aligned

def compare_word(word, gmodel, aligned_model):
	g_word = gmodel.get_vector(word)
	a_word = aligned_model.get_vector(word)

	pdb.set_trace()

	cos = nn.CosineSimilarity(dim=1, eps=1e-6) # I have no idea if this is right
	output = cos(g_word, a_word)

common_english_words = ['the','of','and','a','to','in','is','you','that','it','he','was','for','on','are','as','with','his','they','I']

google_filename = 'GoogleNews-vectors-negative300.bin'
gmodel = KeyedVectors.load_word2vec_format(google_filename, binary=True)
aligned_model_filename = 'my_model_aligned.bin'
aligned_model = Word2Vec.load(my_model_aligned_filename)

for word in common_english_words:
	similarity = compare_word(word, gmodel, aligned_model)
	print(word + "     Similarity: " + str(similarity))
