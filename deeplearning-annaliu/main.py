import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

import data_handler, utils
import model
import pdb

torch.manual_seed(1)

USE_CUDA = True if torch.cuda.is_available() else False
N_FOLDS = 10


parser = argparse.ArgumentParser(description='Hate Speech Classification')
parser.add_argument('--model', type=str, help='type of model')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

# Need to figure out how to not have headers writing to file in middle (Ctrl+F 'retweet_count')
tweet_data = data_handler.prepare_csv()
print("Finished preparing CSV")

# Word embeddings
# vectors = [GloVe(name='42B', dim='300')] # CharNGram(), FastText()
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
vectors = Vectors('wiki.simple.vec', url=url)
# vectors=None

# train, val, test, vocab_size, tweet_vocab = data_handler.read_files(vectors=vectors)
# train_iter, val_iter, test_iter = data_handler.get_bucket_iterators((train, val, test), args.batch_size)
# print("Vocab size ", vocab_size)

train_val_generator = get_dataset(tweet_data, lower=True, vectors=vectors, n_folds=N_FOLDS, seed=42)

for fold, (train, val) in enumerate(train_val_generator):
    train_iter, val_iter = data_handler.get_bucket_iterators((train, val), args.batch_size)

    if args.model == 'CNN':
        model = model.CNNClassifier(model='multichannel', vocab_size=vocab_size, class_number=2)
        if USE_CUDA:
            print("USING CUDA")
            model = model.cuda()
        utils.train(model, train_iter, val_iter, 10) # Change number of epochs later
        print("Validation: ", utils.evaluate(model, val_iter))

    elif args.model == "CNNFeatures":
        model = model.CNNClassifierFeatures(model='multichannel', vocab_size=vocab_size, class_number=2)
        if USE_CUDA:
            print("USING CUDA")
            model = model.cuda()
        utils.train(model, train_iter, val_iter, 10, has_features=True) # Change number of epochs later
        print("Validation: ", utils.evaluate(model, val_iter, has_features=True))

    elif args.model == 'LSTM':
        model = model.LSTMClassifier(256, 300, vocab_size, 2, n_layers=4, batch_sz=args.batch_size) # embedding dim, hidden dim, vocab_size, label_size
        if USE_CUDA:
            print("USING CUDA")
            model = model.cuda()
        utils.train(model, train_iter, val_iter, 30)
        print("Validation: ", utils.evaluate(model, val_iter))

    elif args.model == 'LSTMFeatures':
        model = model.LSTMClassifierFeatures(256, 300, vocab_size, 2, n_layers=4, batch_sz=args.batch_size) # embedding dim, hidden dim, vocab_size, label_size
        if USE_CUDA:
            print("USING CUDA")
            model = model.cuda()
        utils.train(model, train_iter, val_iter, 30, has_features=True)
        print("Validation: ", utils.evaluate(model, val_iter, has_features=True))

# Saving Model
if args.model == "CNN":
    filename = 'cnn_model.sav'
elif args.model == "CNNFeatures":
    filename = 'cnn_model_features.sav'
elif args.model == "LSTM":
    filename = 'lstm_model.sav'
elif args.model == "LSTMFeatures":
    filename = 'lstm_model_features.sav'
torch.save(model.state_dict(), filename)

"""
GuidedBackProp Saliency Analysis
"""

# filename = 'cnn_model.sav'
# cnn = model.CNNClassifier(model='multichannel', vocab_size=vocab_size, class_number=2)
if USE_CUDA:
    print("converting to cuda")
    model = model.cuda()
# model.load_state_dict(torch.load(filename))

counter = 0
batch = next(iter(train_iter))
text, label = utils.process_batch(batch)
for text_i, label_i in zip(text, label):
    # utils.saliency_map(model, text_i, label_i)
    text_words = " ".join([tweet_vocab.vocab.itos[i.data[0]] for i in text_i])
    print("TEXT: ", text_words)
    print("LABEL: ", label_i)
    GBP = utils.GuidedBackprop(model, text_i, label_i-1)
    guided_grads = GBP.generate_gradients()

    metadata = open('./data/metadata' + str(counter) + '.txt', 'w')
    metadata.write(text_words)
    metadata.write(str(label_i.data))
    metadata.close()

    utils.save_saliency_map(counter, guided_grads, tweet_vocab, text_i, label_i)
    counter += 1

