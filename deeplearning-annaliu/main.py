import argparse
import pandas as pd
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

USE_CUDA = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='Hate Speech Classification')
parser.add_argument('--model', type=str, default='CNN',
                    help='type of model')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--bptt', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

# Need to figure out how to not have headers writing to file in middle (Ctrl+F 'retweet_count')
data_handler.prepare_csv()

# Word embeddings
# vectors = [GloVe(name='42B', dim='300')] # CharNGram(), FastText()
# url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
# vectors = Vectors('wiki.simple.vec', url=url)
vectors=None
train, val, test, vocab_size = data_handler.read_files(vectors=vectors)
# train, text, val = data_handler.restore_dataset(train_examples, val_examples, test_examples)
print("Vocab size ", vocab_size)

if args.model == 'CNN':
    train_iter, val_iter, test_iter = data_handler.get_bucket_iterators((train, val, test), args.batch_size)
    model = model.CNNClassifier(model='multichannel', vocab_size=vocab_size, class_number=2)
    if USE_CUDA:
        print("USING CUDA")
        model = model.cuda()
    utils.train(model, train_iter, val_iter, 40) # Change number of epochs later
    print("Validation: ", utils.evaluate(model, val_iter))

    # Saving Model
    filename = 'cnn_model.sav'
    torch.save(model.state_dict(), filename)

elif args.model == "CNNFeatures":
    train_iter, val_iter, test_iter = data_handler.get_bucket_iterators((train, val, test), args.batch_size)
    model = model.CNNClassifierFeatures(model='multichannel', vocab_size=vocab_size, class_number=2)
    if USE_CUDA:
        print("USING CUDA")
        model = model.cuda()
    utils.train(model, train_iter, val_iter, 40, has_features=True) # Change number of epochs later
    print("Validation: ", utils.evaluate(model, val_iter, has_features=True))

    # Saving Model
    filename = 'cnn_model_features.sav'
    torch.save(model.state_dict(), filename)

elif args.model == 'LSTM':
    train_iter, val_iter, test_iter = data_handler.get_bucket_iterators((train, val, test), args.batch_size)
    model = model.LSTMClassifier(300, 200, vocab_size, 4) # embedding dim, hidden dim, vocab_size, label_size
    if USE_CUDA:
        print("USING CUDA")
        model = model.cuda()
    utils.train(model, train_iter, val_iter, 70)
    print("Validation: ", utils.evaluate(model, val_iter))



