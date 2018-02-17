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
import model_test
import pdb

parser = argparse.ArgumentParser(description='Hate Speech Classification')
parser.add_argument('--model', type=str, default='CNN',
                    help='type of model')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--bptt', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

data_handler.prepare_csv()
# Word embeddings
# url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
vectors = [GloVe(name='840B', dim='300'), CharNGram(), FastText()]
train, test, val, vocab_size = data_handler.read_files(vectors=vectors)

if args.model == 'CNN':
    train_iter, test_iter, val_iter = data_handler.get_cnn_iterators((train, test, val), args.batch_size)
    model = model_test.CNNClassifier(model='multichannel', vocab_size=vocab_size, class_number=2)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=0.5)
    utils.train(model, train_iter, 1, optimizer) # Change number of epochs later

elif args.model == 'RNN':
    train_iter, test_iter, val_iter = data_handler.get_rnn_iterators((train, test, val), args.batch_size)
