import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import Memory
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText

import pdb

torch.manual_seed(1)
MEMORY = Memory(cachedir="cache/", verbose=1)

def prepare_csv():
    tweets = []
    file_path = '../Data/tweet_data/NAACL_SRW_2016_tweets.csv'
    # Read in data. Will probably need to CHANGE THIS to get rid of index and header col.
    columns = "id,created_at,text,retweet_count,favorite_count,user_screen_name,user_id,user_followers_count,user_following_count,hate_label".split(',')
    tweet_data = pd.read_csv(file_path, names=columns, encoding='utf-8')
    y = tweet_data['hate_label'] # Define the target variable as y. CHECK IF CORRECT

    train, test = train_test_split(tweet_data, test_size=0.2)
    print(train.shape, test.shape)
    # Get validation dataset as well
    train, val = train_test_split(train, test_size=0.2, random_state=1)

    train.to_csv("cache/tweets_train.csv", index=False)
    val.to_csv("cache/tweets_val.csv", index=False)
    test.to_csv("cache/tweets_test.csv", index=False)

def read_files(lower=False, vectors=None):
    #############################
    #  THIS ALL NEEDS TO BE FIXED
    lower = True if vectors is not None else False
    tweet = data.Field(sequential=False, tensor_type=torch.LongTensor, lower=lower)
    fields = [
        ('id', None),
        ('text', tweet),
        ('label', data.Field(
            use_vocab=False, sequential=False, tensor_type=torch.ByteTensor)),
        ('name', None),
        ('retweet_count', data.Field(
            use_vocab=False, sequential=False, tensor_type=torch.IntTensor)),
        ('favorite_count', data.Field(
            use_vocab=False, sequential=False, tensor_type=torch.IntTensor)),
        ('user_followers_count', data.Field(
            use_vocab=False, sequential=False, tensor_type=torch.IntTensor)),
        ('user_following_count', data.Field(
            use_vocab=False, sequential=False, tensor_type=torch.IntTensor)),
    ]

    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=fields
    )

    # Might need to change this later
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv', skip_header=True,
        fields=fields
    )
    tweet.build_vocab(
        train, val, test, max_size=20000, min_freq=50, vectors=vectors)
    # What do these mean?

    return train, val, test, len(tweet.vocab)
    ####################

# datasets is a tuple of dataset objects. The first one is the train set
def get_cnn_iterators(datasets, batch_size, shuffle=True, repeat=False):
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        datasets, batch_size=batch_size, shuffle=shuffle, repeat=False, device=-1)
    return train_iter, val_iter, test_iter

def get_rnn_iterators(datasets, batch_size, bptt=32, shuffle=True, repeat=False):
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        datasets, batch_size=batch_size, shuffle=shuffle, bptt_len=bptt, repeat=False, device=-1)
    return train_iter, val_iter, test_iter
