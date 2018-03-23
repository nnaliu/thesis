import gensim, logging
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import KeyedVectors
import pandas as pd
import preprocessor as p
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from joblib import Memory
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import numpy as np

import pdb

torch.manual_seed(1)
MEMORY = Memory(cachedir="cache/", verbose=1)

hate_label = {
    'none' : 0,
    'racism' : 1,
    'sexism' : 1,
    'both': 1
}

# Preprocess tweets
def preprocess(s, lowercase=False):
    import string
    table = str.maketrans('', '', string.punctuation)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
    s = p.tokenize(s)
    tokens_raw = s.replace("\n","").strip("'[]\' '").split()

    # Added lemmatization (1/1/18). Potentially should remove to compare results
    # tokens = gensim.utils.lemmatize(s, stopwords=STOPWORDS, min_length=1)

    # Without lemmatization, remove punctuation
    if lowercase:
      tokens = [token.strip().lower().translate(table) for token in tokens_raw]
    else:
      tokens = [token.strip().translate(table) for token in tokens_raw]

    return tokens

def prepare_csv():
    tweets = []
    file_path = '../Data/tweet_data/NAACL_SRW_2016_tweets.csv'
    # Read in data. Will probably need to CHANGE THIS to get rid of index and header col.
    # columns = "id,created_at,text,retweet_count,favorite_count,user_screen_name,user_id,user_followers_count,user_following_count,hate_label".split(',')
    tweet_data = pd.read_csv(file_path, encoding='utf-8') # names=columns,
    tweet_data['text'] = tweet_data['text'].apply(lambda x: preprocess(str(x), lowercase=True))
    tweet_data['text'] = tweet_data['text'].apply(lambda x: ' '.join(x))
    tweet_data['hate_label'] = tweet_data['hate_label'].apply(lambda x: hate_label[x] if x in hate_label else 0)

    # Rescaling my features

    print("PRINTING FEATURES FOR PAPER")
    print("retweet count: " + str(max(tweet_data['retweet_count'])))
    print("favorite_count: " + str(max(tweet_data['favorite_count'])))
    print("user_followers_count: " + str(max(tweet_data['user_followers_count'])))
    print("user_following_count: " + str(max(tweet_data['user_following_count'])))

    tweet_data['retweet_count'] = tweet_data['retweet_count'] / max(tweet_data['retweet_count'])
    tweet_data['favorite_count'] = tweet_data['favorite_count'] / max(tweet_data['favorite_count'])
    tweet_data['user_followers_count'] = tweet_data['user_followers_count'] / max(tweet_data['user_followers_count'])
    tweet_data['user_following_count'] = tweet_data['user_following_count'] / max(tweet_data['user_following_count'])
    pdb.set_trace()

    # train, test = train_test_split(tweet_data, test_size=0.2)
    # print(train.shape, test.shape)
    # # Get validation dataset as well
    # train, val = train_test_split(train, test_size=0.1, random_state=1)

    # train.to_csv("cache/tweets_train.csv", index=False, index_label=False, encoding='utf-8')
    # val.to_csv("cache/tweets_val.csv", index=False, index_label=False, encoding='utf-8')
    # test.to_csv("cache/tweets_test.csv", index=False, index_label=False, encoding='utf-8')

    tweet_data.to_csv("cache/tweets_data.csv", index=False, index_label=False, encoding='utf-8')
    return tweet_data

# https://github.com/galsang/CNN-sentence-classification-pytorch/blob/master/run.py
def get_pretrained_embedding(tweet_vocab, file):
    pretrained_vectors = KeyedVectors.load_word2vec_format(file, binary=True)
    
    wv_matrix = []
    for word in tweet_vocab.vocab.itos:
        if word in pretrained_vectors.vocab:
            wv_matrix.append(pretrained_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)

    print("Returning pretrained embedding")
    return wv_matrix

def get_dataset(lower=False, vectors=None, n_folds=10, seed=42):
    lower = True if vectors is not None else False
    # tweet = data.Field(sequential=False, tensor_type=torch.LongTensor, lower=lower)
    tweet = data.Field(sequential=True)
    label = data.Field(sequential=False)
    # label = data.Field(sequential=False, tensor_type=torch.LongTensor, preprocessing=data.Pipeline(lambda x: int(x)))
    retweet_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    favorite_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    user_followers_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    user_following_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    fields = [
        ('id', None),
        ('created_at', None),
        ('text', tweet),
        ('retweet_count', retweet_count),
        ('favorite_count', favorite_count),
        ('user_screen_name', None),
        ('user_id', None),
        ('user_followers_count', user_followers_count),
        ('user_following_count', user_following_count),
        ('hate_label', label),
    ]

    all_tweets = data.TabularDataset(
        path='cache/tweets_data.csv', format='csv', skip_header=True,
        fields=fields
    )

    tweet.build_vocab(all_tweets, vectors=vectors)
    label.build_vocab(all_tweets)
    tweet_exp = np.array(all_tweets.examples)
    train_val = tweet_exp

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    def iter_fold():
        train_val_arr = []
        for train_idx, val_idx in kf.split(train_val):
            train = data.Dataset(list(train_val[train_idx]), fields)
            val = data.Dataset(list(train_val[val_idx]), fields)
            train_val_arr.append((train, val))
            # yield (train, val,)
        return train_val_arr

    return iter_fold(), tweet


def read_files(lower=False, vectors=None):
    #############################
    #  THIS ALL NEEDS TO BE FIXED
    lower = True if vectors is not None else False
    # tweet = data.Field(sequential=False, tensor_type=torch.LongTensor, lower=lower)
    tweet = data.Field(sequential=True)
    label = data.Field(sequential=False)
    # label = data.Field(sequential=False, tensor_type=torch.LongTensor, preprocessing=data.Pipeline(lambda x: int(x)))
    retweet_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    favorite_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    user_followers_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    user_following_count = data.Field(use_vocab=False, tensor_type=torch.FloatTensor, preprocessing=data.Pipeline(lambda x: float(x)))
    fields = [
        ('id', None),
        ('created_at', None),
        ('text', tweet),
        ('retweet_count', retweet_count),
        ('favorite_count', favorite_count),
        ('user_screen_name', None),
        ('user_id', None),
        ('user_followers_count', user_followers_count),
        ('user_following_count', user_following_count),
        ('hate_label', label),
    ]
    
    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='tweets_train.csv', validation='tweets_val.csv',
        fields=fields
    )
    # Might need to change this later
    test = data.TabularDataset(
        path='cache/tweets_test.csv', format='csv', skip_header=True,
        fields=fields
    )
    tweet.build_vocab(train, vectors=vectors)
    label.build_vocab(train)
    # What do these mean?

    return train, val, test, len(tweet.vocab), tweet
    ####################

# FILL THIS IN
# def restore_dataset(train_exmp, val_exp, test_exp):
#     train = Dataset()
#     val = Dataset()
#     test = Dataset()

# datasets is a tuple of dataset objects. The first one is the train set
def get_bucket_iterators(datasets, batch_size, shuffle=False, repeat=False):
    train_iter, val_iter = data.BucketIterator.splits(
        datasets, batch_size=batch_size, sort_key=lambda x: len(x.text), shuffle=shuffle,
        repeat=False, device=-1)
    return train_iter, val_iter
