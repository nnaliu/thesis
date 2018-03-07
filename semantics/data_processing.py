import csv
import os
import pandas as pd
import pickle
import preprocessor as p
import pdb
import joblib
import pymongo


tweet_folder = '../user-tweets/'

# Preprocess tweets
def preprocess(s, lowercase=False):
    import string
    table = str.maketrans('', '', string.punctuation)
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED)
    s = p.tokenize(s)
    tokens_raw = s.replace('\n','').replace('\r','').split()
    if lowercase:
      tokens = [token.strip().lower().translate(table) for token in tokens_raw]
    else:
      tokens = [token.strip().translate(table) for token in tokens_raw]
    return tokens

def import_data():
    # hate_speech_users = joblib.load('../Data/hs_users.plk.compressed')
    # hate_speech_users.to_csv("../Data/semantic_data/hate_speech_users.csv", index=False, index_label=False, encoding='utf-8')
    dstormer = joblib.load('../Data/dstormer.plk.compressed')
    with open('cache/dstormer.csv', 'a') as f:
        dstormer['preprocessed_txt'] = dstormer['preprocessed_txt'].apply(lambda x: preprocess(str(x), lowercase=True))
        text = dstormer['preprocessed_txt'].values.tolist()
        writer = csv.writer(f)
        writer.writerows(text)

    # dstormer.to_csv("../Data/semantic_data/dstormer.csv", index=False, index_label=False, encoding='utf-8')
    return dstormer #, hate_speech_users

def import_tweets(tweet_folder):
    with open('cache/all_tweets.csv', 'a') as f:
        for i, filename in enumerate(os.listdir(tweet_folder)):
            file_path = tweet_folder + filename
            tweet_data = pd.read_csv(file_path, encoding='utf-8')
            tweet_data['text'] = tweet_data['text'].apply(lambda x: preprocess(str(x), lowercase=True))
            tweets = tweet_data['text'].values.tolist()
            writer = csv.writer(f)
            writer.writerows(tweets)

            if i % 500 == 0:
                print("Processing user #" + str(i))

# import_tweets(tweet_folder)
import_data()