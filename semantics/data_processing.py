import csv
import os
import pandas as pd
import pickle
import preprocessor as p
import pdb

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

import_tweets(tweet_folder)