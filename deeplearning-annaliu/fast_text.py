import data_handler
import fasttext
import csv
from sklearn.metrics import precision_recall_fscore_support
import pdb

data_handler.prepare_fasttext()

N_FOLDS = 10

# train_val_generator, tweet_vocab = data_handler.get_dataset(lower=True, vectors=None, n_folds=N_FOLDS, seed=42)
# my_embed = data_handler.get_pretrained_embedding(tweet_vocab, '../semantics/my_model_dstormer_aligned.txt')
# g_embed = data_handler.get_pretrained_embedding(tweet_vocab, '../semantics/GoogleNews-vectors-negative300.bin')

def train_model():
    p_avg, r_avg, f1_avg = 0., 0., 0.
    p1_avg, r1_avg, f11_avg = 0., 0., 0.

    for count in range(N_FOLDS):
        input_file = './cache/fasttext/train' + str(count) + '.txt'
        output_file = './cache/fasttext/model' + str(count)

        classifier = fasttext.supervised(input_file, output_file, epoch=40, lr=0.1, dim=100,
                                        bucket= 2000000)

        test_file = open("./cache/fasttext/test" + str(count) + '.txt', "r")
        tweet_file = test_file.readlines()

        tweets = [line.split(' ', 1)[1] for line in tweet_file]
        labels = [line.split(' ', 1)[0].split('__label__')[1] for line in tweet_file]

        result = classifier.test('./cache/fasttext/test' + str(count) + '.txt')
        print('\nFOLD ' + str(count))
        print('Number of examples:', result.nexamples)

        preds = classifier.predict(tweets)
        preds = [label for sublist in preds for label in sublist]

        p, r, f1, s = precision_recall_fscore_support(labels, preds, average='weighted')
        p1, r1, f11, s1 = precision_recall_fscore_support(labels, preds, average='micro')

        # p_avg += result.precision
        # r_avg += result.recall

        p_avg += p
        r_avg += r
        f1_avg += f1
        p1_avg += p1
        r1_avg += r1
        f11_avg += f11

        print('VAL - WEIGHTED RESULTS')
        print('precision is ' + str(p))
        print('recall is ' + str(r))
        print('f1 is ' + str(f1))
        print('VAL - MICRO RESULTS')
        print('precision is ' + str(p1))
        print('recall is ' + str(r1))
        print('f1 is ' + str(f11))
        print('\n')

    print('WEIGHTED RESULTS')
    print('average precision is ' + str(p_avg/N_FOLDS))
    print('average recall is ' + str(r_avg/N_FOLDS))
    print('average f1 is ' + str(f1_avg/N_FOLDS))

    print('MICRO RESULTS')
    print('average precision is ' + str(p1_avg/N_FOLDS))
    print('average recall is ' + str(r1_avg/N_FOLDS))
    print('average f1 is ' + str(f11_avg/N_FOLDS))

# writefile = open('../Data/semantic_data/hate_speech.txt', 'w')
# with open('../Data/semantic_data/dstormer.csv', 'r') as f:
#     tweets = f.readlines()[1:]
#     for tweet in tweets:
#         writefile.write('%s' % tweet.split(',', 1)[1].strip("\""))
# with open('../Data/semantic_data/hate_speech_users.csv', 'r') as f:
#     tweets = f.readlines()[1:]
#     for tweet in tweets:
#         writefile.write('%s' % tweet.split(',', 1)[1].strip("\""))

# print("Making skipgram model")
# model = fasttext.skipgram('../Data/semantic_data/hate_speech.txt', 'ft_model')
# print(model.words)

train_model()

# To load this model later:
# from gensim.models import FastText
# model = FastText.load_fasttext_format('ft_model.bin')


















