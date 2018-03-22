import argparse
import math
import copy
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import pdb

torch.manual_seed(1)
USE_CUDA = True if torch.cuda.is_available() else False

class CNN_Mult_Embed(nn.Module):
    def __init__(self, model="non-static", vocab_size=None, embedding_dim=300, embeds=None, class_number=None,
                feature_maps=100, filter_windows=[3,4,5], dropout=(0.25,0.5), features=False):
        super(CNN_Mult_Embed, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_number = class_number
        self.filter_windows = filter_windows
        self.in_channel = 1
        self.out_channel = feature_maps
        self.model = model
        self.embedding = nn.Embedding(self.vocab_size+2, embedding_dim)

        # Pretrained Embedding
        self.embedding.weight.data.copy_(torch.from_numpy(embeds[0]))

        if model == "static":
            self.embedding.weight.requires_grad = False
        elif model == "multichannel":
            self.embedding2 = nn.Embedding(self.vocab_size+2, embedding_dim)
            self.embedding2.weight.data.copy_(torch.from_numpy(embeds[1]))
            # self.embedding2.weight.requires_grad = False
            self.in_channel = 2

        self.conv = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (F, embedding_dim)) for F in filter_windows])
        self.dropout = nn.Dropout(dropout[0])
        self.dropout1 = nn.Dropout(dropout[1])

        if features:
            self.fc = nn.Linear(len(filter_windows) * self.out_channel + 4, class_number)
        else:
            self.fc = nn.Linear(len(filter_windows) * self.out_channel, class_number) # Fully connected layer

    def convolution_max_pool(self, inputs, convolution, i, max_sent_len):
        result_convolution = F.relu(convolution(inputs)).squeeze(3) # (batch_size, out_channel, max_seq_len)
        result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2) # (batch_size, out_channel)
        return result

    def forward(self, inputs, features=None, test=False):
        if features:
            rt, fav, usr_followers, usr_following = features
            rt = rt.type(torch.FloatTensor).sqrt()
            fav = fav.type(torch.FloatTensor).sqrt()
            usr_followers = usr_followers.type(torch.FloatTensor).sqrt()
            usr_following = usr_following.type(torch.FloatTensor).sqrt()
            if USE_CUDA:
                rt, fav, usr_followers, usr_following = rt.cuda(), fav.cuda(), usr_followers.cuda(), usr_following.cuda()

        # Pad inputs if less than filter window size
        if inputs.size()[1] <= max(self.filter_windows):
            inputs = F.pad(inputs, (1, math.ceil((max(self.filter_windows)-inputs.size()[1])/2))) # FINISH THIS PADDING
        
        max_sent_len = inputs.size(1)
        embedding = self.embedding(inputs) # (batch_size, max_seq_len, embedding_size)
        embedding = embedding.unsqueeze(1) # (batch_size, 1, max_seq_len, embedding_size)

        if self.model == "multichannel":
            embedding2 = self.embedding2(inputs)
            embedding2 = embedding2.unsqueeze(1)
            embedding = torch.cat((embedding, embedding2), 1)

        embedding = self.dropout(embedding)
        
        result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]

        if features:
            result = torch.cat(result, 1).type(torch.FloatTensor).cuda() if USE_CUDA else torch.cat(result, 1).type(torch.FloatTensor)
            result = torch.cat((result, rt, fav, usr_followers, usr_following), 1) # [batch_sz x (feature maps x filters) + 4]
            result = self.fc(self.dropout(result))
        else:
            result = self.fc(self.dropout(torch.cat(result, 1)))

        if test and features:
            return result.retain_grad(), embedding.retain_grad()
        if test:
            return Variable(result.data, requires_grad=True), Variable(embedding.data, requires_grad=True)
        return result


class CNNClassifier(nn.Module):
    def __init__(self, model="non-static", vocab_size=None, embedding_dim=300, class_number=None,
                feature_maps=100, filter_windows=[3,4,5], dropout=(0.25,0.5), features=False):
        super(CNNClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_number = class_number
        self.filter_windows = filter_windows
        self.in_channel = 1
        self.out_channel = feature_maps
        self.model = model
        self.embedding = nn.Embedding(vocab_size+2, embedding_dim)

        if model == "static":
            self.embedding.weight.requires_grad = False
        elif model == "multichannel":
            self.embedding2 = nn.Embedding(vocab_size+2, embedding_dim)
            self.embedding2.weight.requires_grad = False
            self.in_channel = 2

        self.conv = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (F, embedding_dim)) for F in filter_windows])
        self.dropout = nn.Dropout(dropout[0])
        self.dropout1 = nn.Dropout(dropout[1])

        if features:
            self.fc = nn.Linear(len(filter_windows) * self.out_channel + 4, class_number)
        else:
            self.fc = nn.Linear(len(filter_windows) * self.out_channel, class_number) # Fully connected layer

    def convolution_max_pool(self, inputs, convolution, i, max_sent_len):
        result_convolution = F.relu(convolution(inputs)).squeeze(3) # (batch_size, out_channel, max_seq_len)
        result = F.max_pool1d(result_convolution, result_convolution.size(2)).squeeze(2) # (batch_size, out_channel)
        return result

    def forward(self, inputs, features=None, test=False):
        if features:
            rt, fav, usr_followers, usr_following = features
            rt = rt.type(torch.FloatTensor).sqrt()
            fav = fav.type(torch.FloatTensor).sqrt()
            usr_followers = usr_followers.type(torch.FloatTensor).sqrt()
            usr_following = usr_following.type(torch.FloatTensor).sqrt()
            if USE_CUDA:
                rt, fav, usr_followers, usr_following = rt.cuda(), fav.cuda(), usr_followers.cuda(), usr_following.cuda()

        # Pad inputs if less than filter window size
        if inputs.size()[1] <= max(self.filter_windows):
            inputs = F.pad(inputs, (1, math.ceil((max(self.filter_windows)-inputs.size()[1])/2))) # FINISH THIS PADDING
        
        max_sent_len = inputs.size(1)
        embedding = self.embedding(inputs) # (batch_size, max_seq_len, embedding_size)
        embedding = embedding.unsqueeze(1) # (batch_size, 1, max_seq_len, embedding_size)

        if self.model == "multichannel":
            embedding2 = self.embedding2(inputs)
            embedding2 = embedding2.unsqueeze(1)
            embedding = torch.cat((embedding, embedding2), 1)

        embedding = self.dropout(embedding)
        
        result = [self.convolution_max_pool(embedding, k, i, max_sent_len) for i, k in enumerate(self.conv)]

        if features:
            result = torch.cat(result, 1).type(torch.FloatTensor).cuda() if USE_CUDA else torch.cat(result, 1).type(torch.FloatTensor)
            result = self.dropout1(result)
            pdb.set_trace()
            result = nn.ReLU(torch.cat((result, rt, fav, usr_followers, usr_following), 1)) # [batch_sz x (feature maps x filters) + 4]
            result = self.fc(result)
        else:
            result = self.dropout1(torch.cat(result, 1))
            pdb.set_trace()
            result = nn.ReLU(result)
            pdb.set_trace()
            result = self.fc(result)

        if test and features:
            return result.retain_grad(), embedding.retain_grad()
        if test:
            return Variable(result.data, requires_grad=True), Variable(embedding.data, requires_grad=True)
        return result

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, n_layers=1, batch_sz=128, dropout_p=(0.25,0.5), bidirectional=True, features=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // self.num_directions, n_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p[0])
        self.dropout1 = nn.Dropout(dropout_p[1])

        if features:
            self.hidden2label = nn.Linear(hidden_dim + 4, label_size)
        else:
            self.hidden2label = nn.Linear(hidden_dim, label_size)

    def init_hidden(self, batch_size=128):
        # the first is the hidden h
        # the second is the cell c
        if USE_CUDA:
            return (Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions)).cuda(),
                Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions)).cuda())

        return (Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions)),
                Variable(torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_dim // self.num_directions)))

    def forward(self, inputs, features=None):
        if features:
            rt, fav, usr_followers, usr_following = features
            rt = rt.type(torch.FloatTensor).sqrt()
            fav = fav.type(torch.FloatTensor).sqrt()
            usr_followers = usr_followers.type(torch.FloatTensor).sqrt()
            usr_following = usr_following.type(torch.FloatTensor).sqrt()
            if USE_CUDA:
                rt, fav, usr_followers, usr_following = rt.cuda(), fav.cuda(), usr_followers.cuda(), usr_following.cuda()
            # use logs of larger numbers! LOGS???
            # discretize these if the numbers are super high!

        batch_size = len(inputs)
        hidden = self.init_hidden(batch_size)
        embeddings = self.dropout(self.embedding(inputs))
        # embeddings1 = embeddings.view(len(inputs[0]), batch_size, -1)
        embeddings1 = embeddings.transpose(0, 1)
        output, hidden = self.lstm(embeddings1, hidden)
        # output: [seq_len x batch x hidden]

        if features:
            output = output[-1].type(torch.FloatTensor).cuda() if USE_CUDA else output[-1].type(torch.FloatTensor)
            result = torch.cat((output, rt, fav, usr_followers, usr_following), 1)
            result = self.hidden2label(self.dropout1(result))
        else:
            result = self.hidden2label(self.dropout1(output[-1]))
        return result
