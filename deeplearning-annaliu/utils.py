import numpy as np
import torchtext 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import pdb

torch.manual_seed(1)

def process_batch(batch):
    # FILL THIS IN HERE
    text, hate_label = batch.text.t_(), batch.hate_label
    if torch.cuda.is_available():
        text = text.cuda()
        hate_label = hate_label.cuda()
    return text, hate_label

def process_batch2(batch):
    # FILL THIS IN HERE
    text, hate_label = batch.text.t_(), batch.hate_label
    torch.cat((text, batch.retweet_count.t(), batch.favorite_count.t(), batch.user_followers_count.t(), batch.user_following_count.t()), 1)
    if torch.cuda.is_available():
        text = text.cuda()
        hate_label = hate_label.cuda()
    return text, hate_label

def train(model, data_iter, val_iter, epochs, scheduler=None, grad_norm=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=0.0001)

    counter = 0
    for epoch in range(epochs):
        counter += 1
        total_loss = 0
        for batch in data_iter:
            text, label = process_batch(batch)
            model.zero_grad()
            logit = model(text)
            label = label - 1
            loss = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=grad_norm)
            optimizer.step()
            total_loss += loss.data
        if counter % 5 == 0:
            print("Validation: ", evaluate(model, val_iter))
        print(str(epoch) + " loss = " + str(total_loss)) # Find a better print statement

def train2(model, data_iter, val_iter, epochs, scheduler=None, grad_norm=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=0.5)

    counter = 0
    for epoch in range(epochs):
        counter += 1
        total_loss = 0
        for batch in data_iter:
            text, label = process_batch(batch)
            model.zero_grad()
            logit = model(text, (batch.retweet_count.t(), batch.favorite_count.t(), batch.user_followers_count.t(), batch.user_following_count.t()))
            label = label - 1
            loss = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=grad_norm)
            optimizer.step()
            total_loss += loss.data
        if counter % 5 == 0:
            print("Validation: ", evaluate(model, val_iter))
        print(str(epoch) + " loss = " + str(total_loss)) # Find a better print statement

def evaluate(model, data_iter):
    model.eval()
    correct, total = 0., 0.
    for batch in data_iter:
        text, label = process_batch(batch)
        probs = model(text)
        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):
            if predicted+1 == label[i].data[0]:
                correct += 1
            total += 1
    return correct / total