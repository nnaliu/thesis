import numpy as np
import torchtext 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import pdb

torch.manual_seed(1)

USE_CUDA = True if torch.cuda.is_available() else False

def process_batch(batch):
    # FILL THIS IN HERE
    text, hate_label = batch.text.t_(), batch.hate_label
    if USE_CUDA:
        text = text.cuda()
        hate_label = hate_label.cuda()
    return text, hate_label

def process_batch2(batch):
    # FILL THIS IN HERE
    text, hate_label = batch.text.t_(), batch.hate_label
    rt, fav = batch.retweet_count.t(), batch.favorite_count.t()
    usr_followers = batch.user_followers_count.t()
    usr_following = batch.user_following_count.t()
    if USE_CUDA:
        text = text.cuda()
        hate_label = hate_label.cuda()
    return text, hate_label, (rt, fav, usr_followers, usr_following)

def train(model, data_iter, val_iter, epochs, scheduler=None, grad_norm=5, has_features=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adagrad(parameters, lr=0.001)
    # optimizer = optim.SGD(parameters, lr=0.1)

    counter = 0
    for epoch in range(epochs):
        counter += 1
        total_loss = 0
        for batch in data_iter:
            model.zero_grad()

            if has_features:
                text, label, features = process_batch2(batch)
                logit = model(text, features)
            else:
                text, label = process_batch(batch)
                logit = model(text)

            # pdb.set_trace()

            label = label - 1
            loss = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=2)
            optimizer.step()
            total_loss += loss.data
        if counter % 10 == 0:
            print("Validation: ", evaluate(model, val_iter, has_features))
        print(str(epoch) + " loss = " + str(total_loss)) # Find a better print statement

def evaluate(model, data_iter, has_features=False):
    model.eval()
    correct, total = 0., 0.
    for batch in data_iter:
        if has_features:
            text, label, features = process_batch2(batch)
            probs = model(text, features)
        else:
            text, label = process_batch(batch)
            probs = model(text)

        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):
            if predicted+1 == label[i].data[0]:
                correct += 1
            total += 1
    return correct / total