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

def train(model, data_iter, epochs, scheduler=None, grad_norm=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=0.5)

    for epoch in range(epochs):
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
        print(str(epoch) + " loss = " + str(total_loss)) # Find a better print statement

def evaluate(model, data_iter):
    model.eval()
    correct, total = 0., 0.
    for batch in val_iter:
        probs = model(batch.text.t_())
        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):
            if predicted+1 == batch.label[i].data[0]:
                correct += 1
            total += 1
    return correct / total

# THIS ALL NEEDS TO BE FIXED
def evaluate(model, data_iter, optimizer):
    model.eval()
    correct, total = 0.0, 0.0
    for batch in data_iter:
        text, label = process_batch(batch)
        probs = model(text.t_())
        _, argmax = probs.max(1)
        for i, predicted in enumerate(list(argmax.data)):
            if predicted+1 == batch.label[i].data[0]:
                correct += 1
            total += 1
    return correct / total