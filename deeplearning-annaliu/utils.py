import numpy as np
import torchtext 
import torch
import torch.autograd as autograd

torch.manual_seed(1)

def process_batch(batch):
    # FILL THIS IN HERE
    text, label = batch.text.t_(), batch.label
    if torch.cuda.is_availabe():
        text = text.cuda()
        label = label.cuda()
    return text, label

def train(model, data_iter, epochs, optimizer, scheduler=None, grad_norm=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_iter:
            text, label = process_batch(batch)
            label = label - 1
            model.zero_grad()
            logit = model(text)
            loss = criterion(logit, label)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=grad_norm)
            optimizer.step()
            total_loss += loss.data
        print(str(epoch) + " loss = " + str(total_loss)) # Find a better print statement

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