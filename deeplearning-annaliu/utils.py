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
    optimizer = optim.Adam(parameters, lr=0.001)
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

class GuidedBackprop():
   
    def __init__(self, model, inputs, target_class):
        super(GuidedBackprop, self).__init__()
        self.model = model
        self.input = inputs
        self.target_class = target_class

        self.gradients = None

        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Only return positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            # If there's a negative gradient, change to zero
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self):
        pdb.set_trace()
        # Forward pass
        output = self.model(self.inputs)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1
        # Backward pass
        output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

def get_positive_negative_saliency(gradient):
    pdb.set_trace()
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


