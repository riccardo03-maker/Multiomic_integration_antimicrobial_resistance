#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import softmax
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    Train loop for the tuning of parameters of neural network.
    '''
    model.train()
    for x, y in dataloader:
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss


def try_nn():
    '''
    Try function to create a neural network.
    '''
    #define the neural network
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 2)
            )
        
        def forward(self, x):
            x = self.linear_relu_stack(x)
            return x
    
    #create tensor with data
    model = NeuralNetwork()
    x = torch.rand(100, 100)
    y = torch.randint(0, 2, size = (100,))
    test_x = torch.rand(20, 100)
    test_y = torch.randint(0, 2, size = (20,))
    dataset = TensorDataset(x, y)
    
    #load dataset and define hyperparameters for model optimization
    dataloader = DataLoader(dataset, batch_size = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 100
    
    #train the model
    for i in range(epochs):
        loss = train_loop(dataloader, model, loss_fn, optimizer)
        print("Epoch: " + str(i+1) + ", loss: " + str(loss))
    
    #test performances
    model.eval()
    pred = model(x)
    predicted_classes = pred.argmax(1)
    
    #compute the classification scores
    predicted_classes = np.array(predicted_classes.numpy(), dtype = np.float64)
    true_classes = np.array(y.numpy(), dtype = np.float64)

    model.eval()
    test_pred = model(test_x)
    predicted_classes_test = test_pred.argmax(1)
    predicted_classes_test = np.array(predicted_classes_test.numpy(), dtype = np.float64)
    true_classes_test = np.array(test_y.numpy(), dtype = np.float64)
    
    print(accuracy_score(true_classes, predicted_classes)) #return is 1 because the model perfectly learns the classes of the training set
    print(accuracy_score(true_classes_test, predicted_classes_test)) #almost 0.5 because there is no relation between input and output,
    #so classes are assigned almost randomly


if(__name__ == '__main__'):
    try_nn()
