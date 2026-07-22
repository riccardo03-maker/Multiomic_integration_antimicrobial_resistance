#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import softmax
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd

from ml_functions.ml_functions import weighted_train_test_split

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


def evaluate(model: nn.Module, X_test: torch.tensor, Y_test: torch.tensor):
    '''
    Evaluate model classification performances for a neural network, giving as output five scores: precision for susceptible and resistent,
    recall for susceptible (0) and resistent (1), and accuracy.

    Parameters
    ----------
        model: torch.nn.Module
            The neural network classification model.
        X_test: torch.tensor
            The input features for the model.
        Y_test: torch.tensor
            The true output classes for each sample in the X_test set.
    Returns
    -------
        precision_s: float
            The precision score for the susceptible class.
        precision_r: float
            The precision score for the resistent class.
        recall_s: float
            The recall score for the susceptible class.
        recall_r: float
            The recall score for the resistent class.
        accuracy: float
            The accuracy of classification.
    '''
    model.eval()
    pred = model(X_test)
    Y_predict = pred.argmax(1)
    Y_predict = np.array(Y_predict.numpy(), dtype = np.int32)
    Y_test = np.array(Y_test.numpy(), dtype = np.int32)

    precision_s = precision_score(Y_test, Y_predict, pos_label = 0)
    precision_r = precision_score(Y_test, Y_predict, pos_label = 1)
    recall_s = recall_score(Y_test, Y_predict, pos_label = 0)
    recall_r = recall_score(Y_test, Y_predict, pos_label = 1)
    accuracy = accuracy_score(Y_test, Y_predict)
    return precision_s, precision_r, recall_s, recall_r, accuracy


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


def neural_network_single_feature_type():
    '''
    Implement a classification algorithm based on a neural network, using only one type of feature (genexp, gpa or snps).

    Since we are using just one type of feature, we don't need to implement any fusion strategy. The neural network used has two
    hidden layer, the first one with a number of nodes equal to half the number of initial features, and the second one with half of the
    node of the first hidden layer.

    Fully connected neural network with a ReLU activation function, a cross-entropy loss function and Adam optimizer.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'accuracy_training', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])

    #define the neural network
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(6026, 3013),
                nn.ReLU(),
                nn.Linear(3013, 1506),
                nn.ReLU(),
                nn.Linear(1506, 2)
            )
        
        def forward(self, x):
            x = self.linear_relu_stack(x)
            return x

    X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = 'Cef', features = ['genexp'], test_size = 0.2, standardize = True, random_state = 42)

    #transform data into torch tensors
    X_train = torch.tensor(X_train.toarray(), dtype = torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype = torch.float32)
    Y_train = torch.tensor(Y_train, dtype = torch.long)
    Y_test = torch.tensor(Y_test, dtype = torch.long)

    model = NeuralNetwork()
    train_data = TensorDataset(X_train, Y_train)

    #load dataset and define hyperparameters for model optimization
    dataloader = DataLoader(train_data, batch_size = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 100
        
    #train the model
    for i in range(epochs):
        loss = train_loop(dataloader, model, loss_fn, optimizer)
        print("Epoch: " + str(i+1) + ", loss: " + str(loss))

    #test performances on the training set
    _, _, _, _, accuracy_training = evaluate(model = model, X_test = X_train, Y_test = Y_train)

    #test performances on the test set
    precision_s, precision_r, recall_s, recall_r, accuracy = evaluate(model = model, X_test = X_test, Y_test = Y_test)

    result_table.loc[len(result_table)] = ['Cef', 'genexp', accuracy_training, precision_s, precision_r, recall_s, recall_r, accuracy]
    result_table.to_csv("pipelines/results/neural_networks/single_features/genexp.csv")


if(__name__ == '__main__'):
    neural_network_single_feature_type()
