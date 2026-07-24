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


all_combinations_of_features = [['genexp'], ['genexp', 'snps'], ['gpa'], ['genexp', 'gpa'], ['genexp', 'gpa', 'snps'], ['gpa', 'snps'], ['snps']]
features_strings = ["genexp", "genexp+snps", "gpa", "genexp+gpa", "genexp+gpa+snps", "gpa+snps", "snps"]


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


def evaluate(model: nn.Module, X_test: torch.tensor, Y_test: torch.tensor, full_tensor: bool = False):
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
        full_tensor: bool (default: False)
            If True, returns the tensor with the predicted classes for each sample. Otherwise, the function returns the classification
            scores.
    Returns
    -------
        Y_predict: np.ndarray
            The array with the predicted classes for all samples (returned only if full_tensor is True).
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
    if full_tensor:
        return Y_predict

    Y_test = np.array(Y_test.numpy(), dtype = np.int32)

    precision_s = precision_score(Y_test, Y_predict, pos_label = 0)
    precision_r = precision_score(Y_test, Y_predict, pos_label = 1)
    recall_s = recall_score(Y_test, Y_predict, pos_label = 0)
    recall_r = recall_score(Y_test, Y_predict, pos_label = 1)
    accuracy = accuracy_score(Y_test, Y_predict)
    return precision_s, precision_r, recall_s, recall_r, accuracy


class early_fusion_nn(nn.Module):
    '''
    This class defines a neural network with the early fusion architecture. The neural network can use just one type of input features,
    or a combination of them. In this latter case different types of input features are combined immediately in the first layer.

    It is a fully connected neural network with two hidden layers: the first layer has 300 nodes while the second hidden layer 
    has 99 nodes. The output layer has two nodes (corresponding to the two classes, susceptible and resistent).

    The number of nodes in the first hidden layer is much smaller than the number of input features to have dimensionality reduction
    (otherwise the network would bee too big to be stored in memory).

    Parameters
    ----------
        number_of_features: int
            The number of input features.
    '''
    def __init__(self, number_of_features: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(number_of_features, 300),
        nn.ReLU(),
        nn.Linear(300, 99),
        nn.ReLU(),
        nn.Linear(99, 2)
    )
                
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y


class intermediate_fusion_nn(nn.Module):
    '''
    This class defines a neural network with the intermediate fusion architecture.

    It takes as input all the features of the three types (gene expression, gpa and snps). The first hidden layer is composed by three
    separated fully connected branches of 100 nodes, one for each type of feature. Then the three branches are merged in the second hidden
    layer, which has 99 nodes. The output layer has two nodes (corresponding to the two classes, susceptible and resistent).

    So, for each type of features a dimensionality reduction is applied before concatenating it with the other types.

    Parameters
        ----------
            number_of_features: int
                The number of input features. This parameter is irrelevant for the intermediate fusion architecture, but it is
                relevant for compatibility in the train_nn and test_nn functions.
    '''
    def __init__(self, number_of_features: int):
        super().__init__()
        self.genexp_branch = nn.Sequential(
            nn.Linear(6026, 100),
            nn.ReLU()
        )
        self.gpa_branch = nn.Sequential(
            nn.Linear(16005, 100),
            nn.ReLU()
        )
        self.snps_branch = nn.Sequential(
            nn.Linear(72236, 100),
            nn.ReLU()
        )
        self.merge_branches = nn.Sequential(
            nn.Linear(300, 99),
            nn.ReLU(),
            nn.Linear(99, 2)
        )

    def forward(self, x):
        genexp_data = x[:, :6026]
        gpa_data = x[:, 6026:22031]
        snps_data = x[:, 22031:]

        genexp_layer = self.genexp_branch(genexp_data)
        gpa_layer = self.gpa_branch(gpa_data)
        snps_layer = self.snps_branch(snps_data)

        #merge the three branches into a single layer
        first_hidden_layer = torch.cat((genexp_layer, gpa_layer, snps_layer), dim=1)
        y = self.merge_branches(first_hidden_layer)

        return y


class late_fusion_nn(nn.Module):
    '''
    This class defines a neural network with the late fusion architecture.

    Baiscally it has a similar structure to the early fusion architecture, excpet for the fact that there are 100 nodes in the first hidden layer
    and 33 in the second hidden layer.

    To classify samples into susceptible or resistent to a certain drug, three independent networks of this type are trained, one for
    each type of feature. Then each network will give its own class for the sample, and that sample will be assigned to the class chosen
    by the majority of the three networks.

    Parameters
    ----------
        number_of_features: int
            The number of input features (depends on the type of input features used).
    '''
    def __init__(self, number_of_features: int):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(number_of_features, 100),
        nn.ReLU(),
        nn.Linear(100, 33),
        nn.ReLU(),
        nn.Linear(33, 2)
    )
                    
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y


loss_fn = nn.CrossEntropyLoss()
neural_networks = {'early_fusion' : early_fusion_nn, 'intermediate_fusion': intermediate_fusion_nn, 'late_fusion': late_fusion_nn}


def train_nn(features: list, drug: str, architecture: str):
    '''
    Train a neural network with the chosen architecture, using as input all the features of the selected type and as output the
    susceptibility and resistance to the selected drug. The weights of the model are then saved, ready to use for a future testing
    of the model performances.

    The training is performed using a cross-entropy loss function and an Adam optimizer. The learning rate is 0.001.

    Parameters
    ----------
        features: list of str
            The types of features used as input for the neural network. Elements of this list can be either 'genexp', 'gpa' or 'snps'.
        drug: str
            The drug for which the neural network has to predict susceptibility or resistance. Must be either 'Cef', 'Cip', 'Mer' or 'Tob'.
        architecture: str
            The type of neural network that is going to be trained.
    '''
    X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, standardize = True, random_state = 42)
    number_of_features = X_train.shape[1]

    #transform data into torch tensors
    X_train = torch.tensor(X_train.toarray(), dtype = torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype = torch.float32)
    Y_train = torch.tensor(Y_train, dtype = torch.long)
    Y_test = torch.tensor(Y_test, dtype = torch.long)
    
    model = neural_networks[architecture](number_of_features = number_of_features)
    train_data = TensorDataset(X_train, Y_train)
    
    #load dataset and define hyperparameters for model optimization
    dataloader = DataLoader(train_data, batch_size = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 100
            
    #train the model
    for i in range(epochs):
        loss = train_loop(dataloader, model, loss_fn, optimizer)
        print("Epoch: " + str(i+1) + ", loss: " + str(loss))

    #save the trained model
    filename = ''
    for feature in features:
        filename += feature
        filename += '_'
    filename += drug
    torch.save(model.state_dict(), "pipelines/nn_trained_models/" + architecture + "/" + architecture + "_" + filename)


def nn_test(architecture: str):
    '''
    Test the performances of a neural network architecture.

    The performance of the chosen architecture of neural network is tested for each drug and each combination of features, using the
    20% of samples that were kept in the test set and were not used for the training of the model. The performance is evaluated through
    five scores: precision of susceptible and resistent classes, recall of susceptible and resistent classes, and accuracy of classification.

    Parameters
    ----------
        architecture: str
            The architecture of the neural network whose performances are being tested
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'accuracy_training', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])

    for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
        for j, features in enumerate(all_combinations_of_features):
            if(architecture != 'early_fusion' and len(features) != 3):
                continue #in architectures different from the early fusion, all features are used

            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, standardize = True, random_state = 42)
            #using always the same random state for the train-test splitting ensures that the samples used for training and testing are
            #always the same in all functions (for a certain drug and a certain combination of features)
            number_of_features = X_train.shape[1]

            #transform data into torch tensors
            X_train = torch.tensor(X_train.toarray(), dtype = torch.float32)
            X_test = torch.tensor(X_test.toarray(), dtype = torch.float32)
            Y_train = torch.tensor(Y_train, dtype = torch.long)
            Y_test = torch.tensor(Y_test, dtype = torch.long)

            #create the model and load the weights obtained during training
            model = neural_networks[architecture](number_of_features = number_of_features)
            filename = ''
            for feature in features:
                filename += feature
                filename += '_'
            filename += drug
            model.load_state_dict(torch.load("pipelines/nn_trained_models/" + architecture + "/" + architecture + "_" + filename, weights_only=True))
            model.eval()

            #test performances on the training set
            _, _, _, _, accuracy_training = evaluate(model = model, X_test = X_train, Y_test = Y_train)
    
            #test performances on the test set
            precision_s, precision_r, recall_s, recall_r, accuracy = evaluate(model = model, X_test = X_test, Y_test = Y_test)
    
            result_table.loc[len(result_table)] = [drug, features_strings[j], accuracy_training, precision_s, precision_r, recall_s, recall_r, accuracy]
            print("Iteration")

    result_table.to_csv("pipelines/results/neural_networks/" + architecture + "/" + architecture + "_scores.csv")


def late_fusion_nn_test():
    '''
    Test the performances of a neural network with the late fusion architecture.

    This function has the same role of the nn_test function, but it is applied on late fusion architecture. In this case, three independent
    classifications are performed for each sample using three networks trained on the three different feature types, then the sample is
    assigned to the class chosen by the majority of the three networks
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])

    for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
        predicted_classes = [] #list with the arrays of predicted classes for each combination of features
        for features in [['genexp'], ['gpa'], ['snps']]:
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, standardize = True, random_state = 42)
            #using always the same random state for the train-test splitting ensures that the samples used for training and testing are
            #always the same in all functions (for a certain drug and a certain combination of features)
            number_of_features = X_train.shape[1]
            
            #transform data into torch tensors
            X_train = torch.tensor(X_train.toarray(), dtype = torch.float32)
            X_test = torch.tensor(X_test.toarray(), dtype = torch.float32)
            Y_train = torch.tensor(Y_train, dtype = torch.long)
            Y_test = torch.tensor(Y_test, dtype = torch.long)
            
            #create the model and load the weights obtained during training
            model = late_fusion_nn(number_of_features = number_of_features)
            model.load_state_dict(torch.load("pipelines/nn_trained_models/late_fusion/late_fusion_" + features[0] + "_" + drug, weights_only=True))
            model.eval()

            Y_predict = evaluate(model = model, X_test = X_test, Y_test = Y_test, full_tensor = True)
            predicted_classes.append(Y_predict)

        #combine the predictions by just summing them: if the sum is 2 or 3, the sample will be assigned to class 1, while if
        #the sum is 1 or 0, the sample will be assigned to class 0
        predicted_classes = np.array(predicted_classes, dtype = np.int32)
        combine_predictions = np.sum(predicted_classes, axis = 0)
        combine_predictions = np.array(combine_predictions >= 2, dtype = np.int32)

        #Y_test is always the same for the same drug, it is not needed to create it again using weighted_train_test_split with all features
        result_table.loc[len(result_table)] = [drug, "genexp+gpa+snps", precision_score(Y_test, combine_predictions, pos_label = 0), 
                                               precision_score(Y_test, combine_predictions, pos_label = 1), recall_score(Y_test, combine_predictions, pos_label = 0),
                                               recall_score(Y_test, combine_predictions, pos_label=1), accuracy_score(Y_test, combine_predictions)]
        print("Iteration")

    result_table.to_csv("pipelines/results/neural_networks/late_fusion/late_fusion_scores.csv")


if(__name__ == '__main__'):
    #for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
        #for features in [['genexp'], ['gpa'], ['snps']]:
            #train_nn(features = features, drug = drug, architecture = 'late_fusion')
    late_fusion_nn_test()