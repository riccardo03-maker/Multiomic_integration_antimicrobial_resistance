#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd

from .ml_functions.ml_functions import weighted_train_test_split, get_non_zero_features, create_list_of_all_features, GenexpStandardizer
from .neural_networks import train_loop, evaluate

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


all_combinations_of_features = [['genexp'], ['genexp', 'snps'], ['gpa'], ['genexp', 'gpa'], ['genexp', 'gpa', 'snps'], ['gpa', 'snps'], ['snps']]
features_strings = ["genexp", "genexp+snps", "gpa", "genexp+gpa", "genexp+gpa+snps", "gpa+snps", "snps"]


class early_fusion_relevant_features_nn(nn.Module):
    '''
    This class defines a neural network with the early fusion architecture, keeping only the features with coefficients different
    from 0 in the logistic regression with Lasso regularization.
    
    It is a fully connected neural network with three hidden layers, all of them with the same number of nodes (equal to the number 
    of input features). The output layer has two nodes (corresponding to the two classes, susceptible and resistant).
    
    Parameters
    ----------
        number_of_features: list
            List with the number of input features of each type.
    '''
    def __init__(self, number_of_features: list):
        super().__init__()

        number_of_features = sum(number_of_features)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(number_of_features, number_of_features),
            nn.ReLU(),
            nn.Linear(number_of_features, number_of_features),
            nn.ReLU(),
            nn.Linear(number_of_features, number_of_features),
            nn.ReLU(),
            nn.Linear(number_of_features, 2)
        )
                    
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y


class intermediate_fusion_relevant_features_nn(nn.Module):
    '''
    This class defines a neural network with the intermediate fusion architecture, keeping only the features with coefficients different
    from 0 in the logistic regression with Lasso regularization.
    
    It takes as input all the features of the three types (gene expression, gpa and snps). The first and second hidden layers are composed by three
    separated fully connected branches, one for each feature type and each one with a number of nodes equal to the number of features
    of that type. Then the three branches are merged in the third hidden layer, which has a number of nodes equal to the sum of the
    features of the three types. The output layer has two nodes (corresponding to the two classes, susceptible and resistant).
    
    Parameters
    ----------
        number_of_features: list
            List with the number of input features of each type.
    '''
    def __init__(self, number_of_features: list):
        super().__init__()
        self.number_of_features = number_of_features

        self.genexp_branch = nn.Sequential(
            nn.Linear(number_of_features[0], number_of_features[0]),
            nn.ReLU(),
            nn.Linear(number_of_features[0], number_of_features[0]),
            nn.ReLU()
        )
        self.gpa_branch = nn.Sequential(
            nn.Linear(number_of_features[1], number_of_features[1]),
            nn.ReLU(),
            nn.Linear(number_of_features[1], number_of_features[1]),
            nn.ReLU()
        )
        self.snps_branch = nn.Sequential(
            nn.Linear(number_of_features[2], number_of_features[2]),
            nn.ReLU(),
            nn.Linear(number_of_features[2], number_of_features[2]),
            nn.ReLU()
        )
        self.merge_branches = nn.Sequential(
            nn.Linear(sum(number_of_features), sum(number_of_features)),
            nn.ReLU(),
            nn.Linear(sum(number_of_features), 2)
        )
    
    def forward(self, x):
        genexp_data = x[:, :self.number_of_features[0]]
        gpa_data = x[:, self.number_of_features[0]:(self.number_of_features[0] + self.number_of_features[1])]
        snps_data = x[:, (self.number_of_features[0] + self.number_of_features[1]):]
    
        genexp_layer = self.genexp_branch(genexp_data)
        gpa_layer = self.gpa_branch(gpa_data)
        snps_layer = self.snps_branch(snps_data)
    
        #merge the three branches into a single layer
        second_hidden_layer = torch.cat((genexp_layer, gpa_layer, snps_layer), dim=1)
        y = self.merge_branches(second_hidden_layer)
    
        return y


loss_fn = nn.CrossEntropyLoss()
neural_networks = {'early_fusion': early_fusion_relevant_features_nn, 'intermediate_fusion': intermediate_fusion_relevant_features_nn, 
                   'late_fusion': early_fusion_relevant_features_nn}
#the late fusion architecture is just composed by three early fusionneural networks, one for each type of feature. Each network takes
# a decision about the class of a sample independently, and then the class chosen by the majority of networks is chosen. 


def train_relevant_features_nn(features: list, drug: str, architecture: str):
    '''
    Train a neural network with the chosen architecture, using as input the relevant features of the selected type (those with a
    coefficient different from 0 in the logistic regression with Lasso regularization) and as output the susceptibility and resistance
    to the selected drug.

    The full dataset for the selected drug and with the selected features is splitted into a training and a test set (containing
    respectively 80% and 20% of samples chosen preserving classes proportions). Then, genexp features (if present) are standardized
    using mean and standard deviation computed on the full training set. After that, all features that have a coefficient equal to zero
    in the logistic regression with L1 regularization are discarded. Finally, the parameters of the neural network with the chosen
    architecture are optimized on the training set with the remaining features, and saved in a specific file, together with a csv file
    showing the value of the loss function across epochs.
    
    The training is performed using a cross-entropy loss function and an Adam optimizer. The learning rate is 0.001.
    
    Parameters
    ----------
        features: list of str
            The types of feature used as input for the neural network. Elements of this list can be either 'genexp', 'gpa' or 'snps'.
        drug: {'Cef', 'Cip', 'Mer', 'Tob'}
            The drug for which the neural network has to predict susceptibility or resistance.
        architecture: {'early_fusion', 'intermediate_fusion', 'late_fusion'}
            The type of neural network that is going to be trained.
    '''
    #set seed for reproducibility
    torch.manual_seed(42)

    #create dataframe for loss function during epochs
    loss_evolution = pd.DataFrame(columns = ['Epoch', 'Loss'])

    number_of_features = [] #list with the number of features for each type
    all_coefficients_dataset = pd.read_csv("pipelines/results/log_reg_coefficients.csv")

    for k, feature in enumerate(features):
        all_features = create_list_of_all_features([feature])
        relevant_features = get_non_zero_features(all_coefficients_dataset, drug, feature)
        relevant_features_indexes = [i for i, feat in enumerate(all_features) if feat in relevant_features]
        #get the positions of the relevant features in the list of all the features of a certain type

        X_train_feat, _, Y_train, _ = weighted_train_test_split(drug = drug, features = [feature], test_size = 0.2, 
                                                                                random_state = 42)

        #standardize genexp features
        standardizer = GenexpStandardizer(features = [feature])
        standardizer.fit(X_train_feat)
        X_train_feat = standardizer.transform(X_train_feat)

        X_train_feat = X_train_feat.toarray()[:, relevant_features_indexes]
        number_of_features.append(len(relevant_features_indexes))

        #concatenate different features into a unique array
        if k == 0:
            X_train = X_train_feat
        else:
            X_train = np.concatenate((X_train, X_train_feat), axis = 1)

    #transform data into torch tensors
    X_train = torch.tensor(X_train, dtype = torch.float32)
    Y_train = torch.tensor(Y_train, dtype = torch.long)
        
    model = neural_networks[architecture](number_of_features = number_of_features)
    train_data = TensorDataset(X_train, Y_train)
        
    #load dataset and define hyperparameters for model optimization
    dataloader = DataLoader(train_data, batch_size = 64)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 150
                
    #train the model
    for i in range(epochs):
        loss = train_loop(dataloader, model, loss_fn, optimizer)
        loss_evolution.loc[len(loss_evolution)] = [i+1, loss.item()]
        print("Epoch: " + str(i+1) + ", loss: " + str(loss.item()))
    
    #save the trained model
    filename = 'rf_'
    for feature in features:
        filename += feature
        filename += '_'
    filename += drug
    torch.save(model.state_dict(), "pipelines/nn_trained_models/" + architecture + "_rf/" + architecture + "_" + filename)
    loss_evolution.to_csv("pipelines/nn_trained_models/" + architecture + "_rf/loss_function/" + architecture + "_" + filename + ".csv")
    
        
def nn_relevant_features_test(architecture: str):
    '''
    Test the performances of a neural network architecture, keeping only the features with a coefficient different from 0 in the 
    logistic regression with Lasso regularization.
    
    The performance of the chosen architecture of neural network is tested for each drug and each combination of features, using the
    20% of samples that were kept in the test set and were not used for the training of the model. If present, genexp data of the test
    set are standardized using mean and standard deviation learnt on the training set.
    
    The performance is evaluated through five scores: precision of susceptible and resistant classes, recall of susceptible and
    resistant classes, and accuracy of classification.
    
    Parameters
    ----------
        architecture: {'early_fusion', 'intermediate_fusion'}
            The architecture of the neural network whose performances are being tested. The late fusion architecture has its own
            dedicated testing function.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'accuracy_training', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    all_coefficients_dataset = pd.read_csv("pipelines/results/log_reg_coefficients.csv")
    
    for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
        for j, features in enumerate(all_combinations_of_features):
            if(architecture == 'intermediate_fusion' and len(features) != 3):
                continue #in architectures different from the early fusion, all features are used

            number_of_features = [] #list with the number of features for each type
            for k, feature in enumerate(features):
                all_features = create_list_of_all_features([feature])
                relevant_features = get_non_zero_features(all_coefficients_dataset, drug, feature)
                relevant_features_indexes = [i for i, feat in enumerate(all_features) if feat in relevant_features]
                #get the positions of the relevant features in the list of all the features of a certain type
            
                X_train_feat, X_test_feat, Y_train, Y_test = weighted_train_test_split(drug = drug, features = [feature], test_size = 0.2, 
                                                                                        random_state = 42)

                #standardize genexp features for both training and test sets, using always mean and standard deviation estimated 
                # on the full training set
                standardizer = GenexpStandardizer(features = [feature])
                standardizer.fit(X_train_feat)
                X_train_feat = standardizer.transform(X_train_feat)
                X_test_feat = standardizer.transform(X_test_feat)
                
                X_train_feat = X_train_feat.toarray()[:, relevant_features_indexes]
                X_test_feat = X_test_feat.toarray()[:, relevant_features_indexes]
                number_of_features.append(len(relevant_features_indexes))
            
                #concatenate different features into a unique array
                if k == 0:
                    X_train = X_train_feat
                    X_test = X_test_feat
                else:
                    X_train = np.concatenate((X_train, X_train_feat), axis = 1)
                    X_test = np.concatenate((X_test, X_test_feat), axis = 1)
            
            #transform data into torch tensors
            X_train = torch.tensor(X_train, dtype = torch.float32)
            X_test = torch.tensor(X_test, dtype = torch.float32)
            Y_train = torch.tensor(Y_train, dtype = torch.long)
            Y_test = torch.tensor(Y_test, dtype = torch.long)
    
            #create the model and load the weights obtained during training
            model = neural_networks[architecture](number_of_features = number_of_features)
            filename = 'rf_'
            for feature in features:
                filename += feature
                filename += '_'
            filename += drug
            model.load_state_dict(torch.load("pipelines/nn_trained_models/" + architecture + "_rf/" + architecture + "_" + filename, weights_only = True))
            model.eval()
    
            #test performances on the training set
            _, _, _, _, accuracy_training = evaluate(model = model, X_test = X_train, Y_test = Y_train)
        
            #test performances on the test set
            precision_s, precision_r, recall_s, recall_r, accuracy = evaluate(model = model, X_test = X_test, Y_test = Y_test)
        
            result_table.loc[len(result_table)] = [drug, features_strings[j], accuracy_training, precision_s, precision_r, recall_s, recall_r, accuracy]
            print("Iteration")
    
    result_table.to_csv("pipelines/results/neural_networks/" + architecture + "_rf_scores.csv")


def late_fusion_relevant_features_test():
    '''
    Test the performances of a neural network with the late fusion architecture, keeping only the features with a coefficient different from 0 in the 
    logistic regression with Lasso regularization.

    This function has the same role of the nn_relevant_features_test function, but it is applied on late fusion architecture. In this case, three independent
    classifications are performed for each sample using three networks trained on the three different feature types, then the sample is
    assigned to the class chosen by the majority of the three networks.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    all_coefficients_dataset = pd.read_csv("pipelines/results/log_reg_coefficients.csv")
    
    for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
        predicted_classes = []
        for features in [['genexp'], ['gpa'], ['snps']]:
            number_of_features = [] #list with the number of features for each type

            for k, feature in enumerate(features):
                all_features = create_list_of_all_features([feature])
                relevant_features = get_non_zero_features(all_coefficients_dataset, drug, feature)
                relevant_features_indexes = [i for i, feat in enumerate(all_features) if feat in relevant_features]
                #get the positions of the relevant features in the list of all the features of a certain type
                
                X_train_feat, X_test_feat, Y_train, Y_test = weighted_train_test_split(drug = drug, features = [feature], test_size = 0.2, 
                                                                                        random_state = 42)

                #standardize genexp features for both training and test sets, using always mean and standard deviation estimated 
                # on the full training set
                standardizer = GenexpStandardizer(features = [feature])
                standardizer.fit(X_train_feat)
                X_train_feat = standardizer.transform(X_train_feat)
                X_test_feat = standardizer.transform(X_test_feat)
                
                X_train_feat = X_train_feat.toarray()[:, relevant_features_indexes]
                X_test_feat = X_test_feat.toarray()[:, relevant_features_indexes]
                number_of_features.append(len(relevant_features_indexes))
                
                #concatenate different features into a unique array
                if k == 0:
                    X_train = X_train_feat
                    X_test = X_test_feat
                else:
                    X_train = np.concatenate((X_train, X_train_feat), axis = 1)
                    X_test = np.concatenate((X_test, X_test_feat), axis = 1)
                
            #transform data into torch tensors
            X_train = torch.tensor(X_train, dtype = torch.float32)
            X_test = torch.tensor(X_test, dtype = torch.float32)
            Y_train = torch.tensor(Y_train, dtype = torch.long)
            Y_test = torch.tensor(Y_test, dtype = torch.long)
        
            #create the model and load the weights obtained during training
            model = early_fusion_relevant_features_nn(number_of_features = number_of_features)
            model.load_state_dict(torch.load("pipelines/nn_trained_models/late_fusion_rf/late_fusion_rf_" + features[0] + "_" + drug,
                                              weights_only = True))
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

    result_table.to_csv("pipelines/results/neural_networks/late_fusion_rf_scores.csv")