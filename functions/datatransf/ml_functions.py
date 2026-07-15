#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import load_npz, hstack, csr_array
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def weighted_train_test_split(drug: str, features: list, test_size: float, standardize: bool = False, random_state: int = None):
    '''
    Given a drug and one or more set of features (gene expression, gpa or snps), this function divides those features data 
    into train and test set, keeping in both sets the same proportion between susceptible and resistant to a certain drug.

    If required, this function can also standardize gene expression data (they are already standardized, but before the division
    between train and test sets, so train and test data are not really independent). Gpa and snps data don't need to be standardized
    because they are binary data (either 0 or 1).

    Parameters
    ----------
        drug: {'Tob', 'Cef', 'Cip', 'Mer'}
            The drug that will be considered to compute the percentages of susceptible and resistant. If the drug chosen is not one
            of the possible choices, the code raises a ValueError.
        features: list of strings
            The features that will be used as input data. The elements in the list must be either 'genexp', 'gpa' or 'snps'. If an 
            element in the list is not one of these three, the code raises a ValueError.
        test_size: float
            The percentage of dataset that will be used as test set.
        standardize: bool (default: False)
            If True, standardize gene expression data after the train-test division.
        random_state: int (default: None)
            Random seed for reproducibility.
    Returns
    -------
        X_train, X_test: csr_array
            Training and test sets of input features.
        Y_train, Y_test: pd.Dataframe
            Training and test sets of output classes.
        strains_train, strains_test: pd.DataFrame
            The strains (and relative indexes) present in train and test sets, respectively.
    Raises
    ------
        ValueError:
            If the input drug is not one of the four possible choices.
        ValueError:
            If the input list of features has a value which is not one of the three possible choices.
    '''
    if drug not in ['Tob', 'Cef', 'Cip', 'Mer']:
        raise ValueError("Drug chosen is not one of the possible choices")
    
    #keep only the index column and the column for the drug chosen in the dataset of targets
    targets = pd.read_csv("./transformed_data/targets/targets.csv")
    columns = [c for c in targets.columns if c in ["Index", "Strain", drug]]
    targets = targets[columns]

    #remove NaN values for the drug chosen
    targets=targets.dropna(subset=drug)

    #divide between s and r
    Y_susceptible = targets[targets[drug]<0.5]
    Y_resistent = targets[targets[drug]>0.5]

    #first split only the targets dataset, the input features will be splitted separately, so that gene expression data can be standardized
    #after the division
    Y_train_s, Y_test_s = train_test_split(Y_susceptible, test_size=test_size, random_state=random_state)
    Y_train_r, Y_test_r = train_test_split(Y_resistent, test_size=test_size, random_state=random_state)

    #separate output classes from "Index" and "Strain" columns
    Y_train_full=pd.concat([Y_train_s, Y_train_r]) #three columns: drug, index and strain
    Y_test_full=pd.concat([Y_test_s, Y_test_r])

    Y_train = np.array(Y_train_full[drug], dtype = np.float64)
    Y_test = np.array(Y_test_full[drug], dtype = np.float64)
    strains_train = Y_train_full[["Index", "Strain"]]
    strains_test = Y_test_full[["Index", "Strain"]]

    #get the indexes of train and test samples, to split also input data
    train_indexes = strains_train["Index"].tolist()
    test_indexes = strains_test["Index"].tolist()

    for i, feature in enumerate(Counter(features).keys()): #keep only unique values of the list of features, to avoid issues with repetitions
        if feature not in ['genexp', 'gpa', 'snps']:
            raise ValueError("Set of features chosen is not one of the possible choices")
        
        #create a sparse matrix for each group of features
        new_features = load_npz("./transformed_data/features/" + feature + "_features.npz")
        
        #divide into train and test, using the same division of the output classes
        new_features_train = new_features[train_indexes]
        new_features_test = new_features[test_indexes]

        if standardize and feature == 'genexp':
            new_features_train = scale(new_features_train.toarray()) #standardization cannot be done using sparse matrices
            new_features_test = scale(new_features_test.toarray()) #so we convert into np.ndarray
            new_features_train = csr_array(new_features_train)
            new_features_test = csr_array(new_features_test)
        
        #append different features data together
        if i == 0:
            X_train = new_features_train
            X_test = new_features_test
        else:
            X_train = hstack([X_train, new_features_train], format = "csr")
            X_test = hstack([X_test, new_features_test], format = "csr")

    return X_train, X_test, Y_train, Y_test, strains_train, strains_test