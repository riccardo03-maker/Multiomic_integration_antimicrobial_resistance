#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_array, load_npz, hstack, vstack
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def weighted_train_test_split(drug: str, features: list, test_size: float, random_state: int = None):
    '''
    Given a drug and one or more set of features (gene expression, gpa or snps), this function divides those features data 
    into train and test set, keeping in both sets the same proportion between susceptible and resistant to a certain drug. 

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

    #merge data for the required set of features into a single sparse matrix
    for i, feature in enumerate(Counter(features).keys()): #keep only unique values of the list of features, to avoid issues with repetitions
        if feature not in ['genexp', 'gpa', 'snps']:
            raise ValueError("Set of features chosen is not one of the possible choices")
        if i==0:
            #initialize the csr_array of input features
            features = load_npz("./transformed_data/features/" + feature + "_features.npz")
        else:
            #from the second iteration, stack new csr_array of input features column-wise (horizontally)
            features = hstack([features, load_npz("./transformed_data/features/" + feature + "_features.npz")], format = "csr")
    
    #Divide features dataset into strains susceptible and resistant to the chosen drug
    X_susceptible = features[Y_susceptible["Index"].tolist()]
    X_resistent = features[Y_resistent["Index"].tolist()]
        
    X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(X_susceptible, Y_susceptible, test_size=test_size, random_state=random_state)
    X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(X_resistent, Y_resistent, test_size=test_size, random_state=random_state)

    X_train=vstack([X_train_s, X_train_r], format = "csr")
    X_test=vstack([X_test_s, X_test_r], format = "csr")
    
    #separate output classes from "Index" and "Strain" columns
    Y_train_full=pd.concat([Y_train_s, Y_train_r]) #three columns: drug, index and strain
    Y_test_full=pd.concat([Y_test_s, Y_test_r])

    Y_train = np.array(Y_train_full[drug], dtype = np.float64)
    Y_test = np.array(Y_test_full[drug], dtype = np.float64)
    strains_train = Y_train_full[["Index", "Strain"]]
    strains_test = Y_test_full[["Index", "Strain"]]

    return X_train, X_test, Y_train, Y_test, strains_train, strains_test