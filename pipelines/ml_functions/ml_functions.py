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


class GenexpStandardizer():
    '''
    This class is implemented as a scikit-learn Transformer. Similarly to a StandardScaler of scikit-learn, it standardizes the training
    set while learning its mean and standard deviation, then it can apply the same standardization (with same mean and standard deviation)
    to the test/validation set.

    However, differently from a StandardScaler, ths standardization is applied only on gene expression data (which are already standardized,
    but before the division between train and test sets, so train and test sets are not really independent). On the contrary, gpa and 
    snps data don't need to be standardized because they are binary data (either 0 or 1).

    Parameters
    ----------
        features: list of str
            The list of feature types (genexp, gpa, snps) present in the training set that will be used to learn standardization parameters.
            If 'genexp' is not in this list, the current instance of this object will leave input data completely unchanged.
    '''
    def __init__(self, features: list):
        self.features = features

    def fit(self, X, y = None):
        '''
        Compute the mean and std of genexp data to be used for later scaling.

        Parameters
        ----------
            X: np.ndarray or scipy.sparse.csr_array
                Data from which to compute mean and standard deviation of genexp data to use for later scaling.
            y: None
                Ignored (only for scikit-learn compatibility)
        '''
        if 'genexp' not in self.features:
            return self
        else:
            #divide genexp data from others
            genexp_data = X[:, :6026]

            #compute mean and standard deviation
            if isinstance(genexp_data, csr_array):
                genexp_data = genexp_data.toarray()
            self.mean = np.mean(genexp_data, axis = 0, dtype = np.float64)
            self.std = np.std(genexp_data, axis = 0, dtype = np.float64)
            return self
            

    def transform(self, X):
        '''
        Perform standardization on genexp data, by centering and scaling

        Parameters
        ----------
            X: np.ndarray or scipy.sparse.csr_array
                Matrix for which we want to standardize genexp data along the feature axis.
        Returns
        -------
            X_transformed: np.ndarray or scipy.sparse.csr_array
                Input array after the standardization of genexp data.
        '''
        if 'genexp' not in self.features:
            return X
        else:
            #divide genexp data from others
            genexp_data = X[:, :6026]
            other_data = X[:, 6026:]

            #subtract mean and divide by standard deviation
            if isinstance(genexp_data, csr_array):
                genexp_data = genexp_data.toarray()
                genexp_data = (genexp_data - self.mean)/self.std
                genexp_data = csr_array(genexp_data)
                X = hstack([genexp_data, other_data], format = 'csr')
            else:
                genexp_data = (genexp_data - self.mean)/self.std
                X = np.hstack([genexp_data, other_data])

            return X


def weighted_train_test_split(drug: str, features: list, test_size: float, random_state: int = None, full_Y: bool = False):
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
        full_Y: bool (default: False)
            If True, returns the output classes datasets (Y_train and Y_test) as pandas.DataFrame, keeping also the strains names and their
            indexes. If False, returns the output classes datasets as numpy.ndarray, removing indexes and strains and keeping only 
            the classes for the selected drug.
    Returns
    -------
        X_train, X_test: csr_array
            Training and test sets of input features.
        Y_train, Y_test: np.ndarray or pd.Dataframe
            Training and test sets of output classes.
    Raises
    ------
        ValueError:
            If the input drug is not one of the four possible choices.
        ValueError:
            If the input list of features has a value which is not one of the three possible choices.
    '''
    if drug not in ['Tob', 'Cef', 'Cip', 'Mer']:
        raise ValueError("Drug chosen is not one of the possible choices")
    
    #keep only the drug chosen
    classes = pd.read_csv("./transformed_data/classes/classes.csv")
    columns = [c for c in classes.columns if c in ["Index", "Strain", drug]]
    classes = classes[columns]

    #remove NaN values for the drug chosen
    classes = classes.dropna(subset = drug)

    #divide between s and r
    Y_susceptible = classes[classes[drug] < 0.5]
    Y_resistant = classes[classes[drug] > 0.5]

    #first split only the classes dataset, the input features will be splitted separately
    Y_train_s, Y_test_s = train_test_split(Y_susceptible, test_size = test_size, random_state = random_state)
    Y_train_r, Y_test_r = train_test_split(Y_resistant, test_size = test_size, random_state = random_state)

    #unify susceptible and resistant samples
    Y_train_full = pd.concat([Y_train_s, Y_train_r]) #three columns: drug, index and strain name
    Y_test_full = pd.concat([Y_test_s, Y_test_r])

    #get the indexes of train and test samples, to split also input data
    train_indexes = Y_train_full["Index"].tolist()
    test_indexes = Y_test_full["Index"].tolist()

    #choose whether to return a pandas dataframe or a numpy ndarray
    if full_Y:
        Y_train = Y_train_full
        Y_test = Y_test_full
    else:
        Y_train = np.array(Y_train_full[drug], dtype = np.float64)
        Y_test = np.array(Y_test_full[drug], dtype = np.float64)

    for i, feature in enumerate(Counter(features).keys()): #keep only unique values of the list of features, to avoid issues with repetitions
        if feature not in ['genexp', 'gpa', 'snps']:
            raise ValueError("Set of features chosen is not one of the possible choices")
        
        #create a sparse matrix for each group of features
        new_features = load_npz("./transformed_data/features/" + feature + "_features.npz")
        
        #divide into train and test, using the same division of the output classes
        new_features_train = new_features[train_indexes]
        new_features_test = new_features[test_indexes]
        
        #append different features data together
        if i == 0:
            X_train = new_features_train
            X_test = new_features_test
        else:
            X_train = hstack([X_train, new_features_train], format = "csr")
            X_test = hstack([X_test, new_features_test], format = "csr")

    return X_train, X_test, Y_train, Y_test


def get_non_zero_features(data: pd.DataFrame, drug: str, feature: str):
    '''
    This function is used to obtain all the features of a certain type with a coefficient different from zero in the logistic 
    regression with Lasso regularization.

    Parameters
    ----------
        data: pd.DataFrame
            The dataset with the coefficients of all the features obtained in the logistic regression with Lasso regularization.
            The dataset is given as input instead of being load into the function because it is a large dataset, and opening it
            requires time, so this allows to open it only once when this function is called several times in a single execution.
        drug: {'Cef', 'Cip', 'Mer', 'Tob'}
            The selected drug.
        feature: {'genexp', 'gpa' or 'snps'}
            The type of feature required.
    Returns
    -------
        relevant_features: list of str
            The list of features of the required type with coefficients different from 0 in the logistic regression.
    '''
    transposed = data.T.iloc[1:] #now drugs are columns and features are rows
    transposed.columns = ['Feature_type', 'Cef', 'Cip', 'Mer', 'Tob']
    data = transposed[['Feature_type', drug]].query(drug + ' != "0.0" and Feature_type == "' + feature + '"')
    #select only the rows where the coefficient for the selected drug is not 0
    
    relevant_features = data.T.columns
    return relevant_features


def create_list_of_all_features(features: list):
    '''
    Create a list of all the features used to train machine learning models.

    Given a list of feature types (gene expression, gpa and snps), this function creates a list with all the features of the required types.
    
    Parameters
    ----------
        features: list of str
            The types of features that will be included in the returned list. The elements in the list must be either 'genexp', 'gpa' or 'snps'. If an 
            element in the list is not one of these three, the code raises a ValueError.
    Returns
    -------
        features_list: list of str
            The list of all the features of the types given as input. The order of the features in the output list depends on the order
            types were given as input.
    '''
    features_list = []
    for feature in Counter(features).keys():
        if feature not in ['genexp', 'gpa', 'snps']:
            raise ValueError("Set of features chosen is not one of the possible choices")
        
        #read the features from the list in the raw data
        with open("raw_data/features_gpa_expr_snps/" + feature + "/" + feature + "_feature_list.txt") as file:
            new_features = list(file)
            new_features = [feat.rstrip("\n") for feat in new_features]
            #eliminate "\n" characters that are at the end of each feature in the list
        features_list += new_features
    return features_list