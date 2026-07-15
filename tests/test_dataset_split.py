from datatransf import weighted_train_test_split
import pytest
from scipy.sparse import csr_array
import pandas as pd
import numpy as np

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def test_incorrect_drug_input():
    '''
    Test the correct raise of a ValueError when the argument provided for the parameter 'drug' is not one of the possible choices.

    GIVEN: I am trying to split data into train and test sets.
    WHEN: I choose as target a drug which is not one of the four possible choices.
    THEN: the code raises a ValueError.
    '''
    with pytest.raises(ValueError):
        weighted_train_test_split(drug = "wrong_drug", features = ['genexp'], test_size = 0.2)


def test_incorrect_features_input():
    '''
    Test the correct raise of a ValueError when one of the elements in the list provided as argument for the parameter 'features' 
    is not one of the possible choices.

    GIVEN: I am trying to split data into train and test sets.
    WHEN: I give a list of features cointaining one element which is not one of the three possible choices.
    THEN: the code raises a ValueError.
    '''
    with pytest.raises(ValueError):
        weighted_train_test_split(drug = 'Tob', features = ['genexp', 'gpa', 'wrong_feature'], test_size = 0.2)


def test_correct_train_test_split():
    '''
    Test the correct behaviour of the weighted_train_test_split function, when trying to create train and test dataset, where the input
    features are data about gene expression and gpa, while output classes are susceptibility or resistance to tobramycin.

    GIVEN: input features about gene expression and gpa, and output classes relative to tobramycin susceptibility or resistance.
    WHEN: I split these data into train and test sets.
    THEN: both train and test input features sets have 22031 columns, and the sum of the number of rows of train and test sets is 406
    (there are 8 NaN for tobramycin). At the same time, train and test output targets sets have one column each, and a total of 406 rows.
    Finally, list of strains of train and test sets must have two columns each, and a total of 406 rows.
    Input features sets must be scipy.sparse.csr_array objects, output targets sets must be numpy.ndarray objects, and list of strains
    must be pandas.DataFrame objects.
    '''
    X_train, X_test, Y_train, Y_test, strains_train, strains_test = weighted_train_test_split(drug = 'Tob', features = ['genexp', 'gpa'], test_size = 0.2)

    assert(isinstance(X_train, csr_array))
    assert(isinstance(X_test, csr_array))
    assert(isinstance(Y_train, np.ndarray))
    assert(isinstance(Y_test, np.ndarray))
    assert(isinstance(strains_train, pd.DataFrame))
    assert(isinstance(strains_test, pd.DataFrame))

    assert(X_train.shape[0] + X_test.shape[0] == 406)
    assert(X_train.shape[1] == 22031) 
    assert(X_test.shape[1] == 22031)

    assert(Y_train.shape[0] + Y_test.shape[0] == 406)
    
    #test that an error occurs when asking for the second element of the shape tuple of the array, meaning that
    #the array is one-dimensional
    with pytest.raises(IndexError):
        Y_train.shape[1]
    with pytest.raises(IndexError):
        Y_test.shape[1]

    assert(strains_train.shape[0] + strains_test.shape[0] == 406)
    assert(strains_train.shape[1] == 2)
    assert(strains_test.shape[1] == 2)


def test_correct_standardization():
    '''
    Test the correct standardization of gene expression data, both in the train and test sets.

    GIVEN: input features relative to gene expression.
    WHEN: I split data into train and test sets, with standardization of gene expression data after the splitting.
    THEN: each feature of gene expression data has 0 mean and 1 standard deviation.
    '''
    X_train, X_test, _, _, _, _ = weighted_train_test_split(drug = 'Tob', features = ['genexp'], test_size = 0.2, standardize = True, random_state = 42)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    test_mean = np.mean(X_test, axis = 0)
    test_std = np.std(X_test, axis = 0)

    assert((np.isclose(train_mean, 0., atol = 0.000001)).all())
    assert((np.isclose(test_mean, 0., atol = 0.000001)).all())
    assert((np.isclose(train_std, 1., atol = 0.000001)).all())
    assert((np.isclose(test_std, 1., atol = 0.000001)).all())


def test_no_standardization_for_other_data():
    '''
    Test that gpa or snps data are not standardized (they are binary data).

    GIVEN: input features relative to gpa.
    WHEN: I split data into train and test sets, with standardization of gene expression data after the splitting.
    THEN: each feature of gpa has either the value 0 or 1 for each sample (isolate). So train and test set contain just two different values
    '''
    X_train, X_test, _, _, _, _ = weighted_train_test_split(drug = 'Tob', features = ['gpa'], test_size = 0.2, standardize = True, random_state = 42)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    assert(len(np.unique(X_train)) == 2)
    assert(len(np.unique(X_test)) == 2)
