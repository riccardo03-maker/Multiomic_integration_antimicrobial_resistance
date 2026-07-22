from ml_functions import weighted_train_test_split, _get_non_zero_features, _get_number_of_samples_by_class, create_list_of_all_features
import pytest
from scipy.sparse import csr_array
import pandas as pd
import numpy as np

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


#Testing train test splitting


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
    Input features sets must be scipy.sparse.csr_array objects, while output targets sets must be numpy.ndarray objects.
    '''
    X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = 'Tob', features = ['genexp', 'gpa'], test_size = 0.2)

    assert(isinstance(X_train, csr_array))
    assert(isinstance(X_test, csr_array))
    assert(isinstance(Y_train, np.ndarray))
    assert(isinstance(Y_test, np.ndarray))

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

    
def test_correct_train_test_split_full_y():
    '''
    This test is identical to the previous one, except that the parameter 'full_Y' of the weighted_train_test_split function is True.

    GIVEN: input features about gene expression and gpa, and output classes relative to tobramycin susceptibility or resistance.
    WHEN: I split these data into train and test sets, with the parameter 'full_Y' = True.
    THEN: both train and test input features sets have 22031 columns, and the sum of the number of rows of train and test sets is 406
    (there are 8 NaN for tobramycin). At the same time, train and test output targets sets have three columns each, and a total of 406 rows.
    Input features sets must be scipy.sparse.csr_array objects, while output targets sets must be pandas.DataFrame objects
    '''
    X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = 'Tob', features = ['genexp', 'gpa'], test_size = 0.2, full_Y = True)

    assert(isinstance(X_train, csr_array))
    assert(isinstance(X_test, csr_array))
    assert(isinstance(Y_train, pd.DataFrame))
    assert(isinstance(Y_test, pd.DataFrame))

    assert(X_train.shape[0] + X_test.shape[0] == 406)
    assert(X_train.shape[1] == 22031) 
    assert(X_test.shape[1] == 22031)

    assert(Y_train.shape[0] + Y_test.shape[0] == 406)
    
    assert(Y_train.shape[1] == 3)
    assert(Y_test.shape[1] == 3)


def test_correct_standardization():
    '''
    Test the correct standardization of gene expression data, both in the train and test sets.

    GIVEN: input features relative to gene expression.
    WHEN: I split data into train and test sets, with standardization of gene expression data after the splitting.
    THEN: each feature of gene expression data has 0 mean and 1 standard deviation.
    '''
    X_train, X_test, _, _ = weighted_train_test_split(drug = 'Tob', features = ['genexp'], test_size = 0.2, standardize = True, random_state = 42)
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
    X_train, X_test, _, _ = weighted_train_test_split(drug = 'Tob', features = ['gpa'], test_size = 0.2, standardize = True, random_state = 42)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    assert(len(np.unique(X_train)) == 2)
    assert(len(np.unique(X_test)) == 2)


# Testing list of all features with non zero coefficients in logistic regression

def test_list_of_non_zero_features():
    '''
    Test the correct creation of the list of features with a coefficient different from 0 in the logistic regression

    GIVEN: the data with all the coefficients for all features in logistic regression, for all drugs
    WHEN: I create the list of all the features with coefficients different from 0 for the Ceftazidim drug
    THEN: I obtain a list of 261 elements (counted using R)
    '''
    relevant_features, relevant_features_types = _get_non_zero_features(drug = 'Cef')
    assert(len(relevant_features) == 261)
    assert(len(relevant_features_types) == 261)


# Testing count of assignments to susceptible and resistent classes


def test_count_class_assignment():
    '''
    Test the correct count of the number of samples assigned to each class after a classification performance.

    GIVEN: a test set with 20 samples, 13 belonging to class 0 (susceptible) and 7 belonging to class 1 (resistent), and a set
    of predicted classes with 9 samples classified as susceptible and 11 classified as resistent.
    WHEN: I count the number of samples in each class for both test set and predicted set.
    THEN: the count returns the correct values.
    '''
    real_classes = [0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.]
    predicted_classes = [0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1.]

    predict_counts, real_counts = _get_number_of_samples_by_class(predicted_classes, real_classes)

    assert(predict_counts[0] == 9)
    assert(predict_counts[1] == 11)
    assert(real_counts[0] == 13)
    assert(real_counts[1] == 7)


def test_all_samples_in_one_class():
    '''
    Test the correct behaviour of the _get_number_of_samples_by_class function when predicted classes of all the samples is the same.

    GIVEN: a test set with 20 samples, 13 belonging to class 0 (susceptible) and 7 belonging to class 1 (resistent), and a set
    of predicted classes with all samples classified as resistent.
    WHEN: I count the number of samples in each class for both test set and predicted set.
    THEN: the count returns the correct values.
    '''
    real_classes = [0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.]
    predicted_classes = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

    predict_counts, real_counts = _get_number_of_samples_by_class(predicted_classes, real_classes)

    assert(predict_counts[0] == 0)
    assert(predict_counts[1] == 20)
    assert(real_counts[0] == 13)
    assert(real_counts[1] == 7)


# Testing creation list of all features

def test_list_of_all_features():
    '''
    Test the correct creation of the list of all features, for all the three types of features (gene expression, gpa and snps).

    GIVEN: I am creating a list of features.
    WHEN: I give as input all the three types of features.
    THEN: the number of elements in the final list is 94267, and the first element of gpa is in position 6026, while the first element of 
    snps is in position 22031.
    '''
    features_list = create_list_of_all_features(['genexp', 'gpa', 'snps'])

    assert(len(features_list) == 94267)
    assert(features_list[6026] == ",,aacA4|1")
    assert(features_list[22031] == "PA14_03290_298867_A_G_T_A|10")
