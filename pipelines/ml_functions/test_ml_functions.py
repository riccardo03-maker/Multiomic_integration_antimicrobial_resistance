from ml_functions import weighted_train_test_split, get_non_zero_features, create_list_of_all_features, GenexpStandardizer
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
    (there are 8 NaN for tobramycin). At the same time, train and test output classes sets have one column each, and a total of 406 rows.
    Input features sets must be scipy.sparse.csr_array objects, while output classes sets must be numpy.ndarray objects.
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
    (there are 8 NaN for tobramycin). At the same time, train and test output classes sets have three columns each, and a total of 406 rows.
    Input features sets must be scipy.sparse.csr_array objects, while output classes sets must be pandas.DataFrame objects
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
    Test the correct standardization of gene expression data.

    GIVEN: input features relative to gene expression.
    WHEN: I split data into train and test sets, and after the splitting I use the GenexpStandardizer to standardize genexp features.
    THEN: after standardization, each gene expression feature has 0 mean and 1 standard deviation across all samples of the training 
    set, while before standardization mean and std are different from 0 and 1.
    '''
    X_train, _, _, _ = weighted_train_test_split(drug = 'Tob', features = ['genexp'], test_size = 0.2, random_state = 42)

    #check mean is not 0 and std is not 1 at the beginning
    train_mean = np.mean(X_train.toarray(), axis = 0)
    train_std = np.std(X_train.toarray(), axis = 0)
    assert(not (np.isclose(train_mean, 0., atol = 0.000001)).all())
    assert(not (np.isclose(train_std, 1., atol = 0.000001)).all())

    standardizer = GenexpStandardizer(features = ['genexp'])
    standardizer.fit(X_train)
    X_train = standardizer.transform(X_train)
    X_train = X_train.toarray()
    
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)

    assert((np.isclose(train_mean, 0., atol = 0.000001)).all())
    assert((np.isclose(train_std, 1., atol = 0.000001)).all())


def test_no_standardization_for_other_data():
    '''
    Test that gpa or snps data are not standardized (they are binary data).

    GIVEN: input features relative to gpa.
    WHEN: I split data into train and test sets, with standardization of gene expression data after the splitting.
    THEN: each feature of gpa has either the value 0 or 1 for each sample, meaning that data have not been standardized. 
    So train and test sets contain just two different values.
    '''
    X_train, _, _, _ = weighted_train_test_split(drug = 'Tob', features = ['gpa'], test_size = 0.2, random_state = 42)
    standardizer = GenexpStandardizer(features = ['gpa'])
    standardizer.fit(X_train)
    X_train = standardizer.transform(X_train)
    X_train = X_train.toarray()

    assert(len(np.unique(X_train)) == 2)

    #check mean and standard deviations are different from 0 and 1
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    assert(not (np.isclose(train_mean, 0., atol = 0.000001)).all())
    assert(not (np.isclose(train_std, 1., atol = 0.000001)).all())


# Testing list of all features with non zero coefficients in logistic regression

def test_list_of_non_zero_features():
    '''
    Test the correct creation of the list of features with a coefficient different from 0 in the logistic regression

    GIVEN: data with all the coefficients for all features in logistic regression, for all drugs
    WHEN: I create the list of all the features with coefficients different from 0 for the Ceftazidim drug, for all the three types of
    features
    THEN: the total number of features of the three types is 78 (counted using R)
    '''
    data = pd.read_csv("pipelines/results/log_reg_coefficients.csv")
    relevant_features_genexp = get_non_zero_features(data = data, drug = 'Cef', feature = 'genexp')
    relevant_features_gpa = get_non_zero_features(data = data, drug = 'Cef', feature = 'gpa')
    relevant_features_snps = get_non_zero_features(data = data, drug = 'Cef', feature = 'snps')
    assert(len(relevant_features_genexp) + len(relevant_features_gpa) + len(relevant_features_snps) == 78)


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
