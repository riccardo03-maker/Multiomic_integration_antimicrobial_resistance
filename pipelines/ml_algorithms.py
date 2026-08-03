#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from .ml_functions.ml_functions import weighted_train_test_split, create_list_of_all_features
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_array

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


all_combinations_of_features = [['genexp'], ['genexp', 'snps'], ['gpa'], ['genexp', 'gpa'], ['genexp', 'gpa', 'snps'], ['gpa', 'snps'], ['snps']]
features_strings = ["genexp", "genexp+snps", "gpa", "genexp+gpa", "genexp+gpa+snps", "gpa+snps", "snps"]
drugs = ['Cef', 'Cip', 'Mer', 'Tob']


def pca():
    '''
    Implement a PCA for all the three types of features, to see if two principal components are enough to split samples correctly
    into the two classes. All samples are used for PCA, without dividing into train and test sets.
    '''
    for drug in drugs:
        #create datasets of input features and output classes, but without dividing into train and test sets
        classes = pd.read_csv("./transformed_data/classes/classes.csv")
        columns = [c for c in classes.columns if c in ["Index", "Strain", drug]]
        classes = classes[columns]
        classes=classes.dropna(subset=drug)

        #get the indexes of the remaining samples (those without NA for the drug considered in this iteration)
        indexes_to_keep = classes["Index"]

        for feature in ['genexp', 'gpa', 'snps']:
            features = load_npz("./transformed_data/features/" + feature + "_features.npz")
            features = features[indexes_to_keep]
            if feature == 'genexp': #standardize
                features = scale(features.toarray()) #standardization cannot be done using sparse matrices, so we convert into np.ndarray
                features = csr_array(features)

            pca = PCA(n_components = 2, random_state = 42)
            pca.fit(features)
            samples_projected = pca.transform(features)

            classes.insert(len(classes.columns), feature + "_1", samples_projected[:, 0])
            classes.insert(len(classes.columns), feature + "_2", samples_projected[:, 1])

            print("Iteration")
        
        classes.to_csv("pipelines/results/pca/pca_" + drug + ".csv")


#dictionary with all machine learning models used
ml_models = {'svc': SVC, 'log_reg': LogisticRegression, 'knn': KNeighborsClassifier, 'lda': LinearDiscriminantAnalysis,
             'svm_paper': LinearSVC}


def cross_validate_model(model_name: str, **kwargs):
    '''
    Given a certain machine learning model, this function implements a cross-validation pipeline to select, for each drug, 
    the combination of features with the best classification performances.

    For each drug and each combination of feature types (genexp, gpa and snps), the full dataset is divided into a train and
    a test set, using 80% of samples for training and 20% for testing, keeping the same proportions between samples susceptible and 
    resistant to a certain drug. Then, a 5-fold cross-validation is applied on the training set. The train set is divided into five 
    folds, and at each iteration of the cross-validation a score is calculated by training the machine learning model on four folds 
    and validating it on the remaining one. So, for each cross-validation we get five scores. The cross-validation is repeated five 
    times, and at the end the score assigned to a certain combination of features for a certain drug is given by the mean of the 25
    scores obtained in this way.

    The cross-validation scores are calculated using the f1 macro score.

    The hyperparameters of the selected model are not optimized using cross-validation, but they can be given as input instead.

    The classification performances obtained for each drug and each combination of features are saved in a csv file.

    Parameters
    ----------
        model_name: {'svc', 'log_reg', 'knn', 'lda', 'svm_paper'}
            The name of the machine learning model whose performances are evaluated through cross-validation. 'svm_paper' corresponds
            to the support vector classification implemented in the paper from which data are taken.
        **kwargs:
            Optional parameters for the machine learning model selected.
    '''
    scores_array = np.zeros(shape = (7, 4), dtype = np.float64)
    standard_deviation_array = np.zeros(shape = (7, 4), dtype = np.float64)
    
    for i, drug in enumerate(drugs):
        for j, features in enumerate(all_combinations_of_features):
    
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, random_state  = 42)
            #use a seed for random train-test splitting, so that for both this function and the model_performance_test function
            #the samples used for training and testing are always the same for the same drug

            if model_name == 'lda':
                X_train = X_train.toarray()
            #linear discriminant analysis does not work on sparse data
    
            model = ml_models[model_name](**kwargs)
            cv_scores = np.array([], dtype = np.float64)
    
            for k in range(5):
                #Perform a 5-fold cross validation 5 times, and average over all the scores
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=k)
                single_iteration_scores = cross_val_score(model, X=X_train, y=Y_train, cv=cv, scoring = 'f1_macro', n_jobs = 1)
                cv_scores = np.concatenate((cv_scores, single_iteration_scores)) 
    
            scores_array[j][i] = cv_scores.mean()
            standard_deviation_array[j][i] = cv_scores.std()
            print("Iteration")

    if model_name == 'svc':
        model_name = model_name + '_' + kwargs['kernel']
        #include also the kernel in the final output file name for support vector classification
        
    scores_data = pd.DataFrame(data = scores_array, index = features_strings, columns = drugs)
    std_data = pd.DataFrame(data = standard_deviation_array, index = features_strings, columns = drugs)
    scores_data.to_csv("./pipelines/results/ml_algorithms/" + model_name + "/" + model_name + "_cv_scores.csv")
    std_data.to_csv("./pipelines/results/ml_algorithms/" + model_name + "/" + model_name + "_cv_std.csv")


def model_performance_test(model_name: str, **kwargs):
    '''
    Given a certain machine learning model, this function evaluates its classification performances.

    For each drug, the combination of features with the highest score obtained through cross-validation is selected. Then the full
    dataset with those features is split into train and test sets, using the same random state of the cross_validate_model function,
    so that the samples ending in the train or in the test set are always the same for a certain drug. In this way the test set is
    completely independent from the samples used for cross-validation.

    The selected machine learning model is then trained on the full training set, and its classification performances evaluated on the test
    set. Performances are evaluated through five scores: precision (for both susceptible and resistant classes), recall (again for both
    classes), and accuracy of classification.

    All the scores are saved in a csv file.

    Parameters:
    ----------
        model_name: {'svc', 'log_reg', 'knn', 'lda', 'svm_paper'}
            The machine learning model whose performances are going to be tested. 'svm_paper' corresponds to the support vector 
            classification implemented in the paper from which data are taken.
        **kwargs:
            Optional parameters for the machine learning model selected.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])

    #include the kernel in the name of the model to open the file with cv scores
    if model_name == 'svc':
        model_name_for_file_opening = model_name + '_' + kwargs['kernel']
    else:
        model_name_for_file_opening = model_name

    cv_scores_table = pd.read_csv("./pipelines/results/ml_algorithms/" + model_name_for_file_opening + "/" 
                                  + model_name_for_file_opening + "_cv_scores.csv")

    for drug in drugs:
        best_features_index = cv_scores_table[drug].idxmax()
        best_combination_of_features = all_combinations_of_features[best_features_index]

        X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = best_combination_of_features, 
                                                                     test_size = 0.2, random_state  = 42)
        #use a seed for random train-test splitting, so that for both this function and the cross_validate_model function
        #the samples used for training and testing are always the same for the same drug
        
        if model_name == 'lda':
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            #linear discriminant analysis does not work on sparse data
            
        model = ml_models[model_name](**kwargs)
        model.fit(X_train, Y_train)            
        Y_predict = model.predict(X_test)

        result_table.loc[len(result_table)] = [drug, features_strings[best_features_index], precision_score(Y_test, Y_predict, pos_label = 0), 
                                                precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0),
                                                recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
        print("Iteration")

    result_table.to_csv("pipelines/results/ml_algorithms/" + model_name_for_file_opening + "/" + model_name_for_file_opening
                        + "_test_scores.csv")


def get_logistic_regression_coefficients():
    '''
    Implement a logistic regression with Lasso regularization, and get the coefficients of all the features.

    For each drug and for each type of feature (genexp, gpa and snps), the full dataset of input features is split into a training 
    and a test set. Then, the model is trained on the whole train set, and all the coefficients of the features are saved in a csv
    file.

    Each row of the dataset stored in the csv file corresponds to a drug, and each column to a feature. The entries represent the coefficient
    of a feature in the logistic regression fitted using as input all the features of the type to which that feature belongs.
    '''
    coefficients = pd.DataFrame(columns = create_list_of_all_features(['genexp', 'gpa', 'snps']))
    #give a name to the features in the final dataset with the coefficients

    what_type_of_features = ['genexp'] * 6026 + ['gpa'] * 16005 + ['snps'] * 72236
    coefficients.loc[0] = what_type_of_features
    #create a row that tells what type is the feature that gives the name to the column

    for drug in drugs:
        coefficients_array = np.array([[]])
        for feature in ['genexp', 'gpa', 'snps']: 
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = [feature], test_size = 0.2, standardize = True,
                                                                               random_state  = 42)
            log_reg = LogisticRegression(C = 0.1, l1_ratio = 1.0, tol = 1e-6, solver = 'liblinear', class_weight = 'balanced',
                                         random_state = 42)
            #l1_ratio 1 is the l1 regularization

            log_reg.fit(X_train, Y_train)

            coefficients_array = np.concatenate((coefficients_array, log_reg.coef_), axis = 1)
            print("Iteration")
        coefficients.loc[len(coefficients)] = coefficients_array[0] #use [0] because coefficients are stored as a column vectors

    coefficients.to_csv("pipelines/results/log_reg_coefficients.csv")