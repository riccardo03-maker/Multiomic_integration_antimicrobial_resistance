#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn import svm
from datatransf import weighted_train_test_split
import numpy as np
import pandas as pd

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']

all_combinations_of_features = [['genexp'], ['genexp', 'snps'], ['gpa'], ['genexp', 'gpa'], ['genexp', 'gpa', 'snps'], ['gpa', 'snps'], ['snps']]
features_strings = ["genexp", "genexp+snps", "gpa", "genexp+gpa", "genexp+gpa+snps", "gpa+snps", "snps"]
c_params = [[0.025, 0.01, 0.04, 0.015, 0.01, 0.04, 0.025], [0.02, 0.02, 0.04, 0.02, 0.015, 0.05, 0.07]
            , [0.025, 0.025, 0.1, 0.04, 0.15, 0.07, 0.02], [0.085, 0.015, 0.04, 0.03, 0.015, 0.07, 0.07]]

def svm_paper_cv():
    '''
    Implement a cross-validation pipeline to find the best combination of data to predict antimicrobial resistance. Based on the original paper.
    It perfoms cross-validation to find, for each drug, the combination of features (gene expression, gpa and snps) that gives the best
    predictive performances

    This algorithm uses the linear Support Vector Classification (SVC), with an L1 regularization and an L2 loss. The full dataset with
    the selected features (the procedure is repeated for each drug and for each combination of features) is divided into a train and a
    test set with respectively 80% and 20% of data, keeping the same proportion between isolates susceptible and resistent to the
    selected drug.

    Then the performance of the linear SVC is evaluated on the train dataset through a 5-fold cross validation. The score used to evaluate
    the classification performances is the macro f1 score (in particular the average of the macro f1 scores obtained in each iteration
    of the cross-validation loop). The cross-validation is repeated 5 times for each drug and each combination of data, and the final score
    is given by the average over the scores obtained in the 5 repetitions.

    In the paper, the optimal value of the C parameter used in the linear SVC is found to a nested cross-validation. Here we don't repeat
    the nested cross-validation, which is computationally expensive, and we just use the best value of C found in the analysis performed
    by the authors of the paper.

    References:
    -----------
        Best values of C: https://github.com/hzi-bifo/Predicting_PA_AMR_paper/blob/master/feature_curves/best_models.txt
    '''
    scores_array = np.zeros(shape = (7, 4), dtype = np.float64)

    for j, drug in enumerate(['Cef', 'Cip', 'Mer', 'Tob']):
        for i, features in enumerate(all_combinations_of_features):

            X_train, X_test, Y_train, Y_test, _, _ = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, random_state  = 42)
            #use a seed for random number generation, so that the splitting between train and test set is always
            #the same for the same drug, and results can be compared more easily

            svm_model = svm.LinearSVC(penalty = 'l1', loss = 'squared_hinge', max_iter = 1000000, tol = 0.000001,
                                      class_weight = "balanced", dual = False, random_state = 1, C = c_params[j][i])
            cv_scores = np.zeros(5)

            for k in range(5):
                #Perform a 10-fold cross validation 5 times, and average over the different results
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=k)
                cv_score = cross_val_score(svm_model, X=X_train, y=Y_train, cv=cv, scoring = 'f1_macro', n_jobs = 1)
                cv_scores[k] = cv_score.mean()

            scores_array[i][j] = cv_scores.mean()
    
    scores_data = pd.DataFrame(data = scores_array, index = features_strings, columns = ['Cef', 'Cip', 'Mer', 'Tob'])
    scores_data.to_csv("./ml_algorithms/results/svm_paper_cv.csv")


if(__name__ == '__main__'):
    svm_paper_cv()


