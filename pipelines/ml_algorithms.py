#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap

from ml_functions.ml_functions import weighted_train_test_split, _get_non_zero_features, create_list_of_all_features, _get_number_of_samples_by_class
import numpy as np
import pandas as pd
import math
from scipy.sparse import load_npz, csr_array
from collections import Counter

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


all_combinations_of_features = [['genexp'], ['genexp', 'snps'], ['gpa'], ['genexp', 'gpa'], ['genexp', 'gpa', 'snps'], ['gpa', 'snps'], ['snps']]
features_strings = ["genexp", "genexp+snps", "gpa", "genexp+gpa", "genexp+gpa+snps", "gpa+snps", "snps"]
drugs = ['Cef', 'Cip', 'Mer', 'Tob']


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
    c_params = [[0.025, 0.01, 0.04, 0.015, 0.01, 0.04, 0.025], [0.02, 0.02, 0.04, 0.02, 0.015, 0.05, 0.07],
            [0.025, 0.025, 0.1, 0.04, 0.15, 0.07, 0.02], [0.085, 0.015, 0.04, 0.03, 0.015, 0.07, 0.07]]
    
    scores_array = np.zeros(shape = (7, 4), dtype = np.float64)

    for j, drug in enumerate(drugs):
        for i, features in enumerate(all_combinations_of_features):

            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, random_state  = 42)
            #use a seed for random number generation, so that the splitting between train and test set is always
            #the same for the same drug, and results can be compared more easily

            svm_model = LinearSVC(penalty = 'l1', loss = 'squared_hinge', max_iter = 1000000, tol = 0.000001,
                                      class_weight = "balanced", dual = False, random_state = 1, C = c_params[j][i])
            cv_scores = np.zeros(5)

            for k in range(5):
                #Perform a 5-fold cross validation 5 times, and average over the different results
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=k)
                cv_score = cross_val_score(svm_model, X=X_train, y=Y_train, cv=cv, scoring = 'f1_macro', n_jobs = 1)
                cv_scores[k] = cv_score.mean()

            scores_array[i][j] = cv_scores.mean()
    
    scores_data = pd.DataFrame(data = scores_array, index = features_strings, columns = drugs)
    scores_data.to_csv("./ml_algorithms/results/svm/svm_paper_cv.csv")


def svm_paper_test(standardize: bool = False):
    '''
    Implement a linear SVC pipeline to predict antimicrobial resistance, using for each drug the best combination of data obtained through
    cross-validation procedure. Based on the paper original paper.

    First this algorithm splits the dataset (where the input features are those selected through cross-validation) into train and test sets
    with respectively 80% and 20% of data, keeping the same proportion between isolates susceptible and resistent to the selected drug.

    Then the model is trained on the training set. The value of the C parameter for the SVC is the best parameter for the combination
    of features used, found by the original paper through a nested cross-validation procedure (the best C value is the one with the least
    number of features inside one standard deviation from the value with the best performance).

    Finally, the performance of the model is evaluated on the separate test set.

    Parameters:
        standardize: bool (default: False)
            If True, data about gene expression are standardized before training the model (data are already standardized, but
            the standardization has been done before splitting between train and test set).

            Data about gpa and snps are not standardized, since they are binary data.
    '''
    features = [['genexp', 'gpa'], ['snps'], ['genexp', 'gpa'], ['genexp', 'gpa']]
    #use only the best combinations of features (obtained in the paper)

    c_params = [0.015, 0.07, 0.04, 0.03]

    for i in range(4):
        X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drugs[i], features = features[i], test_size = 0.2, 
                                                    standardize = standardize, random_state  = 42)

        svm_model = LinearSVC(penalty = 'l1', loss = 'squared_hinge', max_iter = 1000000, tol = 0.000001,
                                    class_weight = "balanced", dual = False, random_state = 1, C = c_params[i])
        svm_model.fit(X_train, Y_train)
        Y_predict = svm_model.predict(X_test)
        #print("F1 score macro " + drugs[i] + ": " + str(f1_score(Y_test, Y_predict, average = 'macro')))
        print("Accuracy " + drugs[i] + ": " + str(accuracy_score(Y_test, Y_predict)))


def logistic_regression():
    '''
    Implement a logistic regression to predict antimicrobial resistance.

    It uses an l1 (Lasso) regularization, so that as many features as possible are ignored
    For each drug and each combination of features, the full dataset is split into a training and a test set. Then, the model is fitted
    on the whole train set and its performances evaluated on the test set, using as scores precision and recall for both classes
    (susceptible (0) and resistent (1)). Moreover, the number of samples in the test set classified as either suceptible or resistent 
    are counted and compared with the real number of susceptible and resistent samples.

    The best value of C (which is the inverse of the regularization strength) is selected through a 5-fold cross-validation procedure
    applied on the train set (by ranking performances based on the accuracy).

    Note: for now we use just one type of feature (genexp, gpa or snps). So for each drug we have a fit for three times.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'best C', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])
    
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
            log_reg = LogisticRegressionCV(cv = 5, Cs = 10, l1_ratios=[1.0], max_iter = 10000, tol = 1e-6, random_state=len(result_table),
                                         solver = 'liblinear', use_legacy_attributes = False, scoring = 'accuracy')
            #l1_ratio 1 is the l1 regularization
            log_reg.fit(X_train, Y_train)
            
            Y_predict = log_reg.predict(X_test)

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            result_table.loc[len(result_table)] = [drug, feature, log_reg.C_, precision_score(Y_test, Y_predict, pos_label = 0), 
                                                   precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0), 
                                                   recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, feature, X_test.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]

            coefficients_array = np.concatenate((coefficients_array, log_reg.coef_), axis = 1)
            print("Iteration")
        coefficients.loc[len(coefficients)] = coefficients_array[0] #use [0] because coefficients are stored as a column vectors

    result_table.to_csv("ml_algorithms/results/logistic_regression/log_reg_scores.csv")
    count_table.to_csv("ml_algorithms/results/logistic_regression/log_reg_counts.csv")
    coefficients.to_csv("ml_algorithms/results/logistic_regression/log_reg_coefficients.csv")


def logistic_regression_with_feature_selection():
    '''
    Implement a logistic regression using only the features with a coefficient different from 0 in the logistic regression performed
    with the previous function.

    This function is identical to the previous one: it uses a logistic regression with Lasso regularization, selecting the best value of C
    through a cross-validation. However, it only uses some of the features (those selected in the previous logistic regression function).

    Moreover, the logistic regression for each drug is performed 4 times, once for each combination of two or more types of features
    (genexp+gpa, gpa+snps, genexp+snps and genexp+gpa+snps). The aim is to see if the combination of two or more types of features in
    the logistic regression can improve classification performances.
    '''
    all_features = create_list_of_all_features(['genexp', 'gpa', 'snps'])
    result_table = pd.DataFrame(columns = ['drug', 'features', 'best C', 'train samples', 'features used', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])

    for drug in drugs:
        for j in [1, 3, 4, 5]: #repeat for all combinations of two or more feature types
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = ['genexp', 'gpa', 'snps'], test_size = 0.2, 
                                                    standardize = True, random_state  = 42)
            #select the positions of the relevant features
            relevant_features, relevant_features_types = _get_non_zero_features(drug)
            relevant_features_indexes = [i for i, feat in enumerate(all_features) if feat in relevant_features]

            #keep only the features relative to the types considered in this iteration loop
            relevant_features_indexes = [index for k, index in enumerate(relevant_features_indexes) 
                                     if relevant_features_types[k] in all_combinations_of_features[j]]
        
            #keep only the relevant features in the input features datasets
            X_train = X_train[:, relevant_features_indexes]
            X_test = X_test[:, relevant_features_indexes]

            log_reg = LogisticRegressionCV(cv = 5, Cs = 10, l1_ratios=[1.0], max_iter = 10000, tol = 1e-6, random_state = 42,
                                         solver = 'liblinear', use_legacy_attributes = False, scoring = 'accuracy')
            #l1_ratio 1 is the l1 regularization
            log_reg.fit(X_train, Y_train)
            
            Y_predict = log_reg.predict(X_test)

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            #get all the features with coefficients different from 0
            number_of_features_used = len([coefficient for coefficient in log_reg.coef_[0] if not math.isclose(coefficient, 0., abs_tol = 1e-15)])

            result_table.loc[len(result_table)] = [drug, features_strings[j], log_reg.C_, X_train.shape[0], number_of_features_used, 
                                                precision_score(Y_test, Y_predict, pos_label = 0), precision_score(Y_test, Y_predict, pos_label = 1), 
                                                recall_score(Y_test, Y_predict, pos_label = 0), recall_score(Y_test, Y_predict, pos_label = 1), 
                                                accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, features_strings[j], X_test.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]
            
            print("Iteration")

    result_table.to_csv("ml_algorithms/results/logistic_regression/log_reg_relevant_features_scores.csv")
    count_table.to_csv("ml_algorithms/results/logistic_regression/log_reg_relevant_features_counts.csv")


def linear_discriminant_analysis():
    '''
    Implement a linear discriminant analysis to predict antimicrobial resistance.

    The pipeline is repeated for each drug and for each combination of input features. The LDA is performed using the singular value
    decomposition, since the calculation of the covariance matrix would require a lot of time because of the huge number of features.

    The classification performances are evaluated using the same scores applied in the logistic regression. Moreover, the number of
    samples in the test set classified as either suceptible or resistent are counted and compared with the real number of susceptible
    and resistent samples.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])

    for drug in drugs:
        for j, features in enumerate(all_combinations_of_features):
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, 
                                                    standardize = True, random_state  = 42)
            lda = LinearDiscriminantAnalysis(solver = 'svd')
            lda.fit(X_train.toarray(), Y_train)
            Y_predict = lda.predict(X_test.toarray())

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            result_table.loc[len(result_table)] = [drug, features_strings[j], precision_score(Y_test, Y_predict, pos_label = 0), 
                                                   precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0),
                                                    recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, features_strings[j], X_test.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]
            #0 is susceptible and 1 is resistent

            print("Iteration")
    
    result_table.to_csv("ml_algorithms/results/lda/lda_scores.csv")
    count_table.to_csv("ml_algorithms/results/lda/lda_counts.csv")


def quadratic_discriminant_analysis():
    '''
    Implement a quadratic discriminant analysis to predict antimicrobial resistance.

    The pipeline is repeated for each drug and for each combination of input features. Before applying the QDA, a dimensionality reduction
    is required to calculate and invert the covariance matrices. The dimensionality reduction is done here using PCA, keeping 200 components.

    The classification performances are evaluated using the same scores applied in the logistic regression. Moreover, the number of
    samples in the test set classified as either suceptible or resistent are counted and compared with the real number of susceptible
    and resistent samples.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])

    for drug in drugs:
        for j, features in enumerate(all_combinations_of_features):
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, 
                                                    standardize = True, random_state  = 42)
            
            #apply dimensionality reduction through PCA
            pca = PCA(n_components = 200)
            pca.fit(X_train)
            X_train_projected = pca.transform(X_train)
            X_test_projected = pca.transform(X_test)

            lda = QuadraticDiscriminantAnalysis(solver = 'eigen', shrinkage = 'auto')
            lda.fit(X_train_projected, Y_train)
            Y_predict = lda.predict(X_test_projected)

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            result_table.loc[len(result_table)] = [drug, features_strings[j], precision_score(Y_test, Y_predict, pos_label = 0), 
                                                   precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0),
                                                    recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, features_strings[j], X_test_projected.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]
            #0 is susceptible and 1 is resistent

            print("Iteration")
    
    result_table.to_csv("ml_algorithms/results/lda/qda_scores.csv")
    count_table.to_csv("ml_algorithms/results/lda/qda_counts.csv")


def knn_performance():
    '''
    Measure the performance of the K-Nearest Neighbours algorithm for different values of K.

    For each drug and each combination of features, a 5-fold cross-validation is performed for different values of K. The accuracy
    score is computed as the average of the accuracies computed at each iteration of the cross-validation. In this way the best value
    of K is selected for each combination of features.
    '''
    best_values_of_k = pd.DataFrame(columns=['drug', 'features', 'best_k'])
    for drug in drugs:
        for j, features in enumerate(all_combinations_of_features):
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, 
                                                    standardize = True, random_state  = 42)
            
            performances = {}
            
            for k in [5, 8, 10, 12, 15, 18, 20]:
                knn = KNeighborsClassifier(n_neighbors=k)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=k)
                cv_scores = cross_val_score(knn, X=X_train, y=Y_train, cv=cv, scoring = 'accuracy', n_jobs = 1)
                performances.update({cv_scores.mean() : k})
            
            best_performance = max(performances.keys())
            best_k = performances[best_performance]
            best_values_of_k.loc[len(best_values_of_k)] = [drug, features_strings[j], best_k]

            print("Iteration")
    
    best_values_of_k.to_csv("ml_algorithms/results/knn/best_values_of_k.csv")


def knn():
    '''
    Implement a knn pipeline using the best values of k obtained through the knn_performance function.

    The pipeline is repeated for each drug and each combination of features. The classification performances are evaluated using the 
    same scores applied in the logistic regression. Moreover, the number of samples in the test set classified as either suceptible 
    or resistent are counted and compared with the real number of susceptible and resistent samples.
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])

    best_k_values = pd.read_csv("ml_algorithms/results/knn/best_values_of_k.csv")['best_k'].to_list()
    iteration_number = 0

    for drug in drugs:
        for j, features in enumerate(all_combinations_of_features):
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, test_size = 0.2, 
                                                    standardize = True, random_state  = 42)

            knn = KNeighborsClassifier(n_neighbors = best_k_values[iteration_number])
            knn.fit(X_train, Y_train)
            Y_predict = knn.predict(X_test)

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            result_table.loc[len(result_table)] = [drug, features_strings[j], precision_score(Y_test, Y_predict, pos_label = 0), 
                                                   precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0),
                                                    recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, features_strings[j], X_test.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]
            #0 is susceptible and 1 is resistent

            print("Iteration")
            iteration_number += 1
    
    result_table.to_csv("ml_algorithms/results/knn/knn_scores.csv")
    count_table.to_csv("ml_algorithms/results/knn/knn_counts.csv")


def svc(kernel: str, C: float, gamma: float = 'scale', degree: int = 3):
    '''
    Implement the support vector classification with different kernels to predict antimicrobial resistance. 
    This does not take into account the methods of the reference paper.

    Parameters
    ----------
        kernel: str
            The kernel used for SVC. This parameter is directly passed to the SVC object of scikit-learn, so the accepted values
            can be found in the sklearn.svm.SVC documentation.
        C: float
            The inverse of the regularization parameter. This parameter is directly passed to the SVC object of scikit-learn.
        gamma: float or str (default: 'auto')
            The parameter gamma for the polynomial, Gaussian and sigmoidal kernels. This parameter is directly passed to the SVC object
            of scikit-learn, so the accepted values can be found in the sklearn.svm.SVC documentation.
        degree: int (default: 3)
            The degree of the polynomial kernel function. This parameter is directly passed to the SVC object of scikit-learn.
    References
    ----------
        sklearn.svm.SVC documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    '''
    result_table = pd.DataFrame(columns = ['drug', 'features', 'precision_s', 'precision_r', 'recall_s', 'recall_r', 'accuracy'])
    count_table = pd.DataFrame(columns = ['drug', 'features', 'test samples', 'predicted_s', 'real_s', 'predicted_r', 'real_r'])

    for drug in drugs:
        for j, features in enumerate(all_combinations_of_features):
            X_train, X_test, Y_train, Y_test = weighted_train_test_split(drug = drug, features = features, 
                                                    test_size = 0.2, standardize = True, random_state  = 42)

            svc = SVC(C = C, kernel = kernel, tol = 1e-6, degree = degree, gamma = gamma)
            svc.fit(X_train, Y_train)
            Y_predict = svc.predict(X_test)

            #get the total number of predicted and real susceptible and resistent samples
            predicted_classes, real_classes = _get_number_of_samples_by_class(Y_predict, Y_test)

            result_table.loc[len(result_table)] = [drug, features_strings[j], precision_score(Y_test, Y_predict, pos_label = 0), 
                                                   precision_score(Y_test, Y_predict, pos_label = 1), recall_score(Y_test, Y_predict, pos_label = 0),
                                                    recall_score(Y_test, Y_predict, pos_label = 1), accuracy_score(Y_test, Y_predict)]
            count_table.loc[len(count_table)] = [drug, features_strings[j], X_test.shape[0], predicted_classes[0],
                                                 real_classes[0], predicted_classes[1], real_classes[1]]
            #0 is susceptible and 1 is resistent

            print("Iteration")
    
    result_table.to_csv("ml_algorithms/results/svc/svc_scores_" + kernel + ".csv")
    count_table.to_csv("ml_algorithms/results/svc/svc_counts_" + kernel + ".csv")


def pca():
    '''
    Implement a PCA for all the three types of features, to see if two principal components are enough to split samples correctly
    into the two classes.
    '''
    for drug in drugs:
        #create datasets of input features and output targets, but without dividing into train and test sets
        targets = pd.read_csv("./transformed_data/targets/targets.csv")
        columns = [c for c in targets.columns if c in ["Index", "Strain", drug]]
        targets = targets[columns]
        targets=targets.dropna(subset=drug)

        #get the indexes of the remaining samples (those without NA for the drug considered in this iteration)
        indexes_to_keep = targets["Index"]

        for feature in ['genexp', 'gpa', 'snps']:
            features = load_npz("./transformed_data/features/" + feature + "_features.npz")
            features = features[indexes_to_keep]
            if feature == 'genexp': #standardize
                features = scale(features.toarray()) #standardization cannot be done using sparse matrices, so we convert into np.ndarray
                features = csr_array(features)

            pca = PCA(n_components = 2)
            pca.fit(features)
            samples_projected = pca.transform(features)

            targets.insert(len(targets.columns), feature + "_1", samples_projected[:, 0])
            targets.insert(len(targets.columns), feature + "_2", samples_projected[:, 1])

            print("Iteration")
        
        targets.to_csv("ml_algorithms/results/pca/pca_" + drug + ".csv")


def isomap():
    '''
    Implement a Isomap for all the three types of features, to see if two components are enough to split samples correctly
    into the two classes.
    '''
    for drug in drugs:
        #create datasets of input features and output targets, but without dividing into train and test sets
        targets = pd.read_csv("./transformed_data/targets/targets.csv")
        columns = [c for c in targets.columns if c in ["Index", "Strain", drug]]
        targets = targets[columns]
        targets=targets.dropna(subset=drug)

        #get the indexes of the remaining samples (those without NA for the drug considered in this iteration)
        indexes_to_keep = targets["Index"]

        for feature in ['genexp', 'gpa', 'snps']:
            features = load_npz("./transformed_data/features/" + feature + "_features.npz")
            features = features[indexes_to_keep]
            if feature == 'genexp': #standardize
                features = scale(features.toarray()) #standardization cannot be done using sparse matrices, so we convert into np.ndarray
                features = csr_array(features)

            isomap = Isomap(n_components = 2, n_neighbors = 20)
            isomap.fit(features)
            samples_projected = isomap.embedding_

            targets.insert(len(targets.columns), feature + "_1", samples_projected[:, 0])
            targets.insert(len(targets.columns), feature + "_2", samples_projected[:, 1])

            print("Iteration")
        
        targets.to_csv("ml_algorithms/results/dim_reduction/isomap/isomap_" + drug + ".csv")


if(__name__ == '__main__'):
    svc(kernel = 'poly', C = 0.1, degree = 3, gamma = 1)


