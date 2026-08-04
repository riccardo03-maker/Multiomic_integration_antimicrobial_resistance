#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pipelines import ml_algorithms
from pipelines import neural_networks
from pipelines import relevant_features_nn

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


if(__name__ == '__main__'):
    '''
    Implement a command line interface that allows the execution of all the machine learning pipelines of the project and 
    the re-creation of the plots.
    '''
    parser = argparse.ArgumentParser()#formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--algorithm', '-a',
        dest = 'algorithm',
        required = False,
        action = 'store',
        default = None,
        help = '''The machine learning algorithm to be executed. By default, for classical machine learning algorithms (so all options except for the three fusion architectures), only the scores obtained on the test set are calculated. To compute also cross-validation scores, use the --cross flag.
                    By default, the neural network architectures use all the features. To use only the relevant features, use the --rf flag.
        ''',
        choices = ['log_reg', 'lda', 'knn', 'svc', 'svc_l1', 'early_fusion', 'intermediate_fusion', 'late_fusion', 'pca', 'log_coef']
    )

    parser.add_argument(
        '--kernel', '-k',
        dest = 'kernel',
        required = False,
        action = 'store',
        default = 'linear',
        help = '''The kernel used for the support vector classification algorithm. This option is considered only when the --algorithm option is equal to 'svc'.
        ''',
        choices = ['linear', 'poly', 'rbf', 'sigmoid']
    )

    parser.add_argument(
        '--cross', '-c',
        dest = 'cross',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Computes the cross-validation score together with the score on the test set.
        If the --algorithm option is not provided or it is equal to 'early_fusion', 'intermediate_fusion', 'late_fusion' or 'pca', this option is ignored.
        '''
    )

    parser.add_argument(
        '--rf', '-r',
        dest = 'rf',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Uses only the features that in the logistic regression had a coefficient different from 0 to train or test the neural network. 
        If the --algorithm option is not provided, or if it is different from 'early_fusion', 'intermediate_fusion' or 'late_fusion', this option is ignored.
        '''
    )

    parser.add_argument(
        '--train', '-t',
        dest = 'train',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Trains the neural network before calculating the classification scores. 
        If the --algorithm option is not provided, or if it is different from 'early_fusion', 'intermediate_fusion' or 'late_fusion', this option is ignored.
        '''
    )

    parser.add_argument(
        '--plot', '-p',
        dest = 'plot',
        required = False,
        action = 'store',
        default = None,
        help = '''The number of the figure in the 'plots' folder to re-create using the ggplot2 library of R. Figures 3 and 6 cannot be re-created (since these figures have not been done with ggplot2).
        For figures 2, 4 and 7, only the single plots are created. The complete figures were built from the single plots using an image editor, and therefore they cannot be re-created using ggplot2.
        ''',
        choices = ['1', '2', '4', '5', '7']
    )

    parser.add_argument(
        '--download', '-d',
        dest = 'download',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Download the neural network models already trained, which are stored in the GitHub repository https://github.com/riccardo03-maker/Neural_networks_antimicrobial_resistance
        '''
    )

    parser.add_argument(
        '--erase', '-e',
        dest = 'erase',
        required = False,
        action = 'store_true',
        default = False,
        help = '''Erase all the files that can be created through this command line application.
        '''
    )

    args = parser.parse_args()

    if args.erase:
        #erase all files that can be created through the scripts in this repository
        subprocess.run(["rm", "plots/figure_1.png"])
        subprocess.run(["rm", "plots/figure_5.png"])
        subprocess.run("rm plots/figure_2/pca*", shell = True)
        subprocess.run("rm plots/figure_4/*scores.png", shell = True)
        subprocess.run("rm plots/figure_7/*scores.png", shell = True)
        subprocess.run("rm pipelines/results/log_reg_coefficients.csv", shell = True)
        subprocess.run("rm pipelines/results/pca/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/knn/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/lda/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/log_reg/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/svc_linear/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/svc_poly/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/svc_rbf/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/svc_sigmoid/*", shell = True)
        subprocess.run("rm pipelines/results/ml_algorithms/svm_paper/*", shell = True)
        subprocess.run("rm pipelines/results/neural_networks/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/early_fusion/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/early_fusion_rf/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/intermediate_fusion/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/intermediate_fusion_rf/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/late_fusion/*", shell = True)
        subprocess.run("rm -rf pipelines/nn_trained_models/late_fusion_rf/*", shell = True)


    if args.algorithm == 'log_reg':
        if args.cross:
            ml_algorithms.cross_validate_model(model_name = 'log_reg', C = 0.1, l1_ratio = 1.0, tol = 1e-6, solver = 'liblinear', 
                                             class_weight = 'balanced', random_state = 42)
        ml_algorithms.model_performance_test(model_name = 'log_reg', C = 0.1, l1_ratio = 1.0, tol = 1e-6, solver = 'liblinear', 
                                             class_weight = 'balanced', random_state = 42)
        
    elif args.algorithm == 'knn':
        if args.cross:
            ml_algorithms.cross_validate_model(model_name = 'knn', n_neighbors = 5)
        ml_algorithms.model_performance_test(model_name = 'knn', n_neighbors = 5)

    elif args.algorithm == 'lda':
        if args.cross:
            ml_algorithms.cross_validate_model(model_name = 'lda', solver = 'svd')
        ml_algorithms.model_performance_test(model_name = 'lda', solver = 'svd')

    elif args.algorithm == 'svc':
        if args.cross:
            ml_algorithms.cross_validate_model(model_name = 'svc', C = 0.1, kernel = args.kernel, tol = 1e-6, 
                                               gamma = 1., degree = 3, class_weight = 'balanced')
        ml_algorithms.model_performance_test(model_name = 'svc', C = 0.1, kernel = args.kernel, tol = 1e-6, 
                                               gamma = 1., degree = 3, class_weight = 'balanced')
        
    elif args.algorithm == 'svc_l1':
        if args.cross:
            ml_algorithms.cross_validate_model(model_name = 'svm_paper', penalty = 'l1', loss = 'squared_hinge', max_iter = 1000000, 
                                               tol = 0.000001, class_weight = "balanced", dual = False, C = 0.1, random_state = 42)
        ml_algorithms.model_performance_test(model_name = 'svm_paper', penalty = 'l1', loss = 'squared_hinge', max_iter = 1000000, 
                                               tol = 0.000001, class_weight = "balanced", dual = False, C = 0.1, random_state = 42)
    
    elif args.algorithm == 'early_fusion':
        if args.rf:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    for features in relevant_features_nn.all_combinations_of_features:
                        relevant_features_nn.train_relevant_features_nn(features = features, drug = drug, architecture = 'early_fusion')
            relevant_features_nn.nn_relevant_features_test(architecture = 'early_fusion')
        else:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    for features in neural_networks.all_combinations_of_features:
                        neural_networks.train_nn(features = features, drug = drug, architecture = 'early_fusion')
            neural_networks.nn_test(architecture = 'early_fusion')

    elif args.algorithm == 'intermediate_fusion':
        if args.rf:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    relevant_features_nn.train_relevant_features_nn(features = ['genexp', 'gpa', 'snps'], drug = drug, 
                                                                    architecture = 'intermediate_fusion')
            relevant_features_nn.nn_relevant_features_test(architecture = 'intermediate_fusion')
        else:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    neural_networks.train_nn(features = ['genexp', 'gpa', 'snps'], drug = drug, architecture = 'intermediate_fusion')
            neural_networks.nn_test(architecture = 'intermediate_fusion')

    elif args.algorithm == 'late_fusion':
        if args.rf:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    for features in [['genexp'], ['gpa'], ['snps']]:
                        relevant_features_nn.train_relevant_features_nn(features = features, drug = drug, architecture = 'late_fusion')
            relevant_features_nn.late_fusion_relevant_features_test()
        else:
            if args.train:
                for drug in ['Cef', 'Cip', 'Mer', 'Tob']:
                    for features in [['genexp'], ['gpa'], ['snps']]:
                        neural_networks.train_nn(features = features, drug = drug, architecture = 'late_fusion')
            neural_networks.late_fusion_nn_test()

    elif args.algorithm == 'pca':
        ml_algorithms.pca()

    elif args.algorithm == 'log_coef':
        ml_algorithms.get_logistic_regression_coefficients()


    if args.plot == '1':
        subprocess.run(["Rscript", "R_scripts/figure_1.R"])
    elif args.plot == '2':
        subprocess.run(["Rscript", "R_scripts/figure_2.R"])
    elif args.plot == '4':
        subprocess.run(["Rscript", "R_scripts/figure_4.R"])
    elif args.plot == '5':
        subprocess.run(["Rscript", "R_scripts/figure_5.R"])
    elif args.plot == '7':
        subprocess.run(["Rscript", "R_scripts/figure_7.R"])


    if args.download:
        subprocess.run(["bash", "bash_scripts/download_nn_models.sh"])