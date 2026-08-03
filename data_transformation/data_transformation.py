#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_array, save_npz
import numpy as np
import pandas as pd

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def create_list_of_all_strains(file_path: str) -> list:
    '''
    Create a list with the names of all the strains (samples) used for machine learning pipelines.

    This list will be useful to remove extra reference strains that appear in gene expression and gpa features, and that are not useful 
    for the machine learning pipelines.

    Parameters
    ----------
        file_path: str
            The path to the file containing the list of the names of all strains.
    Returns
    -------
        strains_list: list
            List with the names of all strains.
    '''
    with open (file_path) as strains:
        strains_list = list(strains)
        strains_list = [strain.rstrip("\n") for strain in strains_list]
        #eliminate "\n" characters that are at the end of each strain in the list
        return strains_list


def transform_features(data: str) -> csr_array:
    '''
    Transform raw features data, creating sparse matrices with 414 rows (the number of samples) and columns given by the features.
    It also removes the extra strains present in genexp and gpa data.

    Parameters
    ----------
        data: {'genexp', 'gpa', 'snps'}
            The type of feature that is going to be transformed.
    Return
    ------
        data_matrix: scipy.sparse.csr_array
            The sparse matrix of features of the input type, with 414 rows (the number of samples used in the analysis)
    '''
    raw_data_folder = "./raw_data/features_gpa_expr_snps/" + data
    raw_data_path = raw_data_folder + "/" + data +"_feature_vect.npz"

    #create a csr sparse matrix from raw features data
    with np.load(raw_data_path) as raw_matrix:
        data_matrix = csr_array((raw_matrix["data"], raw_matrix["indices"], raw_matrix["indptr"]), shape = raw_matrix["shape"], dtype = np.float64)

    #create list of the 414 strains
    strains_list = create_list_of_all_strains("./transformed_data/strains_list.txt")

    #create list of the strains for the feature analyzed, including extra strains
    all_feature_strains = create_list_of_all_strains(raw_data_folder + "/" + data + "_strains_list.txt")

    #create a list with the indexes of the extra strains with respect to the 414 samples we are interested in
    strains_to_remove_indices = [i for i, strain in enumerate(all_feature_strains) if strain not in strains_list]

    #remove those indices
    data_matrix = csr_array(np.delete(data_matrix.toarray(), strains_to_remove_indices, axis = 0), dtype = np.float64)
    return data_matrix


def create_classes():
    '''
    Create a csv file with susceptibility and resistance to the four drugs for each sample.

    A sample susceptible to a drug is indicated as a 0, while a sample resistant to a drug is indicated as a 1. A missing value means
    that it is not clear whether that sample is susceptible or resistant to that drug.  
    '''
    classes = pd.read_csv("transformed_data/classes/phenotypes.txt")

    #remove column of Colistin resistance, which is not one of the four drugs we are interested in
    classes = classes.loc[:, ~classes.columns.str.startswith("Colistin")]

    #create a progressive index for each sample
    classes = classes.assign(Index = range(len(classes)))

    classes.columns = ["Strain", "Tob", "Cef", "Cip", "Mer", "Index"]
    classes.to_csv("./transformed_data/classes/classes.csv")


if (__name__ == '__main__'):
    #transform gene expression, gpa and snps data
    for feature in ['genexp', 'gpa', 'snps']:
        data_matrix = transform_features(feature)
        save_npz("./transformed_data/features/" + feature + "_features.npz", data_matrix)

    #create table with susceptibility and resistance to the four drugs for all samples
    create_classes()