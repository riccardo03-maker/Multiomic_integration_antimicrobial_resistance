#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import csr_array, save_npz
import numpy as np
from collections import Counter

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def create_list_of_all_strains(file_path: str) -> list:
    '''
    Create a list of all the strains (isolates) used for the machine learning pipeline.

    This list will be useful to remove extra reference strains that appear in some of the features data, and that are not part of the
    machine learning pipeline.

    Parameters
    ----------
        file_path: str
            The path to the file containing the list of all strains.
    Returns
    -------
        strains_list: list
            The list of all strains.
    '''
    with open (file_path) as strains:
        strains_list = list(strains)
        strains_list = [strain.rstrip("\n") for strain in strains_list]
        #eliminate "\n" characters that are at the end of each strain in the list
        return strains_list


def transform_features(data: str) -> csr_array:
    '''
    Transform raw features data, creating sparse matrices with 414 rows (the number of strains) and columns given by the features.

    Parameters
    ----------
        data: {'genexp', 'gpa', 'snps'}
            The type of feature that is going to be transformed.
    Return
    ------
        data_matrix: scipy.sparse.csr_array
            The sparse matrix of features of the input type, with 414 rows (the number of strains used in the analysis)
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

    #create a list with the indexes of the extra strains with respect to the 414 isolates we are interested in
    strains_to_remove_indices = [i for i, strain in enumerate(all_feature_strains) if strain not in strains_list]

    #remove those indices
    data_matrix = csr_array(np.delete(data_matrix.toarray(), strains_to_remove_indices, axis = 0), dtype = np.float64)
    return data_matrix


if (__name__ == '__main__'):
    #transform gene expression, gpa and snps data
    for feature in ['genexp', 'gpa', 'snps']:
        data_matrix = transform_features(feature)
        save_npz("./transformed_data/features/" + feature + "_features.npz", data_matrix)