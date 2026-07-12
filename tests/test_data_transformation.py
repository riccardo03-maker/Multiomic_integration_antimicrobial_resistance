#!/usr/bin/python
# -*- coding: utf-8 -*-

from datatransf import create_list_of_all_strains, transform_features

__author__=['Riccardo Grandicelli']
__email__=['riccardograndicelli03@gmail.com']


def test_list_of_all_strains():
    '''
    Test the correct creation of the list of all strains

    GIVEN: the list of all strains in the file "transformed_data/strains_list.txt"
    WHEN: I create the list of all strains from that file path
    THEN: I get a list with 414 elements with the right names
    '''
    strains_list = create_list_of_all_strains("transformed_data/strains_list.txt")
    assert(len(strains_list) == 414)
    assert(strains_list[96] == "CH4860")
    assert(strains_list[168] == "ESP044")


def test_genexp_data_transformation():
    '''
    Test the correct transformation of genexp data into a sparse matrix with 414 rows and 6026 columns

    GIVEN: data about gene expression features
    WHEN: I transform those data to remove extra strains
    THEN: I obtain a scipy.sparse.csr_array with 414 rows and 6026 columns
    '''
    genexp_matrix = transform_features("genexp")
    assert(genexp_matrix.shape[0] == 414)
    assert(genexp_matrix.shape[1] == 6026)