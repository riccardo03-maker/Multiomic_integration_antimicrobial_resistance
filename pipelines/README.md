# Machine learning pipelines

This is the most important folder of this repository. It contains the code for all machine learning pipelines, the classification scores obtained using these pipelines, and all the already trained neural network models.

## Table of contents

| Directory | Description |
|---|---|
|[ml_functions](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/pipelines/ml_functions)| Python functions (and relative tests) useful for machine learning pipelines|
|nn_trained_models| Folder to store trained neural network models (this folder is created during raw data transformation)|
|[results](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/pipelines/results)| Results for all machine learning pipelines|
|[ml_algorithms.py](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/pipelines/ml_algorithms.py)|Implementation of all classical machine learning algorithms|
|[neural_networks.py](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/pipelines/neural_networks.py)|Functions for implementation, training and testing of neural networks|
|[relevant_features_nn.py](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/pipelines/relevant_features_nn.py)|Functions for implementation, training and testing of neural networks using only relevant features|