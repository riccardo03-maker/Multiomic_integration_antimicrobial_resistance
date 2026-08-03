# Integration of multiomic data to predict antimicrobial resistance

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/riccardo03-maker/Multiomic_integration_antimicrobial_resistance.svg?style=plastic)](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/pulls)
[![GitHub issues](https://img.shields.io/github/issues/riccardo03-maker/Multiomic_integration_antimicrobial_resistance.svg?style=plastic)](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/issues)

[![GitHub stars](https://img.shields.io/github/stars/riccardo03-maker/Multiomic_integration_antimicrobial_resistance.svg?label=Stars&style=social)](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/riccardo03-maker/Multiomic_integration_antimicrobial_resistance.svg?label=Watch&style=social)](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/watchers)

This is the GitHub repository of the project "Integration of multiomic data to predict antimicrobial resistance". It contains all the code used for data preprocessing and analysis, for all the machine learning pipelines, and to plot all the figures of the project.

## Prerequisites

This repository is written in both Python and R languages. The library ggplot2 of R is used for the plots, while the rest of the code is written in Python.

A Python version of 3.10 or higher is required for the correct usage of this repository. All the required Python packages are reported in the [requirements.txt](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/requirements.txt), and they are automatically installed during the configuration of the repository (see the [Configuration](#configuration) section below).

To execute the code for the plots, a R version of 4.3.1 or higher is required, together with the package `tidyverse` (which also includes the `ggplot2` package). If R is already installed, `tidyverse` can be installed directly from the command line:

```bash
Rscript -e 'install.packages("tidyverse", repos="https://cloud.r-project.org")'
```

## Configuration

To use this repository, first clone it in your working directory

To install the `SandNet` package, first clone this repository

```bash
git clone https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance
```
and move to the project root directory

```bash
cd Multiomic_integration_antimicrobial_resistance
```
Before running the code, you need to install all required Python packages, and to extract and transform raw data so that they are ready for the machine learning pipelines. You can do all these things by just executing a bash script

```bash
bash bash_scripts/transform_data.sh
```

Now you are ready to use the code in this repository.

## Usage

The entire code in this repository can be executed directly from the `ml.py` script. This Python script works as a command line application, with three main options:

* `--algorithm`: executes a machine learning pipeline
* `--plots`: re-create one of the figures in the 'plots' folder using the `ggplot2` library of R
* `--erase`: eliminate all the files that can be created through this command line application

The `ml.py` command line application can be used through the following syntax:

```bash
$ python ml.py --help

usage: ml.py [-h] [--algorithm {log_reg,lda,knn,svc,svc_l1,early_fusion,intermediate_fusion,late_fusion,pca,log_coef}]
             [--kernel {linear,poly,rbf,sigmoid}] [--cross] [--rf] [--train] [--plot {1,2,4,5,7}] [--download] [--erase]

options:
  -h, --help            show this help message and exit
  --algorithm, -a {log_reg,lda,knn,svc,svc_l1,early_fusion,intermediate_fusion,late_fusion,pca,log_coef}
                        The machine learning algorithm to be executed. By default, for classical machine learning algorithms (so all options except for
                        the three fusion architectures), only the scores obtained on the test set are calculated. To compute also cross-validation scores,
                        use the --cross flag. By default, the neural network architectures use all the features. To use only the relevant features, use
                        the --rf flag.
  --kernel, -k {linear,poly,rbf,sigmoid}
                        The kernel used for the support vector classification algorithm. This option is considered only when the --algorithm option is
                        equal to 'svc'.
  --cross, -c           Computes the cross-validation score together with the score on the test set. If the --algorithm option is not provided or it is
                        equal to 'early_fusion', 'intermediate_fusion' or 'late_fusion', this option is ignored.
  --rf, -r              Uses only the features that in the logistic regression had a coefficient different from 0 to train or test the neural network. If
                        the --algorithm option is not provided, or if it is different from 'early_fusion', 'intermediate_fusion' or 'late_fusion', this
                        option is ignored.
  --train, -t           Trains the neural network before calculating the classification scores. If the --algorithm option is not provided, or if it is
                        different from 'early_fusion', 'intermediate_fusion' or 'late_fusion', this option is ignored.
  --plot, -p {1,2,4,5,7}
                        The number of the figure in the 'plots' folder to re-create using the ggplot2 library of R. Figures 3 and 6 cannot be re-created
                        (since these figures have not been done with ggplot2). For figures 2, 4 and 7, only the single plots are created. The complete
                        figures were built from the single plots using an image editor, and therefore they cannot be re-created using ggplot2.
  --download, -d        Download the neural network models already trained, which are stored in the GitHub repository
                        https://github.com/riccardo03-maker/Neural_networks_antimicrobial_resistance
  --erase, -e           Erase all the files that can be created through this command line interface.
```

Each output of the code (which can be for example a csv file of scores, or a plot) is stored in a dedicated folder. So, the `ml.py` must always be executed from the project root directory to have all outputs stored in the correct folders.

### Algorithm

The `--algorithm` option allows the execution of a machine learning pipeline (both classical machine learning algorithms and neural networks), chosen between the following ones:
* log_reg: logistic regression
* lda: linear discriminant analysis
* knn: K-nearest neighbours
* svc: support vector classification (the kernel can be specified with the `--kernel` argument)
* svc_l1: support vector classification with linear kernel and l1 regularization
* pca: principal component analysis
* log_coef: create the table of coefficients in logistic regression with all the three types of features
* early_fusion: neural network with early fusion architecture
* intermediate_fusion: neural network with intermediate fusion architecture
* late_fusion: neural network with late fusion architecture

Before using a neural network architecture to classify samples, the model needs to be trained. This can be done in two ways:

1) By directly training the model using the `--train` option

```bash
python ml.py --algorithm early_fusion --train
```
2) By downloading models already trained using the `--download` option

```bash
python ml.py --download
```

This downloads the trained models for all the possible neural network architectures.

### Plots

The `--plot` option allows the re-creation of figures 1, 2, 4, 5 and 7 in the [`plots`](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/plots) folder, using the `ggplot2` library of R.

### Erase

The `--erase` option removes all the files that can be created using this command line application. These files are:
* All the plots, except for figure 3 and 6, and the complete figures 2, 4 and 7
* The cross-validation and test scores, for both classical machine learning algorithms and neural networks
* The table with the coefficients of features in logistic regression
* The tables with pca results
* The trained neural network models

## Testing

A set of test functions are provided to test the behaviour of some functions that are frequently used in the machine learning pipelines. These test functions are stored in the [`test_data_transformation.py`](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/data_transformation/test_data_transformation.py) and in the [`test_ml_functions.py`](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/pipelines/ml_functions/test_ml_functions.py) scripts, and they can be run directly from the project root directory using the `pytest` package

```bash
python -m pytest
```

## Table of contents

Description of the folders of this repository

| Directory | Description |
|---|---|
| [R_scripts](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/R_scripts)| All R scripts (all the scripts to create the figures)|
|[bash_scripts](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/bash_scripts)| All shell scripts (to transform raw data and to download neural networks models)|
|[data_transformation](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/data_transformation)| Python functions (and relative tests) for raw data transformation|
|[pipelines](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/pipelines)| Code and results for all machine learning pipelines|
|[plots](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/plots)|All figures used in the project (not only those created with `ggplot2`)|
|[raw_data](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/tree/main/raw_data)| Raw data taken directly from [1]|
|transformed_data| Folder with transformed data ready to be used (this folder is created during raw data transformation)|
|[ml.py](https://github.com/riccardo03-maker/Multiomic_integration_antimicrobial_resistance/blob/main/ml.py)| Command line interface to execute all functions in this repository|

## Authors

*  **Riccardo Grandicelli**

## References

[1] Khaledi A., et al., 2020, *Predicting antimicrobial resistance in Pseudomonas aerug-
inosa with machine learning-enabled molecular diagnostics*, EMBO Molecular
Medicine, 12, EMMM201910264
