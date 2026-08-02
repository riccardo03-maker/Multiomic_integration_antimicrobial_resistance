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

Now you are ready to use the code in this repository. Remember that each output of the code (which can be for example a csv file of scores, or a plot) is stored in a dedicated folder. So, the code must always be executed from the project root directory to have all outputs stored in the correct folders.

## Usage

All the machine learning pipelines, as well as all the functions to create plots, can be executed directly from the `ml.py` script. This Python script works as a command line application, which can be used through the following syntax:

```bash
$ python ml.py --help
```

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
