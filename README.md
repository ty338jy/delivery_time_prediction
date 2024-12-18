# Final Combined Project

This project is the final combined submission for D400 and D100.

## Project Overview

This project predicts the delivery time based on a dataset taken from Kaggle 

https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset/data?select=train.csv

The file has been downloaded and saved under data/raw folder. 

The main purpose of the project would be to perform exploratory / expanantory analysis (in eda_cleaning.ipynb), then apply different machine learning techniques (GLM and LGBM in specific) to predict the food delivery time and compare model performance (model_training.py)

## Repository Structure
FINAL_PROJECT/
│
├── data/                         # Folder for dataset
│   ├── prepared/                 
│   │   └── prepared_data.parquet # output from eda_cleaning.ipynb
│   │
│   └── raw/                     
│       └── data.csv              # raw data downloaded from Kaggle
│   
├── final_combined/               # folder for modularized codes
│   ├── data/                     # scripts for data load 
│   ├── evaluation/               # scripts for model evaluation 
│   ├── feature_engineering/      # script for feature engineering
│   ├── preprocess/               # scrupt for preprocessing
│   ├── visualization/            # script for visualisation 
│   └── __init__.py              
│
├── tests/                        # folder for unit tests
│
├── .gitignore                    # git ignore file
├── .pre-commit-config.yaml       # pre-commit hooks configuration
├── eda_cleaning.ipynb            # notebook for exploratory data analysis and cleaning
├── environment.yml               # conda environment configuration
├── model_training.py             # main script for model training
├── pyproject.toml                # project configuration for build tools
├── README.md                     # documentation file
├── Report.pdf                    # PDF report for the project
└── setup.cfg                     # project setup configuration


### Installation 

conda env create -f environment.yml

conda activate final

pip install .

pre-commit install

pre-commit run --all-files

### Usage

python model_training.py
