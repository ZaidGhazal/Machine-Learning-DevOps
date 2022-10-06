# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn for a fictional telecommunication company. The data set contains information about the customers and their services. The goal is to predict the churn of the customers. The data set is available on Kaggle.

### Dataset

The dataset is available on Kaggle. The dataset contains 7043 rows and 21 columns. The data set contains information about the customers and their services. The goal is to predict the churn of the customers.

The data set includes information about:

    Customers who left within the last month – the column is called Churn
    Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
    Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
    Demographic info about customers – gender, age range, and if they have partners and dependents

### Folders

- `data` folder: contains the data set `bank_churn.csv`
- `images` folder: contains the images produced from EDA and model training
- `logs` folder: contains the logs produced during the program execution
- `models` folder: contains the trained models

### Files

- `churn_notebook.ipynb` : Jupyter notebook containing the code for eda, feature engineering, model training and evaluation
- `churn_library.py` : Python script containing the functions performing eda, features encoding, feature engineering, model training, and reporting the results
- `churn_script_logging_and_tests.py` : Python script containing the testing functions performing logging and testing on the main functions of the `churn_library.py` script




## Running Files

Running the following commands will result executing the the program to  perform eda, features encoding, 
feature engineering, model training, and reporting the results.

```
$ ipython churn_script_logging_and_tests.py
```
