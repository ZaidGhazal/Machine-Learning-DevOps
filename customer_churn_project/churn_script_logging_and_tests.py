""" This module contains test_functions to test the churn_libirary.py module functions
auther: Zaid Ghazal
Date: 2022-10-06
"""
import os
import logging
from urllib import response
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    pytest.df = df


def test_eda():
    '''
    test perform eda function
    '''
    try:
        df = pytest.df
        perform_eda(df)

    except Exception as err:
        logging.error("ERROR Testing: perform_eda, %s", err)
        raise err

    try:
        assert all([True for val in df['Churn']if val in [0, 1]]
                   ), "Churn column has values other than 0 and 1"
        assert os.path.exists(
            "images/eda/customer_age_distribution.png"), "customer_age_distribution.png not found"
        assert os.path.exists(
            "images/eda/churning_count.png"), "churning_count.png not found"
        assert os.path.exists(
            "images/eda/correlation_matrix.png"), "correlation_matrix.png not found"

        pytest.df = df
        logging.info("SUCCESS Output: perform_eda")

    except AssertionError as err:
        logging.error("Error Output: Testing perform_eda: %s", err)
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        response = "Churn"
        df = pytest.df
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df = encoder_helper(df, cat_columns, response)
        pytest.df = df

        logging.info("SUCCESS Testing: encoder_helper")

    except Exception as err:
        logging.error("Error Testing: encoder_helper: %s", err)
        raise err

    try:
        assert all([True for col in cat_columns if col + "_" + \
                   response in df.columns]), "encoder_helper did not create new columns"
        logging.info("SUCCESS Output: encoder_helper")
    except AssertionError as err:
        logging.error("Error Output: encoder_helper: %s", err)
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    df = pytest.df
    response = "Churn"

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, response)

        logging.info("SUCCESS Testing: perform_feature_engineering")
    except Exception as err:
        logging.error("ERROR Testing: perform_feature_engineering, %s", err)
        raise err

    try:
        assert X_train.shape[0] > 0, "X_train has no rows"
        assert X_train.shape[1] > 0, "X_train has no columns"
        assert X_test.shape[0] > 0, "X_test has no rows"
        assert X_test.shape[1] > 0, "X_test has no columns"
        assert y_train.shape[0] > 0, "y_train has no rows"
        assert y_test.shape[0] > 0, "y_test has no rows"

        pytest.X_train = X_train
        pytest.X_test = X_test
        pytest.y_train = y_train
        pytest.y_test = y_test

        logging.info("SUCCESS Output: perform_feature_engineering")

    except AssertionError as err:
        logging.error("ERROR Output: perform_feature_engineering, %s", err)
        raise err


def test_train_models():
    '''
    test train_models
    '''
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS Testing: train_models")
    except Exception as err:
        logging.error("ERROR Testing: train_models, %s", err)
        raise err

    try:
        assert os.path.exists(
            "models/rfc_model.pkl"), "rfc_model.pkl not found"
        assert os.path.exists(
            "models/logistic_model.pkl"), "logistic_model.pkl not found"
        assert os.path.exists(
            "images/results/roc_curve.png"), "roc_curve.png not found"
        logging.info("SUCCESS Output: train_models")
    except AssertionError as err:
        logging.error("ERROR Output: train_models, %s", err)
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
