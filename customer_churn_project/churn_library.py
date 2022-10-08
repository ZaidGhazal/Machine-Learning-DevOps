"""This module contains functions for the customer churn project
auther: Zaid Ghazal
Date: 2022-10-06
"""
import warnings
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
warnings.filterwarnings("ignore")


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    df: pandas dataframe
    '''
    # import data from path
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
                    df: pandas dataframe

    output:
                    None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # create and save some plots related to EDA
    plt.figure(figsize=(20, 10))
    plt.figure(figsize=(20, 10))
    sns.countplot(x='Churn', data=df)
    plt.title("Churning Count", fontsize=20)
    plt.ylabel("Count")
    plt.xlabel("Churn")
    plt.savefig('images/eda/churning_count.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title("Customer Age Distribution", fontsize=20)
    plt.ylabel("Count")
    plt.xlabel("Age")
    plt.savefig('images/eda/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths=2)
    plt.title("Correlation Matrix", fontsize=20)
    plt.savefig('images/eda/correlation_matrix.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
                    df: pandas dataframe
                    category_lst: list of columns that contain categorical features
                    response: string of response name [optional argument that could
                    be used for naming variables or index y column]

    output:
                    df: pandas dataframe with new columns for
    '''

    # create categorized columns for each categorical column
    if response:
        cascade = "_" + response
    else:
        cascade = "_Churn"

    for cat_col in category_lst:
        grouped_col = df.groupby(cat_col).mean()['Churn']
        categorized_col_name = cat_col + cascade
        df[categorized_col_name] = df[cat_col].apply(
            lambda val: grouped_col[val])

    return df


def perform_feature_engineering(df, response):
    '''
    Split the dataset into training and testing subset with features and target seperated
    input:
                            df: pandas dataframe
                            response: string of response name [optional argument that
                            could be used for naming variables or index y column]

    output:
                            X_train: X training data
                            X_test: X testing data
                            y_train: y training data
                            y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Split the data into training and testing sets, keep only specific columns
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    if response:
        y = df[response]
    else:
        y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                            None
    '''
    # save classification report for training % testing data
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/random_forest_classification_report.png")

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/logistic_regression_classification_report.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
                            X_train: X training data
                            X_test: X testing data
                            y_train: y training data
                            y_test: y testing data
    output:
                            None
    '''

    # Define regressors
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Instantiate the grid search model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')

    # plots
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_curve.png")


if __name__ == "__main__":
    # load data
    data = import_data("data/bank_data.csv")
    # perform EDA and save related plots
    perform_eda(data)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # get data after encoding categorical columns
    data = encoder_helper(data, cat_columns, "Churn")
    train_features, test_features, train_target, test_target = perform_feature_engineering(
        data, "Churn")

    # Train and save models
    train_models(train_features, test_features, train_target, test_target)
    rfc_model = joblib.load('models/rfc_model.pkl')
    lrc_model = joblib.load('models/logistic_model.pkl')
    y_train_preds_lr = lrc_model.predict(train_features)
    y_test_preds_lr = lrc_model.predict(test_features)

    y_train_preds_rf = rfc_model.predict(train_features)
    y_test_preds_rf = rfc_model.predict(test_features)

    #
    classification_report_image(
        train_target,
        test_target,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    feature_importance_plot(
        rfc_model,
        train_features,
        "images/results/feature_importance.png")


if __name__ == "__main__":
    data = import_data("data/bank_data.csv")
    perform_eda(data)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data = encoder_helper(data, cat_columns, "Churn")
    train_features, test_features, train_target, test_target = perform_feature_engineering(
        data, "Churn")

    train_models(train_features, test_features, train_target, test_target)
    rfc_model = joblib.load('models/rfc_model.pkl')
    lrc_model = joblib.load('models/logistic_model.pkl')
    train_preds_lr = lrc_model.predict(train_features)
    test_preds_lr = lrc_model.predict(test_features)

    train_preds_rfc = rfc_model.predict(train_features)
    test_preds_rfc = rfc_model.predict(test_features)

    # Save classification report for training % testing data
    classification_report_image(
        train_target,
        test_target,
        train_preds_lr,
        train_preds_rfc,
        test_preds_lr,
        test_preds_rfc)

    # Creates and stores the feature importances in pth
    feature_importance_plot(
        rfc_model,
        train_features,
        "images/results/feature_importance.png")
