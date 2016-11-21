"""
To get standard out, run nosetests as follows:
  $ nosetests -sv tests
"""
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from nose.tools import assert_equal, assert_not_equal, with_setup

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

def get_boston_regression_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.33, random_state=42)
    return df_boston_train, df_boston_test

def get_titanic_binary_classification_dataset():
    try:
        df_titanic = pd.read_csv(os.path.join('tests', 'titanic.csv'))
    except Exception as e:
        print('Error')
        print(e)
        dataset_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv'
        df_titanic = pd.read_csv(dataset_url)
        df_titanic.to_csv(os.path.join('tests', 'titanic.csv'))
    # print(df_titanic)
    df_titanic = df_titanic.drop(['boat', 'body'], axis=1)
    df_titanic_train, df_titanic_test = train_test_split(df_titanic, test_size=0.33, random_state=42)
    return df_titanic_train, df_titanic_test


df_titanic_train, df_titanic_test = get_titanic_binary_classification_dataset()


def test_binary_classification():
    df_titanic_train_local = df_titanic_train.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    df_titanic_test_local = df_titanic_test.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train_local)

    test_score = ml_predictor.score(df_titanic_test_local, df_titanic_test_local.survived)
    # Right now we're getting a score of -.2136
    assert test_score > -0.22



df_boston_train, df_boston_test = get_boston_regression_dataset()

def test_regression():
    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, verbose=False)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV, verbose=False)
    print(test_score)

    # Currently, we expect to get a score of -3.87
    assert -4 < test_score


