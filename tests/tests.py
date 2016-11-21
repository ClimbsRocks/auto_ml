"""
To get standard out, run nosetests as follows:
  $ nosetests -sv tests
"""
import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from nose.tools import assert_equal, assert_not_equal, with_setup

import dill
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
        # Do not write the index that pandas automatically creates
        df_titanic.to_csv(os.path.join('tests', 'titanic.csv'), index=False)
    # print(df_titanic)
    df_titanic = df_titanic.drop(['boat', 'body'], axis=1)
    df_titanic_train, df_titanic_test = train_test_split(df_titanic, test_size=0.33, random_state=42)
    return df_titanic_train, df_titanic_test




def test_binary_classification():
    df_titanic_train, df_titanic_test = get_titanic_binary_classification_dataset()
    df_titanic_train = df_titanic_train.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    df_titanic_test = df_titanic_test.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived, verbose=0)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    assert -0.215 < test_score < -0.17




def test_regression():
    df_boston_train, df_boston_test = get_boston_regression_dataset()
    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, verbose=False)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV, verbose=0)

    # Currently, we expect to get a score of -3.09
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_regression():
    df_boston_train, df_boston_test = get_boston_regression_dataset()
    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, verbose=False)

    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_boston_test, df_boston_test.MEDV)
    # Make sure our score is good, but not unreasonably good
    assert -3.2 < test_score < -2.8


def test_saving_trained_pipeline_binary_classification():
    df_titanic_train, df_titanic_test = get_titanic_binary_classification_dataset()
    df_titanic_train = df_titanic_train.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    df_titanic_test = df_titanic_test.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)
    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train)

    file_name = ml_predictor.save()

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    test_score = saved_ml_pipeline.score(df_titanic_test, df_titanic_test.survived)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    assert -0.215 < test_score < -0.17


