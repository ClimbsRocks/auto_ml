# This file is just to test passing a bunch of different parameters into train to make sure that things work
# At first, it is not necessarily testing whether those things have the intended effect or not

import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

from nose.tools import assert_equal, assert_not_equal, with_setup

import dill

import utils_testing as utils

def test_optimize_final_model():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, optimize_final_model=True)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_perform_feature_selection():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=False)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_perform_feature_scaling():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=False)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


# TODO: run tests for each of the different models
# TODO: test for picking the best model automatically