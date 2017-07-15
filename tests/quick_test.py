"""
nosetests -sv --nologcapture tests/quick_test.py
"""

import datetime
import os
import random
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'
os.environ['KERAS_BACKEND'] = 'theano'

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model

from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score

import dill
import numpy as np
import utils_testing as utils


# def regression_test():
#     # a random seed of 42 has ExtraTreesRegressor getting the best CV score, and that model doesn't generalize as well as GradientBoostingRegressor.
#     np.random.seed(0)

#     df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

#     column_descriptions = {
#         'MEDV': 'output'
#         , 'CHAS': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

#     ml_predictor.train(df_boston_train, model_names=['DeepLearningRegressor'])

#     test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

#     print('test_score')
#     print(test_score)

#     assert -3.35 < test_score < -2.8


def classification_test(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names=['DeepLearningClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17

