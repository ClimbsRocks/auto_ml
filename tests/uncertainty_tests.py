import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
from sklearn.model_selection import train_test_split

import utils_testing as utils


# Tests on regression models:

def test_basic_predict_uncertainty(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, model_names=model_name, train_uncertainty_model=True)

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    # Bumping this up since without these features our score drops
    lower_bound = -4.0
    if model_name == 'DeepLearningRegressor':
        lower_bound = -14.5
    if model_name == 'LGBMRegressor':
        lower_bound = -4.95


    assert lower_bound < test_score < -2.8

    ml_predictor.get_uncertainty_prediction(df_boston_test)

