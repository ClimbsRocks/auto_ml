import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
from sklearn.model_selection import train_test_split

import utils_testing as utils

def test_calibrate_final_model_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    # Take a third of our test data (a tenth of our overall data) for calibration
    df_titanic_test, df_titanic_calibration = train_test_split(df_titanic_test, test_size=0.33, random_state=42)

    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }


    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, calibrate_final_model=True, X_test=df_titanic_calibration, y_test=df_titanic_calibration.survived)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.14 < test_score < -0.12

def test_calibrate_final_model_missing_X_test_y_test_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    # Take a third of our test data (a tenth of our overall data) for calibration
    df_titanic_test, df_titanic_calibration = train_test_split(df_titanic_test, test_size=0.33, random_state=42)

    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }


    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    # This should still work, just with warning printed
    ml_predictor.train(df_titanic_train, calibrate_final_model=True)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.14 < test_score < -0.12

