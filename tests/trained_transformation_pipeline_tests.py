import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path


from auto_ml import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
from sklearn.model_selection import train_test_split

import utils_testing as utils

def test_already_transformed_X():
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

    # pass in return_trans_pipeline, and get the trans pipeline
    trans_pipeline = ml_predictor.train(df_titanic_train, model_names='LogisticRegression', return_transformation_pipeline=True)

    # get transformed X through transformation_only
    X_train_transformed = ml_predictor.transform_only(df_titanic_train)

    # create a new predictor
    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    # pass in trained trans pipeline, and make sure it works
    ml_predictor.train(df_titanic_train, trained_transformation_pipeline=trans_pipeline)
    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.14 < test_score < -0.12

    # pass in both a trans pipeline and a previously transformed X, and make sure that works
    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(None, trained_transformation_pipeline=trans_pipeline, transformed_X=X_train_transformed, transformed_y=df_titanic_train.survived)
    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.14 < test_score < -0.12
