"""
To get standard out, run nosetests as follows:
nosetests -sv tests
nosetests --verbosity=2 --detailed-errors --nologcapture --processes=4 --process-restartworker --process-timeout=1000 tests
"""
import datetime
import os
import random
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

from auto_ml import Predictor
from auto_ml.utils_models import load_ml_model

from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score

import dill
import numpy as np
import utils_testing as utils

def test_is_backwards_compatible_with_models_trained_using_1_9_6():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    with open(os.path.join('tests','trained_ml_model_v_1_9_6.dill'), 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    print('predictions')
    print(predictions)
    print('predictions[0]')
    print(predictions[0])
    print('type(predictions)')
    print(type(predictions))
    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good

    assert -2.8 < first_score < -2.1

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_boston_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_boston_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.1 < duration.total_seconds() / 1.0 < 15


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('second_score')
    print(second_score)
    # Make sure our score is good, but not unreasonably good

    assert -2.8 < second_score < -2.1


def test_all_algos_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names=['LogisticRegression', 'RandomForestClassifier', 'RidgeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier', 'DeepLearningClassifier', 'XGBClassifier', 'LGBMClassifier', 'LinearSVC'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17

def test_linear_model_analytics_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names='RidgeClassifier')

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    # Linear models aren't super great on this dataset...
    assert -0.37 < test_score < -0.17


def test_all_algos_regression():
    # a random seed of 42 has ExtraTreesRegressor getting the best CV score, and that model doesn't generalize as well as GradientBoostingRegressor.
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, model_names=['LinearRegression', 'RandomForestRegressor', 'Ridge', 'GradientBoostingRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'SGDRegressor', 'PassiveAggressiveRegressor', 'Lasso', 'LassoLars', 'ElasticNet', 'OrthogonalMatchingPursuit', 'BayesianRidge', 'ARDRegression', 'MiniBatchKMeans', 'DeepLearningRegressor', 'XGBRegressor', 'XGBClassifier',  'LinearSVR'])

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    assert -3.35 < test_score < -2.8

def test_input_df_unmodified():
    np.random.seed(42)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    df_shape = df_boston_train.shape
    ml_predictor.train(df_boston_train)

    training_shape = df_boston_train.shape
    assert training_shape[0] == df_shape[0]
    assert training_shape[1] == df_shape[1]


    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    assert -3.35 < test_score < -2.8

def test_model_uses_user_provided_training_params(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    try:
        ml_predictor.train(df_titanic_train, model_names='RidgeClassifier', training_params={'this_param_is_not_valid': True})
        assert False
    except ValueError as e:
        assert True


def test_is_backwards_compatible_with_models_trained_using_1_9_6():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    with open(os.path.join('tests','trained_ml_model_v_1_9_6.dill'), 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)

    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    print('predictions')
    print(predictions)
    print('predictions[0]')
    print(predictions[0])
    print('type(predictions)')
    print(type(predictions))
    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good

    assert -2.8 < first_score < -2.1

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_boston_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_boston_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.1 < duration.total_seconds() / 1.0 < 15


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('second_score')
    print(second_score)


def test_ignores_new_invalid_features():

    # One of the great unintentional features of auto_ml is that you can pass in new features at prediction time, that weren't present at training time, and they're silently ignored!
    # One edge case here is new features that are strange objects (lists, datetimes, intervals, or anything else that we can't process in our default data processing pipeline). Initially, we just ignored them in dict_vectorizer, but we need to ignore them earlier.
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train)

    file_name = ml_predictor.save(str(random.random()))

    saved_ml_pipeline = load_ml_model(file_name)

    os.remove(file_name)
    try:
        keras_file_name = file_name[:-5] + '_keras_deep_learning_model.h5'
        os.remove(keras_file_name)
    except:
        pass


    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_boston_test_dictionaries:
        if random.random() > 0.9:
            row['totally_new_feature'] = datetime.datetime.now()
            row['really_strange_feature'] = random.random
            row['we_should_really_ignore_this'] = Predictor
            row['pretty_vanilla_ignored_field'] = 8
            row['potentially_confusing_things_here'] = float('nan')
            row['potentially_confusing_things_again'] = float('inf')
            row['this_is_a_list'] = [1,2,3,4,5]
        predictions.append(saved_ml_pipeline.predict(row))

    print('predictions')
    print(predictions)
    print('predictions[0]')
    print(predictions[0])
    print('type(predictions)')
    print(type(predictions))
    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('first_score')
    print(first_score)
    # Make sure our score is good, but not unreasonably good

    lower_bound = -3.2
    assert lower_bound < first_score < -2.8

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_boston_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_boston_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.1 < duration.total_seconds() / 1.0 < 15


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print('second_score')
    print(second_score)
    # Make sure our score is good, but not unreasonably good

    assert lower_bound < second_score < -2.8
