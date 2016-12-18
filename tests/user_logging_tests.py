# This set of tests id specifically designed to make sure auto_ml is user friendly- throwing useful warnings where possible about what specific actions the user can take to avoid an error, instead of throwing the non-obvious error messages that the underlying libraries will choke on.
import dill
from nose.tools import raises
import numpy as np
import os
import random
import sys
import warnings
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor
import utils_testing as utils


def test_unmarked_date_column():
    np.random.seed(0)

    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()

    column_descriptions = {
        'airline_sentiment': 'output'
        , 'airline': 'categorical'
        , 'text': 'nlp'
        , 'tweet_location': 'categorical'
        , 'user_timezone': 'categorical'
        # tweet_created is our date column. We want to test that "forgetting" to mark it as a date column throws a user warning before throwing the error
        # , 'tweet_created': 'date'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    with warnings.catch_warnings(record=True) as w:

        try:
            ml_predictor.train(df_twitter_train)
        except TypeError as e:
            pass

        print('Here are the caught warnings:')
        print(w)

        assert len(w) == 1

@raises(ValueError)
def test_bad_val_in_column_descriptions():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
        , 'fare': 'this_is_a_bad_value'
    }

    with warnings.catch_warnings(record=True) as w:

        ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
        print('we should be throwing a warning for the user to give them useful feedback')
        assert len(w) == 1

@raises(ValueError)
def test_missing_output_col_in_column_descriptions():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        # 'survived': 'output'
        'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

@raises(ValueError)
def test_bad_val_for_type_of_estimator():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        # 'survived': 'output'
        'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='invalid_type_of_estimator', column_descriptions=column_descriptions)


def test_nans_in_output_column():
    np.random.seed(0)

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

def test_verify_features_does_not_work_by_default():
    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)

    file_name = ml_predictor.save(str(random.random()))

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)
    os.remove(file_name)

    with warnings.catch_warnings(record=True) as w:

        results = saved_ml_pipeline.named_steps['final_model'].verify_features(df_titanic_test)

        print('Here are the caught warnings:')
        print(w)

        assert len(w) == 1

        assert results == None


def test_verify_features_finds_missing_prediction_features():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
        , 'sex': 'categorical'
    }


    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, verify_features=True)

    file_name = ml_predictor.save(str(random.random()))

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)
    os.remove(file_name)

    # Remove the "age" column from our prediction data
    df_titanic_test = df_titanic_test.drop('age', axis=1)

    missing_features = saved_ml_pipeline.named_steps['final_model'].verify_features(df_titanic_test)
    print('missing_features')
    print(missing_features)


    print("len(missing_features['prediction_not_training'])")
    print(len(missing_features['prediction_not_training']))
    print("len(missing_features['training_not_prediction'])")
    print(len(missing_features['training_not_prediction']))
    assert len(missing_features['prediction_not_training']) == 0
    assert len(missing_features['training_not_prediction']) == 1





def test_verify_features_finds_missing_training_features():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
        , 'sex': 'categorical'
    }

    # Remove the "sibsp" column from our training data
    df_titanic_train = df_titanic_train.drop('sibsp', axis=1)

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, verify_features=True)

    file_name = ml_predictor.save(str(random.random()))

    with open(file_name, 'rb') as read_file:
        saved_ml_pipeline = dill.load(read_file)
    os.remove(file_name)


    missing_features = saved_ml_pipeline.named_steps['final_model'].verify_features(df_titanic_test)
    print('missing_features')
    print(missing_features)


    print("len(missing_features['prediction_not_training'])")
    print(len(missing_features['prediction_not_training']))
    print("len(missing_features['training_not_prediction'])")
    print(len(missing_features['training_not_prediction']))
    assert len(missing_features['prediction_not_training']) == 1
    assert len(missing_features['training_not_prediction']) == 0




def test_verify_features_finds_no_missing_features_when_none_are_missing():
        np.random.seed(0)

        df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

        column_descriptions = {
            'survived': 'output'
            , 'embarked': 'categorical'
            , 'pclass': 'categorical'
            , 'sex': 'categorical'
        }


        ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
        ml_predictor.train(df_titanic_train, verify_features=True)

        file_name = ml_predictor.save(str(random.random()))

        with open(file_name, 'rb') as read_file:
            saved_ml_pipeline = dill.load(read_file)
        os.remove(file_name)

        missing_features = saved_ml_pipeline.named_steps['final_model'].verify_features(df_titanic_test)
        print('missing_features')
        print(missing_features)


        print("len(missing_features['prediction_not_training'])")
        print(len(missing_features['prediction_not_training']))
        print("len(missing_features['training_not_prediction'])")
        print(len(missing_features['training_not_prediction']))
        assert len(missing_features['prediction_not_training']) == 0
        assert len(missing_features['training_not_prediction']) == 0

