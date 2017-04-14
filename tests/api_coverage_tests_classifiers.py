# This file is just to test passing a bunch of different parameters into train to make sure that things work
# At first, it is not necessarily testing whether those things have the intended effect or not

import datetime
import os
import random
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import dill
import numpy as np
from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils_testing as utils

# def test_optimize_final_model_classification(model_name=None):
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, optimize_final_model=True, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     # Small sample sizes mean there's a fair bit of noise here
#     assert -0.226 < test_score < -0.17

def test_perform_feature_selection_true_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=True, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.225 < test_score < -0.17


def test_perform_feature_selection_false_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=False, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    lower_bound = -0.215

    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.221

    assert lower_bound < test_score < -0.17


def test_perform_feature_scaling_true_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=True, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    lower_bound = -0.215
    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.235

    assert lower_bound < test_score < -0.17

def test_perform_feature_scaling_false_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=False, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    lower_bound = -0.215
    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.235

    assert lower_bound < test_score < -0.17


def test_user_input_func_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    def age_bucketing(data):

        def define_buckets(age):
            if age <= 17:
                return 'youth'
            elif age <= 40:
                return 'adult'
            elif age <= 60:
                return 'adult2'
            else:
                return 'over_60'

        if isinstance(data, dict):
            data['age_bucket'] = define_buckets(data['age'])
        else:
            data['age_bucket'] = data.age.apply(define_buckets)

        return data

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
        , 'age_bucket': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=False, user_input_func=age_bucketing, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    lower_bound = -0.215
    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.235

    assert lower_bound < test_score < -0.17

def test_binary_classification_predict_on_Predictor_instance(model_name=None):
    np.random.seed(0)


    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)

    #
    predictions = ml_predictor.predict(df_titanic_test)
    test_score = accuracy_score(predictions, df_titanic_test.survived)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    print(test_score)
    assert .65 < test_score < .75



def test_multilabel_classification_predict_on_Predictor_instance(model_name=None):
    np.random.seed(0)

    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()
    ml_predictor = utils.train_basic_multilabel_classifier(df_twitter_train)

    predictions = ml_predictor.predict(df_twitter_test)
    test_score = accuracy_score(predictions, df_twitter_test.airline_sentiment)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    print('test_score')
    print(test_score)
    assert 0.67 < test_score < 0.79


def test_binary_classification_predict_proba_on_Predictor_instance(model_name=None):
    np.random.seed(0)


    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    ml_predictor = utils.train_basic_binary_classifier(df_titanic_train)

    #
    predictions = ml_predictor.predict_proba(df_titanic_test)
    predictions = [pred[1] for pred in predictions]
    test_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    # Right now we're getting a score of -.205
    # Make sure our score is good, but not unreasonably good
    print(test_score)
    assert -0.215 < test_score < -0.17


def test_nlp_multilabel_classification(model_name=None):
    np.random.seed(0)

    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()

    column_descriptions = {
        'airline_sentiment': 'output'
        , 'airline': 'categorical'
        , 'text': 'nlp'
        , 'tweet_location': 'categorical'
        , 'user_timezone': 'categorical'
        , 'tweet_created': 'date'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_twitter_train, model_names=model_name)

    test_score = ml_predictor.score(df_twitter_test, df_twitter_test.airline_sentiment)

    # Make sure our score is good, but not unreasonably good
    print('test_score')
    print(test_score)
    assert 0.67 < test_score < 0.79

# For some reason, this test now causes a Segmentation Default on travis when run on python 3.5.
# home/travis/.travis/job_stages: line 53:  8810 Segmentation fault      (core dumped) nosetests -v --with-coverage --cover-package auto_ml tests
# It didn't error previously
# It appears to be an environment issue (possibly cuased by running too many parallelized things, which only happens in a test suite), not an issue with auto_ml. So we'll run this test to make sure the library functionality works, but only on some environments
if os.environ.get('TRAVIS_PYTHON_VERSION', '0') != '3.5':
    def test_compare_all_models_classification(model_name=None):
        np.random.seed(0)

        df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

        column_descriptions = {
            'survived': 'output'
            , 'embarked': 'categorical'
            , 'pclass': 'categorical'
        }

        ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

        ml_predictor.train(df_titanic_train, compare_all_models=True, model_names=model_name)

        test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

        print('test_score')
        print(test_score)

        assert -0.215 < test_score < -0.17


def test_pass_in_list_of_dictionaries_train_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    list_titanic_train = df_titanic_train.to_dict('records')

    ml_predictor.train(list_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_pass_in_list_of_dictionaries_predict_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    list_titanic_train = df_titanic_train.to_dict('records')

    ml_predictor.train(df_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test.to_dict('records'), df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_include_bad_y_vals_train_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    df_titanic_train.ix[1, 'survived'] = None
    df_titanic_train.ix[8, 'survived'] = None
    df_titanic_train.ix[26, 'survived'] = None

    ml_predictor.train(df_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test.to_dict('records'), df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17



def test_include_bad_y_vals_predict_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    df_titanic_test.ix[1, 'survived'] = float('nan')
    df_titanic_test.ix[8, 'survived'] = float('inf')
    df_titanic_test.ix[26, 'survived'] = None

    ml_predictor.train(df_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test.to_dict('records'), df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_list_of_single_model_name_classification(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names=[model_name])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17

if os.environ.get('TRAVIS_PYTHON_VERSION', '0') != '3.5':
    def test_getting_single_predictions_nlp_date_multilabel_classification(model_name=None):
        # auto_ml does not support multilabel classification for deep learning at the moment
        if model_name == 'DeepLearningClassifier':
            return

        np.random.seed(0)

        df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()

        column_descriptions = {
            'airline_sentiment': 'output'
            , 'airline': 'categorical'
            , 'text': 'nlp'
            , 'tweet_location': 'categorical'
            , 'user_timezone': 'categorical'
            , 'tweet_created': 'date'
        }

        ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
        ml_predictor.train(df_twitter_train, model_names=model_name)

        file_name = ml_predictor.save(str(random.random()))

        if model_name == 'DeepLearningClassifier':
            from auto_ml.utils_models import load_keras_model

            saved_ml_pipeline = load_keras_model(file_name)
        else:
            with open(file_name, 'rb') as read_file:
                saved_ml_pipeline = dill.load(read_file)

        os.remove(file_name)

        df_twitter_test_dictionaries = df_twitter_test.to_dict('records')

        # 1. make sure the accuracy is the same

        predictions = []
        for row in df_twitter_test_dictionaries:
            predictions.append(saved_ml_pipeline.predict(row))

        print('predictions')
        print(predictions)

        first_score = accuracy_score(df_twitter_test.airline_sentiment, predictions)
        print('first_score')
        print(first_score)
        # Make sure our score is good, but not unreasonably good
        lower_bound = 0.67
        if model_name == 'LGBMClassifier':
            lower_bound = 0.655
        assert lower_bound < first_score < 0.79

        # 2. make sure the speed is reasonable (do it a few extra times)
        data_length = len(df_twitter_test_dictionaries)
        start_time = datetime.datetime.now()
        for idx in range(1000):
            row_num = idx % data_length
            saved_ml_pipeline.predict(df_twitter_test_dictionaries[row_num])
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print('duration.total_seconds()')
        print(duration.total_seconds())

        # It's very difficult to set a benchmark for speed that will work across all machines.
        # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
        # That's about 1 millisecond per prediction
        # Assuming we might be running on a test box that's pretty weak, multiply by 3
        # Also make sure we're not running unreasonably quickly
        time_upper_bound = 3
        if model_name == 'XGBClassifier':
            time_upper_bound = 4
        assert 0.2 < duration.total_seconds() < time_upper_bound


        # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

        predictions = []
        for row in df_twitter_test_dictionaries:
            predictions.append(saved_ml_pipeline.predict(row))

        print('predictions')
        print(predictions)
        print('df_twitter_test_dictionaries')
        print(df_twitter_test_dictionaries)
        second_score = accuracy_score(df_twitter_test.airline_sentiment, predictions)
        print('second_score')
        print(second_score)
        # Make sure our score is good, but not unreasonably good
        assert lower_bound < second_score < 0.79




# def test_optimize_entire_pipeline_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, optimize_entire_pipeline=True, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17


# def test_X_test_and_y_test_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, X_test=df_titanic_test, y_test=df_titanic_test.survived, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17


# def test_compute_power_1_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, compute_power=1, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17


# This test passes, but takes a long time to run. deprecating it for now until we rethink what we really want compute_power to accomplish
# def test_compute_power_9_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, compute_power=9, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17


# If the user passes in X_test and y_test, we will use those to determine the best model, rather than CV scores
# def test_select_from_multiple_classification_models_using_X_test_and_y_test():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, model_names=['LogisticRegression', 'RandomForestClassifier', 'RidgeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier'], X_test=df_titanic_test, y_test=df_titanic_test.survived, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17



# I'm not sure what a reasonable value here would be
# def test_multilabel_classification_predict_proba_on_Predictor_instance():
#     np.random.seed(0)

#     df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()
#     ml_predictor = utils.train_basic_multilabel_classifier(df_twitter_train)

#     predictions = ml_predictor.predict_proba(df_twitter_test)
#     test_score = accuracy_score(predictions, df_twitter_test.airline_sentiment)
#     # Right now we're getting a score of -.205
#     # Make sure our score is good, but not unreasonably good
#     print('test_score')
#     print(test_score)
#     assert 0.67 < test_score < 0.79




# Right now this just takes forever to run. Will have to look into optimizing
# def test_compute_power_10():
    # np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     # This test tries something like 2,000 different combinations of hyperparameters for the pipeline
#     # To make this more reasonable, we'll cut down the data size to be a fraction of it's full size, so we are just testing whether everything runs
#     df_titanic_train, df_titanic_train_ignore = train_test_split(df_titanic_train, train_size=0.1, random_state=42)


#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, compute_power=10, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17


# TODO: run tests for each of the different models
# TODO: test for picking the best model automatically
# optimize_entire_pipeline
# X_test, y_test
# take_log_of_y
# ideally something about XGB not needing to be installed, but running if it is installed
# Dates
# NLP
# train ml ensemble
# ideally use XGB
# print results for linear model
# Try passing in a list of dictionaries everywhere (train, predict, predict_proba, score, on both the saved pipeline and the ml_predictor object)
# call predict and predict_proba on the ml_predictor object, not just the final pipeline
# Test the num_features for standard problem (should be all of them). Then test num_features if we pass in a ton of duplicate columns where it should be removing some features. then same thing, but perform_feature_selection=False. we can get the num_features from the final trained model. even if some features are not useful, we should be able to see the number of features the model has considered.


