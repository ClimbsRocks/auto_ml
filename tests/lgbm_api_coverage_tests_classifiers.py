# This file is just to test passing a bunch of different parameters into train to make sure that things work
# At first, it is not necessarily testing whether those things have the intended effect or not

import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import dill
import numpy as np
from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils_testing as utils

# This test works individually, but it runs into some kind of multiprocessing hang where we can't run more than one optimize_final_model test across our entire test suite. Ignoring this one, since it is costly, and we know it works.
def test_optimize_final_model_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, optimize_final_model=True, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    # Small sample sizes mean there's a fair bit of noise here
    assert -0.226 < test_score < -0.17


def test_perform_feature_selection_true_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=True, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.225 < test_score < -0.17

def test_perform_feature_selection_false_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_selection=False, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_perform_feature_scaling_true_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=True, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17

def test_perform_feature_scaling_false_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, perform_feature_scaling=False, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_optimize_entire_pipeline_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, optimize_entire_pipeline=True, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_X_test_and_y_test_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, X_test=df_titanic_test, y_test=df_titanic_test.survived, model_names=['LGBMClassifier'])

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


# def test_compute_power_1_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, compute_power=1, model_names=['LGBMClassifier'])

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

#     ml_predictor.train(df_titanic_train, compute_power=9, model_names=['LGBMClassifier'])

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17



# This test hangs. Not sure why. Some kind of multiprocessing error.
# def test_all_algos_classification():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, model_names=['LGBMClassifier', 'LogisticRegression', 'RandomForestClassifier', 'RidgeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier'])

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     assert -0.215 < test_score < -0.17

# If the user passes in X_test and y_test, we will use those to determine the best model, rather than CV scores
def test_select_from_multiple_classification_models_using_X_test_and_y_test():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train, model_names=['LGBMClassifier', 'LogisticRegression', 'RandomForestClassifier', 'RidgeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier'], X_test=df_titanic_test, y_test=df_titanic_test.survived)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    assert -0.215 < test_score < -0.17


def test_binary_classification_predict_on_Predictor_instance():
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



def test_multilabel_classification_predict_on_Predictor_instance():
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


def test_binary_classification_predict_proba_on_Predictor_instance():
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

#     ml_predictor.train(df_titanic_train, compute_power=10, model_names=['LGBMClassifier'])

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


