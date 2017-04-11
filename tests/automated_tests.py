# This file is just to test passing a bunch of different parameters into train to make sure that things work
# At first, it is not necessarily testing whether those things have the intended effect or not



from collections import OrderedDict
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.argv.append('is_test_suite')


from auto_ml import Predictor

import dill
import numpy as np
from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils_testing as utils


# TODO:
    # Take in scoring upper bounds and lower bounds (with default values, which we'll use most of the time)
    # take in training_params_to_test
    # take in train_categorical_ensemble_or_not, defaulting to false
        # if true, pass in all the other training_params_to_test we want
    # I think ultimately we'll have two loops scrolling through different test params
    # an outer loop for train_categorical_ensemble_or_not, which we can extend to other features like that

    # Then we'll have all of our API coverage tests laid out, and run the entire group once for each training_params_to_test we encounter


# Remaining TODOs:
# 1. make this work for regressors
# 2. make sure we're not running tests multiple times
# 3. Figure out which tests only need to be run once, and reorganize them into the appropriate file
# 4. go through all our model name files, delete repeated tests, and copy in any missing tests
# 5. handle train_categorical_ensemble
# 6. Update upper and lower bounds programmatically


training_parameters = {
    'model_names': [None, 'GradientBoosting', 'XGB', 'DeepLearning', 'LGBM']
    # , 'train_categorical_ensemble': [True, False]
}


# MVP:
    # No expected_scores
    # no train_categorical_ensemble
    # classifiers only
    # see if we need to deal with complexity around nosetests not finding these tests since they're in a loop or not

# Structure:
    # This file has a dictionary mapping from test_name to the test definition


import api_coverage_tests_classifiers as classifier_tests
# import regressor_tests
# Make this an OrderedDict so that we run the tests in a consistent order
test_names = OrderedDict([
    # 'test_optimize_final_model_classification': classifier_tests.test_optimize_final_model_classification
    ('test_perform_feature_selection_true_classification', classifier_tests.test_perform_feature_selection_true_classification),
    ('test_perform_feature_selection_false_classification', classifier_tests.test_perform_feature_selection_false_classification),
    ('test_perform_feature_scaling_true_classification', classifier_tests.test_perform_feature_scaling_true_classification),
    ('test_perform_feature_scaling_false_classification', classifier_tests.test_perform_feature_scaling_false_classification),
    # , 'test_compare_all_models_classification': classifier_tests.test_compare_all_models_classification
    # , 'test_pass_in_list_of_dictionaries_train_classification': classifier_tests.test_pass_in_list_of_dictionaries_train_classification
    # , 'test_pass_in_list_of_dictionaries_predict_classification': classifier_tests.test_pass_in_list_of_dictionaries_predict_classification
    # , 'test_include_bad_y_vals_train_classification': classifier_tests.test_include_bad_y_vals_train_classification
    # , 'test_include_bad_y_vals_predict_classification': classifier_tests.test_include_bad_y_vals_predict_classification
    # , 'test_list_of_single_model_name_classification': classifier_tests.test_list_of_single_model_name_classification
    ('test_user_input_func_classification', classifier_tests.test_user_input_func_classification),
    ('test_binary_classification_predict_on_Predictor_instance', classifier_tests.test_binary_classification_predict_on_Predictor_instance),
    ('test_multilabel_classification_predict_on_Predictor_instance', classifier_tests.test_multilabel_classification_predict_on_Predictor_instance),
    ('test_binary_classification_predict_proba_on_Predictor_instance', classifier_tests.test_binary_classification_predict_proba_on_Predictor_instance)
])

# Test to move to singles:
# test_compare_all_models_classification
# test_pass_in_list_of_dictionaries_train_classification
# test_pass_in_list_of_dictionaries_predict_classification
# test_include_bad_y_vals_train_classification
# test_include_bad_y_vals_predict_classification
# test_list_of_single_model_name_classification

expected_scores = {
    'test_name': {
        'upper_bound_stringofallparams': 10
        , 'lower_bound_stringofallparams': 0
    }
}

def test_generator():
    for model_name in training_parameters['model_names']:
        for name, test in test_names.items():
            test_model_name = model_name
            if '_classification' in name and model_name is not None:
                test_model_name = model_name + 'Classifier'
            test.description = str(test_model_name) + '_' + name
            yield test, test_model_name




