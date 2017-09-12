# Splitting this in half to see if it helps with how nosetests splits up our parallel tests
from collections import OrderedDict
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

import classifiers as classifier_tests
import regressors as regressor_tests


training_parameters = {
    'model_names': ['DeepLearning', 'GradientBoosting', 'XGB', 'LGBM', 'CatBoost']
}


# Make this an OrderedDict so that we run the tests in a consistent order
test_names = OrderedDict([
    ('optimize_final_model_regression', regressor_tests.optimize_final_model_regression),
    # ('getting_single_predictions_regression', regressor_tests.getting_single_predictions_regression),
    # ('feature_learning_getting_single_predictions_regression', regressor_tests.feature_learning_getting_single_predictions_regression),
    # ('categorical_ensembling_regression', regressor_tests.categorical_ensembling_regression),
    # ('feature_learning_categorical_ensembling_getting_single_predictions_regression', regressor_tests.feature_learning_categorical_ensembling_getting_single_predictions_regression)
])

def test_generator():
    for model_name in training_parameters['model_names']:
        for test_name, test in test_names.items():
            test_model_name = model_name + 'Regressor'

            test.description = str(test_model_name) + '_' + test_name
            yield test, test_model_name
