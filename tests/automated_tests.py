from collections import OrderedDict
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

import classifiers as classifier_tests
import regressors as regressor_tests

# Remaining TODOs:
# 3. Figure out which tests only need to be run once, and reorganize them into the appropriate file
# 5. handle train_categorical_ensemble
# 6. adjust bounds for categorical_ensemble
# 7. Make sure categorical_ensembling works with feature_learning!

# Feature TODOs:
# 1. Remove nans from categorical_column (replace with None?)
# 2. don't try to use nan as a key. instead, use our default category
# 3. verify feature_learning passes all the tests
# 4. add minimum category size as a param (otherwise, just use default- if the user wants to make these their own small category, totally on them)
# 5. check to make sure a category has more than 1 label if training classifier


training_parameters = {
    'model_names': ['GradientBoosting', 'XGB', 'DeepLearning', 'LGBM']
    # 'model_names': ['DeepLearning']
    # , 'train_categorical_ensemble': [True, False]
}


# Make this an OrderedDict so that we run the tests in a consistent order
test_names = OrderedDict([
    ('optimize_final_model_classification', classifier_tests.optimize_final_model_classification),
    ('perform_feature_selection_true_classification', classifier_tests.perform_feature_selection_true_classification),
    ('perform_feature_selection_false_classification', classifier_tests.perform_feature_selection_false_classification),
    ('perform_feature_scaling_true_classification', classifier_tests.perform_feature_scaling_true_classification),
    ('perform_feature_scaling_false_classification', classifier_tests.perform_feature_scaling_false_classification),
    ('user_input_func_classification', classifier_tests.user_input_func_classification),
    ('binary_classification_predict_on_Predictor_instance', classifier_tests.binary_classification_predict_on_Predictor_instance),
    ('multilabel_classification_predict_on_Predictor_instance', classifier_tests.multilabel_classification_predict_on_Predictor_instance),
    ('binary_classification_predict_proba_on_Predictor_instance', classifier_tests.binary_classification_predict_proba_on_Predictor_instance),
    ('getting_single_predictions_classification', classifier_tests.getting_single_predictions_classification),
    ('feature_learning_getting_single_predictions_classification', classifier_tests.feature_learning_getting_single_predictions_classification),
    ('getting_single_predictions_nlp_date_multilabel_classification', classifier_tests.getting_single_predictions_nlp_date_multilabel_classification),
    ('categorical_ensembling_classification', classifier_tests.categorical_ensembling_classification),

    ('optimize_final_model_regression', regressor_tests.optimize_final_model_regression),
    ('perform_feature_selection_true_regression', regressor_tests.perform_feature_selection_true_regression),
    ('perform_feature_selection_false_regression', regressor_tests.perform_feature_selection_false_regression),
    ('perform_feature_scaling_true_regression', regressor_tests.perform_feature_scaling_true_regression),
    ('perform_feature_scaling_false_regression', regressor_tests.perform_feature_scaling_false_regression),
    ('getting_single_predictions_regression', regressor_tests.getting_single_predictions_regression),
    ('feature_learning_getting_single_predictions_regression', regressor_tests.feature_learning_getting_single_predictions_regression),
    ('categorical_ensembling_regression', regressor_tests.categorical_ensembling_regression)
])


def test_generator():
    for model_name in training_parameters['model_names']:
        for test_name, test in test_names.items():
            test_model_name = model_name
            if '_classification' in test_name and model_name is not None:
                test_model_name = model_name + 'Classifier'
            elif '_regression' in test_name and model_name is not None:
                test_model_name = model_name + 'Regressor'

            test.description = str(test_model_name) + '_' + test_name
            yield test, test_model_name
