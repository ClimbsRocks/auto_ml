from collections import OrderedDict
import multiprocessing
import pathos
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

import math
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss, accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score, log_loss, roc_auc_score
import numpy as np

def advanced_scoring_classifiers(probas, actuals, name=None):

    print('Here is our brier-score-loss, which is the default value we optimized for while training, and is the value returned from .score() unless you requested a custom scoring metric')
    print('It is a measure of how close the PROBABILITY predictions are.')
    if name != None:
        print(name)

    # Sometimes we will be given "flattened" probabilities (only the probability of our positive label), while other times we might be given "nested" probabilities (probabilities of both positive and negative, in a list, for each item).
    try:
        probas = [proba[1] for proba in probas]
    except:
        pass

    print(format(brier_score_loss(actuals, probas), '.4f'))

    print('\nHere is the trained estimator\'s overall accuracy (when it predicts a label, how frequently is that the correct label?)')
    predicted_labels = []
    for pred in probas:
        if pred >= 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    print(format(accuracy_score(y_true=actuals, y_pred=predicted_labels) * 100, '.1f') + '%')

    print('Here is the accuracy of our trained estimator at each level of predicted probabilities')

    # create summary dict
    summary_dict = OrderedDict()
    for num in range(0, 110, 10):
        summary_dict[num] = []

    for idx, proba in enumerate(probas):
        proba = math.floor(int(proba * 100) / 10) * 10
        summary_dict[proba].append(actuals[idx])

    for k, v in summary_dict.items():
        if len(v) > 0:
            print('Predicted probability: ' + str(k) + '%')
            actual = sum(v) * 1.0 / len(v)

            # Format into a prettier number
            actual = round(actual * 100, 0)
            print('Actual: ' + str(actual) + '%')
            print('# preds: ' + str(len(v)) + '\n')

    print('\n\n')

def calculate_and_print_differences(predictions, actuals, name=None):
    pos_differences = []
    neg_differences = []
    # Technically, we're ignoring cases where we are spot on
    for idx, pred in enumerate(predictions):
        difference = pred - actuals[idx]
        if difference > 0:
            pos_differences.append(difference)
        elif difference < 0:
            neg_differences.append(difference)

    if name != None:
        print(name)
    print('Count of positive differences (prediction > actual):')
    print(len(pos_differences))
    print('Count of negative differences:')
    print(len(neg_differences))
    if len(pos_differences) > 0:
        print('Average positive difference:')
        print(sum(pos_differences) * 1.0 / len(pos_differences))
    if len(neg_differences) > 0:
        print('Average negative difference:')
        print(sum(neg_differences) * 1.0 / len(neg_differences))


def advanced_scoring_regressors(predictions, actuals, verbose=2, name=None):

    print('\n\n***********************************************')
    if name != None:
        print(name)
    print('Advanced scoring metrics for the trained regression model on this particular dataset:\n')

    # 1. overall RMSE
    print('Here is the overall RMSE for these predictions:')
    print(mean_squared_error(actuals, predictions)**0.5)

    # 2. overall avg predictions
    print('\nHere is the average of the predictions:')
    print(sum(predictions) * 1.0 / len(predictions))

    # 3. overall avg actuals
    print('\nHere is the average actual value on this validation set:')
    print(sum(actuals) * 1.0 / len(actuals))

    # 2(a). median predictions
    print('\nHere is the median prediction:')
    print(np.median(predictions))

    # 3(a). median actuals
    print('\nHere is the median actual value:')
    print(np.median(actuals))

    # 4. avg differences (not RMSE)
    print('\nHere is the mean absolute error:')
    print(mean_absolute_error(actuals, predictions))

    print('\nHere is the median absolute error (robust to outliers):')
    print(median_absolute_error(actuals, predictions))

    print('\nHere is the explained variance:')
    print(explained_variance_score(actuals, predictions))

    print('\nHere is the R-squared value:')
    print(r2_score(actuals, predictions))

    # 5. pos and neg differences
    calculate_and_print_differences(predictions, actuals, name=name)
    # 6.

    actuals_preds = list(zip(actuals, predictions))
    # Sort by PREDICTED value, since this is what what we will know at the time we make a prediction
    actuals_preds.sort(key=lambda pair: pair[1])
    actuals_sorted = [act for act, pred in actuals_preds]
    predictions_sorted = [pred for act, pred in actuals_preds]

    if verbose > 2:
        print('Here\'s how the trained predictor did on each successive decile (ten percent chunk) of the predictions:')
        for i in range(1,10):
            print('\n**************')
            print('Bucket number:')
            print(i)
            # There's probably some fenceposting error here
            min_idx = int((i - 1) / 10.0 * len(actuals_sorted))
            max_idx = int(i / 10.0 * len(actuals_sorted))
            actuals_for_this_decile = actuals_sorted[min_idx:max_idx]
            predictions_for_this_decile = predictions_sorted[min_idx:max_idx]

            print('Avg predicted val in this bucket')
            print(sum(predictions_for_this_decile) * 1.0 / len(predictions_for_this_decile))
            print('Avg actual val in this bucket')
            print(sum(actuals_for_this_decile) * 1.0 / len(actuals_for_this_decile))
            print('RMSE for this bucket')
            print(mean_squared_error(actuals_for_this_decile, predictions_for_this_decile)**0.5)
            calculate_and_print_differences(predictions_for_this_decile, actuals_for_this_decile)

    print('')
    print('\n***********************************************\n\n')

def rmse_func(y, predictions):
    return mean_squared_error(y, predictions)**0.5


scoring_name_function_map = {
    'rmse': rmse_func
    , 'median_absolute_error': median_absolute_error
    , 'r2': r2_score
    , 'r-squared': r2_score
    , 'mean_absolute_error': mean_absolute_error
    , 'accuracy': accuracy_score
    , 'accuracy_score': accuracy_score
    , 'log_loss': log_loss
    , 'roc_auc': roc_auc_score
    , 'brier_score_loss': brier_score_loss
}


class RegressionScorer(object):

    def __init__(self, scoring_method=None):

        if scoring_method is None:
            scoring_method = 'rmse'

        self.scoring_method = scoring_method

        if callable(scoring_method):
            self.scoring_func = scoring_method
        else:
            self.scoring_func = scoring_name_function_map[scoring_method]

        self.scoring_method = scoring_method


    def score(self, estimator, X, y, took_log_of_y=False, advanced_scoring=False, verbose=2, name=None):
        if isinstance(estimator, GradientBoostingRegressor):
            X = X.toarray()

        predictions = estimator.predict(X)

        if took_log_of_y:
            for idx, val in enumerate(predictions):
                predictions[idx] = math.exp(val)

        score = self.scoring_func(y, predictions)
        # if scoring == 'rmse':
        #     score = mean_squared_error(y, predictions)**0.5
        # elif scoring == 'median_absolute_error':
        #     score = median_absolute_error(y, predictions)

        if advanced_scoring == True:
            if hasattr(estimator, 'name'):
                print(estimator.name)
            advanced_scoring_regressors(predictions, y, verbose=verbose, name=name)
        return - 1 * score


class ClassificationScorer(object):

    def __init__(self, scoring_method=None):

        if scoring_method is None:
            scoring_method = 'brier_score_loss'

        if callable(scoring_method):
            self.scoring_func = scoring_method
        else:
            self.scoring_func = scoring_name_function_map[scoring_method]

        self.scoring_method = scoring_method


    def score(self, estimator, X, y, advanced_scoring=False):
        if isinstance(estimator, GradientBoostingClassifier):
            X = X.toarray()
        # clean_ys = []
        # # try:
        # for val in y:
        #     val = int(val)
        #     clean_ys.append(val)
        # y = clean_ys
        # except:
        #     pass

        predictions = estimator.predict_proba(X)

        if self.scoring_method == 'brier_score_loss':
            probas = [row[1] for row in predictions]
            predictions = probas
        # score = brier_score_loss(y, probas)

        # if self.scoring_method in ['accuracy_score', 'accuracy']:
        #     predictions = [1 if x[1] >= 0.5 else 0 for x in predictions]

        score = self.scoring_func(y, predictions)

        if advanced_scoring:
            return (-1 * score, predictions)
        else:
            return -1 * score
