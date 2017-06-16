from collections import OrderedDict
import math

from auto_ml import utils
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss, accuracy_score, explained_variance_score, mean_absolute_error, median_absolute_error, r2_score, log_loss, roc_auc_score
import numpy as np
from tabulate import tabulate

bad_vals_as_strings = set([str(float('nan')), str(float('inf')), str(float('-inf')), 'None', 'none', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'inf', '-inf', 'np.nan', 'numpy.nan'])

def advanced_scoring_classifiers(probas, actuals, name=None):
    # pandas Series don't play nice here. Make sure our actuals list is indeed a list
    actuals = list(actuals)
    predictions = list(probas)

    print('Here is our brier-score-loss, which is the default value we optimized for while training, and is the value returned from .score() unless you requested a custom scoring metric')
    print('It is a measure of how close the PROBABILITY predictions are.')
    if name != None:
        print(name)

    # Sometimes we will be given "flattened" probabilities (only the probability of our positive label), while other times we might be given "nested" probabilities (probabilities of both positive and negative, in a list, for each item).
    try:
        probas = [proba[1] for proba in probas]
    except:
        pass

    brier_score = brier_score_loss(actuals, probas)
    print(format(brier_score, '.4f'))


    print('\nHere is the trained estimator\'s overall accuracy (when it predicts a label, how frequently is that the correct label?)')
    predicted_labels = []
    for pred in probas:
        if pred >= 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    print(format(accuracy_score(y_true=actuals, y_pred=predicted_labels) * 100, '.1f') + '%')


    print('\nHere is a confusion matrix showing predictions vs. actuals by label:')
    #it would make sense to use sklearn's confusion_matrix here but it apparently has no labels
    #took this idea instead from: http://stats.stackexchange.com/a/109015
    conf = pd.crosstab(pd.Series(actuals), pd.Series(predicted_labels), rownames=['v Actual v'], colnames=['Predicted >'], margins=True)
    print(conf)

    #I like knowing the per class accuracy to see if the model is mishandling imbalanced data.
    #For example, if it is predicting 100% of observations to one class just because it is the majority
    #Wikipedia seems to call that Positive/negative predictive value
    print('\nHere is predictive value by class:')
    df = pd.concat([pd.Series(actuals,name='actuals'),pd.Series(predicted_labels,name='predicted')],axis=1)
    targets = list(df.predicted.unique())
    for i in range(0,len(targets)):
        tot_count = len(df[df.predicted==targets[i]])
        true_count = len(df[(df.predicted==targets[i]) & (df.actuals == targets[i])])
        print('Class: ',targets[i],'=',float(true_count)/tot_count)

    print('\nHere is the accuracy of our trained estimator at each level of predicted probabilities')
    print('For a verbose description of what this means, please visit the docs:')
    print('http://auto-ml.readthedocs.io/en/latest/analytics.html#interpreting-predicted-probability-buckets-for-classifiers')

    bucket_results = pd.qcut(probas, q=10, duplicates='drop')

    df_probas = pd.DataFrame(probas, columns=['Predicted Probability Of Bucket'])
    df_probas['Actual Probability of Bucket'] = actuals
    df_probas['Bucket Edges'] = bucket_results

    df_buckets = df_probas.groupby(df_probas['Bucket Edges'])
    print(tabulate(df_buckets.mean(), headers='keys', floatfmt='.4f', tablefmt='psql', showindex='always'))

    print('\n\n')
    return brier_score


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
    # pandas Series don't play nice here. Make sure our actuals list is indeed a list
    actuals = list(actuals)
    predictions = list(predictions)

    print('\n\n***********************************************')
    if name != None:
        print(name)
    print('Advanced scoring metrics for the trained regression model on this particular dataset:\n')

    # 1. overall RMSE
    print('Here is the overall RMSE for these predictions:')
    rmse = mean_squared_error(actuals, predictions)**0.5
    print(rmse)

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
    calculate_and_print_differences(predictions=predictions, actuals=actuals, name=name)

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
    return rmse

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


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default


    def score(self, estimator, X, y, took_log_of_y=False, advanced_scoring=False, verbose=2, name=None):
        X, y = utils.drop_missing_y_vals(X, y, output_column=None)

        if isinstance(estimator, GradientBoostingRegressor):
            X = X.toarray()

        predictions = estimator.predict(X)

        if took_log_of_y:
            for idx, val in enumerate(predictions):
                predictions[idx] = math.exp(val)

        try:
            score = self.scoring_func(y, predictions)
        except ValueError:

            bad_val_indices = []
            for idx, val in enumerate(y):
                if str(val) in bad_vals_as_strings:
                    bad_val_indices.append(idx)

            predictions = [val for idx, val in enumerate(predictions) if idx not in bad_val_indices]
            y = [val for idx, val in enumerate(y) if idx not in bad_val_indices]

            print('Found ' + str(len(bad_val_indices)) + ' null or infinity values in the y values. We will ignore these, and report the score on the rest of the dataset')
            score = self.scoring_func(y, predictions)

        if advanced_scoring == True:
            if hasattr(estimator, 'name'):
                print(estimator.name)
            advanced_scoring_regressors(predictions, y, verbose=verbose, name=name)
        return - 1 * score


class ClassificationScorer(object):

    def __init__(self, scoring_method=None):

        if scoring_method is None:
            scoring_method = 'brier_score_loss'

        self.scoring_method = scoring_method

        if callable(scoring_method):
            self.scoring_func = scoring_method
        else:
            self.scoring_func = scoring_name_function_map[scoring_method]


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default


    def clean_probas(self, probas):
        print('Warning: We have found some values in the predicted probabilities that fall outside the range {0, 1}')
        print('This is likely the result of a model being trained on too little data, or with a bad set of hyperparameters. If you get this warning while doing a hyperparameter search, for instance, you can probably safely ignore it')
        print('We will cap those values at 0 or 1 for the purposes of scoring, but you should be careful to have similar safeguards in place in prod if you use this model')
        if not isinstance(probas[0], list):
            probas = [min(max(pred, 0), 1) for pred in probas]
            return probas
        else:
            cleaned_probas = []
            for proba_tuple in probas:
                cleaned_tuple = []
                for item in proba_tuple:
                    cleaned_tuple.append(max(min(item, 1), 0))
                cleaned_probas.append(cleaned_tuple)
            return cleaned_probas



    def score(self, estimator, X, y, advanced_scoring=False):

        X, y = utils.drop_missing_y_vals(X, y, output_column=None)

        if isinstance(estimator, GradientBoostingClassifier):
            X = X.toarray()

        predictions = estimator.predict_proba(X)


        if self.scoring_method == 'brier_score_loss':
            # At the moment, Microsoft's LightGBM returns probabilities > 1 and < 0, which can break some scoring functions. So we have to take the max of 1 and the pred, and the min of 0 and the pred.
            probas = [max(min(row[1], 1), 0) for row in predictions]
            predictions = probas

        try:
            score = self.scoring_func(y, predictions)
        except ValueError as e:
            bad_val_indices = []
            for idx, val in enumerate(y):
                if str(val) in bad_vals_as_strings:
                    bad_val_indices.append(idx)

            predictions = [val for idx, val in enumerate(predictions) if idx not in bad_val_indices]
            y = [val for idx, val in enumerate(y) if idx not in bad_val_indices]

            print('Found ' + str(len(bad_val_indices)) + ' null or infinity values in the y values. We will ignore these, and report the score on the rest of the dataset')
            try:
                score = self.scoring_func(y, predictions)
            except ValueError:
                # Sometimes, particularly for a badly fit model using either too little data, or a really bad set of hyperparameters during a grid search, we can predict probas that are > 1 or < 0. We'll cap those here, while warning the user about them, because they're unlikely to occur in a model that's properly trained with enough data and reasonable params
                predictions = self.clean_probas(predictions)
                score = self.scoring_func(y, predictions)


        if advanced_scoring:
            return (-1 * score, predictions)
        else:
            return -1 * score
