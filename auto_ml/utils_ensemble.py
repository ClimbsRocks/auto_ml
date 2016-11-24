import numpy as np
import pandas as pd

try:
    from auto_ml.utils_scoring import advanced_scoring_regressors, advanced_scoring_classifiers
except ImportError:
    from ..auto_ml.utils_scoring import advanced_scoring_regressors, advanced_scoring_classifiers

import pathos

from sklearn.base import BaseEstimator, TransformerMixin


class Ensemble(object):

    def __init__(self, ensemble_predictors, type_of_estimator, method='average'):
        self.ensemble_predictors = ensemble_predictors
        self.type_of_estimator = type_of_estimator
        self.method = method


    # ################################
    # Get a dataframe that is all the predictions from all the sub-models
    # ################################
    # Note that we will get these predictions in parallel (relatively quick)

    def get_all_predictions(self, df):

        def get_predictions_for_one_estimator(estimator, df):
            estimator_name = estimator.named_steps['final_model'].name

            if self.type_of_estimator == 'regressor':
                predictions = estimator.predict(df)
            else:
                # For classifiers
                predictions = list(estimator.predict_proba(df))
            return_obj = {estimator_name: predictions}
            return return_obj


        # Open a new multiprocessing pool
        pool = pathos.multiprocessing.ProcessPool()

        # Since we may have already closed the pool, try to restart it
        try:
            pool.restart()
        except AssertionError as e:
            pass

        # Pathos doesn't like datasets beyond a certain size. So fall back on single, non-parallel predictions instead.
        try:
            predictions_from_all_estimators = pool.map(lambda predictor: get_predictions_for_one_estimator(predictor, df), self.ensemble_predictors, chunksize=100)
            predictions_from_all_estimators = list(predictions_from_all_estimators)
        except:
            predictions_from_all_estimators = map(lambda predictor: get_predictions_for_one_estimator(predictor, df), self.ensemble_predictors)
            predictions_from_all_estimators = list(predictions_from_all_estimators)


        # Once we have gotten all we need from the pool, close it so it's not taking up unnecessary memory
        pool.close()
        pool.join()


        results = {}
        for result_dict in predictions_from_all_estimators:
            results.update(result_dict)

        # if this is a single row we are getting predictions from, just return a dictionary with single values for all the predictions
        if isinstance(df, dict):
            return results
        else:
            predictions_df = pd.DataFrame.from_dict(results, orient='columns')

            return predictions_df

    # Gets summary stats on a set of predictions
    def get_summary_stats(self, predictions_df):
        summarized_predictions = []

        # Building in support for multi-class problems
        # Each row represents a single row that we want to get a prediction for
        # Each row is a list, with predicted probabilities from as many sub-estimators as we have trained
        # Each item in those subestimator lists represents the predicted probability of that class
        for row_idx, row in predictions_df.iterrows():
            row_results = {}

            if self.type_of_estimator == 'classifier':
                num_classes = len(row[0])
                for class_prediction_idx in range(num_classes):
                    class_preds = [estimator_prediction[class_prediction_idx] for estimator_prediction in row]

                    class_summarized_predictions = self.get_summary_stats_from_row(class_preds, prefix='subpredictor_class=' + str(class_prediction_idx))
                    row_results.update(class_summarized_predictions)
            else:
                row_summarized = self.get_summary_stats_from_row(row, prefix='subpredictors_')
                row_results.update(row_summarized)

            summarized_predictions.append(row_results)

        results_df = pd.DataFrame(summarized_predictions)
        return results_df


    def get_summary_stats_from_row(self, row, prefix=''):
        results = {}

        results[prefix + '_median'] = np.median(row)
        results[prefix + '_average'] = np.average(row)
        results[prefix + '_max'] = np.max(row)
        results[prefix + '_min'] = np.min(row)
        results[prefix + '_range'] = results[prefix + '_max'] - results[prefix + '_min']

        return results





    # ################################
    # Public API to get a single prediction from each row, where that single prediction is somehow an ensemble of all our trained subpredictors
    # ################################

    def predict(self, df):

        predictions_df = self.get_all_predictions(df)

        # If this is just a single dictionary we're getting predictions from:
        if isinstance(df, dict):
            # predictions_df is just a dictionary where all the values are the predicted values from one of our subpredictors. we'll want that as a list
            predicted_vals = predictions_df.values()
            if self.method == 'median':
                return np.median(predicted_vals)
            elif self.method == 'average' or self.method == 'mean':
                return np.average(predicted_vals)
            elif self.method == 'max':
                return np.max(predicted_vals)
            elif self.method == 'min':
                return np.min(predicted_vals)

        else:

            summarized_predictions = []
            for idx, row in predictions_df.iterrows():
                if self.method == 'median':
                    summarized_predictions.append(np.median(row))
                elif self.method == 'average' or self.method == 'mean':
                    summarized_predictions.append(np.average(row))
                elif self.method == 'max':
                    summarized_predictions.append(np.max(row))
                elif self.method == 'min':
                    summarized_predictions.append(np.min(row))


            return summarized_predictions

    # ################################
    # Public API to get a propbability predictions from each row, where each row will have a list of values, representing the probability of that class
    # ################################
    def predict_proba(self, df):

        predictions_df = self.get_all_predictions(df)

        if isinstance(df, dict):
            # predictions_df is just a dictionary where all the values are the predicted values from one of our subpredictors. we'll want that as a list
            predicted_vals = predictions_df.values()
            predicted_vals = [pred[1] for pred in predicted_vals]
            if self.method == 'median':
                return np.median(predicted_vals)
            elif self.method == 'average' or self.method == 'mean':
                return np.average(predicted_vals)
            elif self.method == 'max':
                return np.max(predicted_vals)
            elif self.method == 'min':
                return np.min(predicted_vals)

        else:

            summarized_predictions = []

            # Building in support for multi-class problems
            # Each row represents a single row that we want to get a prediction for
            # Each row is a list, with predicted probabilities from as many sub-estimators as we have trained
            # Each item in those subestimator lists represents the predicted probability of that class
            for row_idx, row in predictions_df.iterrows():
                row_ensembled_probabilities = []

                num_classes = len(row[0])
                for class_prediction_idx in range(num_classes):
                    class_preds = [estimator_prediction[class_prediction_idx] for estimator_prediction in row]

                    if self.method == 'median':
                        row_ensembled_probabilities.append(np.median(class_preds))
                    elif self.method == 'average' or self.method == 'mean':
                        row_ensembled_probabilities.append(np.average(class_preds))
                    elif self.method == 'max':
                        row_ensembled_probabilities.append(np.max(class_preds))
                    elif self.method == 'min':
                        row_ensembled_probabilities.append(np.min(class_preds))
                summarized_predictions.append(row_ensembled_probabilities)
            return summarized_predictions



    # ################################
    # Find the best enemble method that is not ml
    # ################################
    def find_best_ensemble_method(self, df, actuals):
        predictions_df = self.get_all_predictions(df)

        summary_df = self.get_summary_stats(predictions_df)

        for method in ['min', 'max', 'average', 'median']:
            print(method)
            for col in summary_df.columns:
                if method in col:
                    if self.type_of_estimator == 'regressor':
                        advanced_scoring_regressors(summary_df[col], actuals, name=method)
                    else:
                        advanced_scoring_classifiers(summary_df[col], actuals)




class AddEnsembledPredictions(BaseEstimator, TransformerMixin):

    def __init__(self, ensembler, type_of_estimator):
        self.ensembler = ensembler
        self.type_of_estimator = type_of_estimator


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        predictions = self.ensembler.get_all_predictions(X)

        summarized_predictions = self.ensembler.get_summary_stats(predictions)

        # If this is a classifier, the predictions from each estimator will be an array of predicted probabilities.
        # We will need to unpack that list
        if self.type_of_estimator == 'classifier':
            flattened_predictions_dfs = []
            for col in predictions:
                flattened_df = pd.DataFrame(predictions[col].tolist())
                col_names = []
                for col_num in flattened_df:
                    col_names.append('subpredictor_' + str(col) + '_class=' + str(col_num))

                flattened_df.columns = col_names

                # Drop the first column in the DataFrame, since it just contains the inverse data from the other column(s)
                flattened_df = flattened_df.drop(flattened_df.columns[0], axis=1)

                flattened_predictions_dfs.append(flattened_df)

            predictions = pd.concat(flattened_predictions_dfs, axis=1)

        X = X.reset_index(drop=True)
        # X = pd.concat([X, predictions, summarized_predictions], axis=1)
        X = pd.concat([predictions, summarized_predictions], axis=1)

        return X
