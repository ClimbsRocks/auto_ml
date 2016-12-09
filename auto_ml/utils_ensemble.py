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


        # Don't bother parallelizing if this is a single dictionary
        if isinstance(df, dict):
            predictions_from_all_estimators = map(lambda predictor: get_predictions_for_one_estimator(predictor, df), self.ensemble_predictors)
            predictions_from_all_estimators = list(predictions_from_all_estimators)

        else:
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
            try:
                pool.join()
            except AssertionError:
                pass


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

        # print('predictions_df inside get_summary_stats')
        # print(predictions_df)

        # TODO(PRESTON): Super hacky. see if we can just build in native support for dictionaries instead
        # if isinstance(predictions_df, dict):
        #     predictions_df = pd.DataFrame(predictions_df)
        summarized_predictions = []

        # Building in support for multi-class problems
        # Each row represents a single row that we want to get a prediction for
        # Each row is a list, with predicted probabilities from as many sub-estimators as we have trained
        # Each item in those subestimator lists represents the predicted probability of that class
        if isinstance(predictions_df, dict):
            flattened_dict = []
            for k, v in predictions_df.items():
                flattened_dict.append(v)

            summarized_predictions.append(self.process_one_row(flattened_dict))

        else:
            for row_idx, row in predictions_df.iterrows():

                row_results = self.process_one_row(row)
                summarized_predictions.append(row_results)

        results_df = pd.DataFrame(summarized_predictions)
        return results_df

    def process_one_row(self, row):
        row_results = {}

        if self.type_of_estimator == 'classifier':
            # TODO(PRESTON): This is erroring out when we use 'ml' as our ensemble method
            # TypeError: object of type 'numpy.float64' has no len()
            num_classes = len(row[0])
            for class_prediction_idx in range(num_classes):
                class_preds = [estimator_prediction[class_prediction_idx] for estimator_prediction in row]

                class_summarized_predictions = self.get_summary_stats_from_row(class_preds, prefix='subpredictor_class=' + str(class_prediction_idx))
                row_results.update(class_summarized_predictions)
        else:
            row_summarized = self.get_summary_stats_from_row(row, prefix='subpredictors_')
            row_results.update(row_summarized)

        return row_results

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
            # NOTE: this only works for binary classification at the moment
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

    def __init__(self, ensembler, type_of_estimator, include_original_X=False):
        self.ensembler = ensembler
        self.type_of_estimator = type_of_estimator
        self.include_original_X = include_original_X


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        predictions = self.ensembler.get_all_predictions(X)

        # print('predictions inside AddEnsembledPredictions.predict() when using ML to ensemble together:')
        # print(predictions)

        summarized_predictions = self.ensembler.get_summary_stats(predictions)
        # print('summarized_predictions inside AddEnsembledPredictions.predict() when using ML to ensemble together:')
        # print(summarized_predictions)

        # print('summarized_predictions inside AddEnsembledPredictions')
        # print(summarized_predictions)

        # If this is a classifier, the predictions from each estimator will be an array of predicted probabilities.
        # We will need to unpack that list
        # print('predictions')
        # print(predictions)
        if self.type_of_estimator == 'classifier':

            flattened_predictions_dfs = []
            for col in predictions:
                # print('col in predictions inside AddEnsembledPredictions')
                # print(col)

                # each column of values in predictions is a list, containing predicted probabilities for each label (two labels for binary classification, more labelse for multi-label classification)
                if isinstance(predictions, dict):
                    flattened_df = pd.DataFrame(predictions[col])
                else:
                    flattened_df = pd.DataFrame(predictions[col].tolist())
                # So this flattened_df is now a df where each column represents the predicted probabilities for one of those classes, for this given subpredictor


                col_names = []
                for col_num in flattened_df:
                    col_names.append('subpredictor_' + str(col) + '_class=' + str(col_num))

                flattened_df.columns = col_names

                # Drop the first column in the DataFrame, since it just contains the inverse data from the other column(s)
                # TODO(PRESTON): somewhere we appear to be inverting our probability predictions. The line below might be the culprit
                flattened_df = flattened_df.drop(flattened_df.columns[0], axis=1)

                flattened_predictions_dfs.append(flattened_df)

            # predictions is now a DataFrame where each column holds the positive label probabilities from one of our classifiers
            # Collectively, predictions holds the probabilities of the positive case for all of our classifiers
            predictions = pd.concat(flattened_predictions_dfs, axis=1)

        if isinstance(X, dict):
            X = pd.DataFrame(X, index=[0])


        # Either only include predicted vals, or include predictions along with all the original features of X
        if self.include_original_X is True:
            X = X.reset_index(drop=True)
            X = pd.concat([X, predictions, summarized_predictions], axis=1)
        else:
            X = pd.concat([predictions, summarized_predictions], axis=1)



        # print('X at the end of AddEnsembledPredictions')

        return X
