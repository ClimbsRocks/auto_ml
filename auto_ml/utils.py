from collections import OrderedDict
import csv
import datetime
import math
import numpy as np
import os
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, RANSACRegressor,                 LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

import pandas as pd

import xgboost as xgb

def split_output(X, output_column_name, verbose=False):
    y = []
    for row in X:
        y.append(
            row.pop(output_column_name, None)
        )

    if verbose:
        print('Just to make sure that your y-values make sense, here are the first 100 sorted values:')
        print(sorted(y)[:100])
        print('And here are the final 100 sorted values:')
        print(sorted(y)[-100:])

    return X, y


# Hyperparameter search spaces for each model
def get_search_params(model_name):
    grid_search_params = {

        'XGBClassifier': {
            'max_depth': [1, 3, 8, 25],
            # 'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76]
        },
        'XGBRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 3, 8, 25],
            # 'lossl': ['ls', 'lad', 'huber', 'quantile'],
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            # 'objective': ['reg:linear', 'reg:gamma'],
            # 'learning_rate': [0.01, 0.1],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76],

        },
        'GradientBoostingRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 3, 8, 25],
            'max_features': ['sqrt', 'log2', None],
            # 'loss': ['ls', 'lad', 'huber', 'quantile']
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            'loss': ['ls', 'lad', 'huber'],
            # 'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
            'subsample': [0.5, 1.0]
        },
        'GradientBoostingClassifier': {
            'loss': ['deviance', 'exponential'],
            'max_depth': [1, 3, 8, 25],
            'max_features': ['sqrt', 'log2', None],
            # 'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76]

        },

        'LogisticRegression': {
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['newton-cg', 'lbfgs', 'sag']
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'RandomForestClassifier': {
            'criterion': ['entropy', 'gini'],
            'class_weight': [None, 'balanced'],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RandomForestRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RidgeClassifier': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'Ridge': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'ExtraTreesRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'AdaBoostRegressor': {
            'base_estimator': [None, LinearRegression(n_jobs=-1)],
            'loss': ['linear','square','exponential']
        },
        'RANSACRegressor': {
            'min_samples': [None, .1, 100, 1000, 10000],
            'stop_probability': [0.99, 0.98, 0.95, 0.90]
        },
        'Lasso': {
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'ElasticNet': {
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'LassoLars': {
            'positive': [True, False],
            'max_iter': [50, 100, 250, 500, 1000]
        },

        'OrthogonalMatchingPursuit': {
            'n_nonzero_coefs': [None, 3, 5, 10, 25, 50, 75, 100, 200, 500]
        },

        'BayesianRidge': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001]
        },

        'ARDRegression': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'alpha_2': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001],
            'threshold_lambda': [100, 1000, 10000, 100000, 1000000]
        },

        'SGDRegressor': {
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'alpha': [.0000001, .000001, .00001, .0001, .001]
        },

        'PassiveAggressiveRegressor': {
            'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        },

        'SGDClassifier': {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'class_weight': ['balanced', None]
        },

        'Perceptron': {
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'class_weight': ['balanced', None]
        },

        'PassiveAggressiveClassifier': {
            'loss': ['hinge', 'squared_hinge'],
            'class_weight': ['balanced', None],
            'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        }

    }

    return grid_search_params[model_name]



class BasicDataCleaning(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions=None):
        self.column_descriptions = column_descriptions
        self.vals_to_del = set([None, float('Inf'), 'ignore', 'nan', 'NaN', 'Inf', 'inf', 'None', ''])
        self.vals_to_ignore = set(['regressor', 'classifier', 'output', 'ignore'])
        self.tfidfvec = TfidfVectorizer()

    def fit(self, X, y=None):


        inputflag=False
        text_col_indicators = set(['text', 'nlp'])

        #Condition check if there is text or nlp field only then do tfidf
        #follow loop will be excuted only one time , must see if there is any other logic.
        #currently this is needed becuase this is the only way to get to know if there is any senetence as inputs in columns
        #TODO alternatively any option from config file would be helpful which will remove this following loop
        for row in X:
            for key, val in row.items():
                column_desciption = self.column_descriptions.get(key)
                if column_desciption in text_col_indicators:
                    inputflag = True
                    break

        # must look at an alternate way of doing this
        if inputflag:
            corpus = []
            for row in X:
                for key, val in row.items():
                    col_desc = self.column_descriptions.get(key)
                    if col_desc in text_col_indicators:
                            corpus.append(val)
            self.tfidfvec.fit(corpus)
            return self
        else:
            return self

    def transform(self, X, y=None):
        clean_X = []
        deleted_values_sample = []
        deleted_info = {}
        text_col_indicators = set(['text', 'nlp'])


        for row in X:
            clean_row = {}
            for key, val in row.items():
                col_desc = self.column_descriptions.get(key)

                if col_desc == 'categorical':
                    clean_row[key] = str(val)
                elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
                    if val not in self.vals_to_del and pd.notnull(val):
                        try:
                            try:
                                # Try to float the value
                                floated_val = float(val)
                            except:
                                # If we can't float it directly, try to remove any commas that might be in the string form of a number
                                # For example, 12,845 cannot be floated direclty, but 12845 can be.
                                floated_val = float(val.replace(',', ''))
                            clean_row[key] = floated_val
                        except Exception as e:
                            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                            print('We were not able to automatically process this column. Here is some information to help you debug:')
                            print('column name:')
                            print(key)
                            print('value:')
                            print(val)
                            print('Type of this column as passed into column_descriptions:')
                            print(col_desc)
                            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                            raise
                elif col_desc == 'date':
                    clean_row = add_date_features(val, clean_row, key)
                # if input column contains text, then in such a case calculated tfidf which if already fitted before transform
                elif col_desc in text_col_indicators:
                    #add keys as features and tfvector values as values into cleanrow dictionary object
                    keys = self.tfidfvec.get_feature_names()
                    tfvec = self.tfidfvec.transform([val]).toarray()
                    for i in range(len(tfvec[0])):
                        clean_row[keys[i]] = tfvec[0][i]
                elif col_desc in self.vals_to_ignore:
                    pass
                else:
                    # If we have gotten here, the value is not any that we recognize
                    # This is most likely a typo that the user would want to be informed of, or a case while we're developing on auto_ml itself.
                    # In either case, it's useful to log it.
                    if len(deleted_values_sample) < 10:
                        deleted_values_sample.append(row[key])
                    deleted_info[key] = col_desc


            clean_X.append(clean_row)

        if len(deleted_values_sample) > 0:
            print('When transforming the data, we have encountered some values in column_descriptions that are not currently supported. The values stored at these keys have been deleted to allow the rest of the pipeline to run. Here\'s some info about these columns:' )
            print(deleted_info)
            print('And some example values from these columns:')
            print(deleted_values_sample)


        return clean_X

def add_date_features(date_val, row, date_col):

    row[date_col + '_day_of_week'] = str(date_val.weekday())
    row[date_col + '_hour'] = date_val.hour

    minutes_into_day = date_val.hour * 60 + date_val.minute

    if row[date_col + '_day_of_week'] in (5,6):
        row[date_col + '_is_weekend'] = True
    elif row[date_col + '_day_of_week'] == 4 and row[date_col + '_hour'] > 16:
        row[date_col + '_is_weekend'] = True
    else:
        row[date_col + '_is_weekend'] = False

        # Grab rush hour times for the weekdays.
        # We are intentionally not grabbing them for the weekends, since weekend behavior is likely very different than weekday behavior.
        if minutes_into_day < 120:
            row[date_col + '_is_late_night'] = True
        elif minutes_into_day < 11.5 * 60:
            row[date_col + '_is_off_peak'] = True
        elif minutes_into_day < 13.5 * 60:
            row[date_col + '_is_lunch_rush_hour'] = True
        elif minutes_into_day < 17.5 * 60:
            row[date_col + '_is_off_peak'] = True
        elif minutes_into_day < 20 * 60:
            row[date_col + '_is_dinner_rush_hour'] = True
        elif minutes_into_day < 22.5 * 60:
            row[date_col + '_is_off_peak'] = True
        else:
            row[date_col + '_is_late_night'] = True

    return row


def get_model_from_name(model_name):
    model_map = {
        # Classifiers
        'LogisticRegression': LogisticRegression(n_jobs=-2),
        'RandomForestClassifier': RandomForestClassifier(n_jobs=-2),
        'RidgeClassifier': RidgeClassifier(),
        'XGBClassifier': xgb.XGBClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),

        'SGDClassifier': SGDClassifier(n_jobs=-1),
        'Perceptron': Perceptron(n_jobs=-1),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),

        # Regressors
        'LinearRegression': LinearRegression(n_jobs=-2),
        'RandomForestRegressor': RandomForestRegressor(n_jobs=-2),
        'Ridge': Ridge(),
        'XGBRegressor': xgb.XGBRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor(n_jobs=-1),
        'AdaBoostRegressor': AdaBoostRegressor(n_estimators=5),
        'RANSACRegressor': RANSACRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(presort=False),

        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),
        'SGDRegressor': SGDRegressor(shuffle=False),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(shuffle=False),

        # Clustering
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=8)
    }
    return model_map[model_name]


# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# It also gives us more granular control over things like turning the input for GradientBoosting into dense matrices, or appending a set of dummy 1's to the end of sparse matrices getting predictions from XGBoost.
class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model, model_name, X_train=None, y_train=None, ml_for_analytics=False, type_of_estimator='classifier', output_column=None):

        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator
        # This is purely a placeholder so we can set it if we have to if this is a subpredictor
        # In that case, we will set it after the whole pipeline has trained and we are abbreviating the subpredictor pipeline
        self.output_column = output_column

        if self.type_of_estimator == 'classifier':
            self._scorer = brier_score_loss_wrapper
        else:
            self._scorer = rmse_scoring


    def fit(self, X, y):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X_fit = scipy.sparse.hstack((X, ones))

        else:
            X_fit = X

        model_to_fit = get_model_from_name(self.model_name)

        if self.ml_for_analytics:
            self.feature_ranges = []

            # Grab the ranges for each feature
            if scipy.sparse.issparse(X_fit):
                for col_idx in range(X_fit.shape[1]):
                    col_vals = X_fit.getcol(col_idx).toarray()
                    col_vals = sorted(col_vals)

                    # if the entire range is 0 - 1, just append the entire range.
                    # This works well for binary variables.
                    # This will break when we do min-max normalization to the range of 0,1
                    # TODO(PRESTON): check if all values are 0's and 1's, or if we have any values in between.
                    if col_vals[0] == 0 and col_vals[-1] == 1:
                        self.feature_ranges.append(1)
                    else:
                        twentieth_percentile = col_vals[int(len(col_vals) * 0.2)]
                        eightieth_percentile = col_vals[int(len(col_vals) * 0.8)]

                        self.feature_ranges.append(eightieth_percentile - twentieth_percentile)
                    del col_vals


        self.model = get_model_from_name(self.model_name)

        self.model.fit(X_fit, y)

        return self


    def score(self, X, y):
        # At the time of writing this, GradientBoosting does not support sparse matrices for predictions
        if self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X = X.todense()

        if self._scorer is not None:
            if self.type_of_estimator == 'regressor':
                return self._scorer(self, X, y)
            elif self.type_of_estimator == 'classifier':
                return self._scorer(self, X, y)


        else:
            return self.model.score(X, y)


    def predict_proba(self, X):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X = scipy.sparse.hstack((X, ones))

        if self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X = X.todense()

        try:
            return self.model.predict_proba(X)
        except AttributeError:
            print('This model has no predict_proba method. Returning results of .predict instead.')
            raw_predictions = self.model.predict(X)
            tupled_predictions = []
            for prediction in raw_predictions:
                if prediction == 1:
                    tupled_predictions.append([0,1])
                else:
                    tupled_predictions.append([1,0])
            return tupled_predictions


    def predict(self, X):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X_predict = scipy.sparse.hstack((X, ones))

        elif self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X_predict = X.todense()

        else:
            X_predict = X

        return self.model.predict(X_predict)


def advanced_scoring_classifiers(probas, actuals):
    print('Here is how our trained estimator does at each level of predicted probabilities')
    # create summary dict
    summary_dict = OrderedDict()
    for num in range(0, 100, 10):
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


def write_gs_param_results_to_file(trained_gs, most_recent_filename):

    timestamp_time = datetime.datetime.now()
    write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp_time)

    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = 'pipeline_grid_search_results.csv'
    write_header = False
    if not os.path.isfile(file_name):
        write_header = True

    with open(file_name, 'a') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(['timestamp', 'scorer', 'best_score', 'all_grid_scores'])
        writer.writerow([timestamp_time, scorer, best_score, grid_scores])


def write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp):

    timestamp_time = timestamp
    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = most_recent_filename

    write_header = False
    make_header = False
    if not os.path.isfile(most_recent_filename):
        header_row = ['timestamp', 'scorer', 'best_score', 'cv_mean', 'cv_all']
        write_header = True
        make_header = True

    rows_to_write = []

    for score in grid_scores:

        row = [timestamp_time, scorer, best_score, score[1], score[2]]

        for k, v in score[0].items():
            if make_header:
                header_row.append(k)
            row.append(v)
        rows_to_write.append(row)
        make_header = False


    with open(file_name, 'a') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(header_row)
        for row in rows_to_write:
            writer.writerow(row)


def get_feature_selection_model_from_name(type_of_estimator, model_name):
    # TODO(PRESTON): eventually let threshold be user-configurable (or grid_searchable)
    # TODO(PRESTON): optimize the params used here
    model_map = {
        'classifier': {
            'SelectFromModel': SelectFromModel(RandomForestClassifier(n_jobs=-1)),
            'RFECV': RFECV(estimator=RandomForestClassifier(n_jobs=-1), step=0.1),
            'GenericUnivariateSelect': GenericUnivariateSelect(),
            'RandomizedSparse': RandomizedLogisticRegression(),
            'KeepAll': 'KeepAll'
        },
        'regressor': {
            'SelectFromModel': SelectFromModel(RandomForestRegressor(n_jobs=-1)),
            'RFECV': RFECV(estimator=RandomForestRegressor(n_jobs=-1), step=0.1),
            'GenericUnivariateSelect': GenericUnivariateSelect(),
            'RandomizedSparse': RandomizedLasso(),
            'KeepAll': 'KeepAll'
        }
    }

    return model_map[type_of_estimator][model_name]


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, type_of_estimator, feature_selection_model='SelectFromModel'):


        self.type_of_estimator = type_of_estimator
        self.feature_selection_model = feature_selection_model


    def fit(self, X, y=None):

        self.selector = get_feature_selection_model_from_name(self.type_of_estimator, self.feature_selection_model)

        if self.selector == 'KeepAll':
            if scipy.sparse.issparse(X):
                num_cols = X.shape[0]
            else:
                num_cols = len(X[0])

            self.support_mask = [True for col_idx in range(num_cols) ]
        else:
            self.selector.fit(X, y)
            self.support_mask = self.selector.get_support()
        return self


    def transform(self, X, y=None):
        if self.selector == 'KeepAll':
            return X
        else:
            return self.selector.transform(X)


def rmse_scoring(estimator, X, y, took_log_of_y=False):
    if isinstance(estimator, GradientBoostingRegressor):
        X = X.toarray()
    predictions = estimator.predict(X)
    if took_log_of_y:
        for idx, val in enumerate(predictions):
            predictions[idx] = math.exp(val)
    rmse = mean_squared_error(y, predictions)**0.5
    return - 1 * rmse


def brier_score_loss_wrapper(estimator, X, y, advanced_scoring=False):
    if isinstance(estimator, GradientBoostingClassifier):
        X = X.toarray()

    predictions = estimator.predict_proba(X)
    probas = [row[1] for row in predictions]
    score = brier_score_loss(y, probas)
    if advanced_scoring:
        return (-1 * score, probas)
    else:
        return -1 * score

# Used for CustomSparseScaler
def get_all_attribute_names(list_of_dictionaries, cols_to_avoid):
    attribute_hash = {}
    for dictionary in list_of_dictionaries:
        for k, v in dictionary.items():
            attribute_hash[k] = True

    # All of the columns in column_descriptions should be avoided. They're probably either categorical or NLP data, both of which cannot be scaled.

    attribute_list = [k for k, v in attribute_hash.items() if k not in cols_to_avoid]
    return attribute_list


# Scale sparse data to the 90th and 10th percentile
# Only do so for values that actuall exist (do absolutely nothing with rows that do not have this data point)
class CustomSparseScaler(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions, truncate_large_values=False, perform_feature_scaling=True):
        self.column_descriptions = column_descriptions
        self.cols_to_avoid = set([k for k, v in column_descriptions.items()])
        self.truncate_large_values = truncate_large_values
        self.perform_feature_scaling = perform_feature_scaling


    def fit(self, X, y=None):
        if self.perform_feature_scaling:
            attribute_list = get_all_attribute_names(X, self.column_descriptions)

            attributes_per_round = [[], [], []]

            attributes_summary = {}

            # Randomly assign each attribute to one of three buckets
            # We will summarize the data in three separate iterations, to avoid duplicating too much data in memory at any one point.
            for attribute in attribute_list:
                bucket_idx = int(random.random() * 3)
                attributes_per_round[bucket_idx].append(attribute)
                attributes_summary[attribute] = []

            for bucket in attributes_per_round:

                attributes_to_summarize = set(bucket)

                for row in X:
                    for k, v in row.items():
                        if k in attributes_to_summarize:
                            attributes_summary[k].append(v)

                for attribute in bucket:

                    # Sort our collected data for that column
                    attributes_summary[attribute].sort()
                    col_vals = attributes_summary[attribute]
                    tenth_percentile = col_vals[int(0.05 * len(col_vals))]
                    ninetieth_percentile = col_vals[int(0.95 * len(col_vals))]

                    # It's probably not a great idea to pass in as continuous data a column that has 0 variation from it's 10th to it's 90th percentiles, but we'll protect against it here regardless
                    col_range = ninetieth_percentile - tenth_percentile
                    if col_range > 0:
                        attributes_summary[attribute] = [tenth_percentile, ninetieth_percentile, ninetieth_percentile - tenth_percentile]
                    else:
                        del attributes_summary[attribute]
                        self.cols_to_avoid.add(attribute)

                    del col_vals

            self.attributes_summary = attributes_summary

        return self


    # Perform basic min/max scaling, with the minor caveat that our min and max values are the 10th and 90th percentile values, to avoid outliers.
    def transform(self, X, y=None):
        if self.perform_feature_scaling:
            for row in X:
                for k, v in row.items():
                    if k not in self.cols_to_avoid and self.attributes_summary.get(k, False):
                        min_val = self.attributes_summary[k][0]
                        max_val = self.attributes_summary[k][1]
                        attribute_range = self.attributes_summary[k][2]
                        scaled_value = (v - min_val) / attribute_range
                        if self.truncate_large_values:
                            if scaled_value < 0:
                                scaled_value = 0
                            elif scaled_value > 1:
                                scaled_value = 1
                        row[k] = scaled_value

        return X


class AddPredictedFeature(BaseEstimator, TransformerMixin):


    def __init__(self, type_of_estimator=None, model_name='MiniBatchKMeans', include_original_X=False, y_train=None):
        # 'regressor' or 'classifier'
        self.type_of_estimator = type_of_estimator
        # Name of the model to fit.
        self.model_name = model_name
        # WHether to append a single new feature onto the entire existing X feature set and return the entire X dataset plus this new feature, or whether to only return a single feature for the predcicted value
        self.include_original_X = include_original_X
        # If this is for an esembled subpredictor, these are the y values we will train the predictor on while running .fit()
        self.y_train = y_train
        self.n_clusters = 8


    def fit(self, X, y=None):
        self.model = get_model_from_name(self.model_name)

        if self.y_train is not None:
            y = y_train

        if self.model_name == 'MiniBatchKMeans':
            self.model.fit(X)
        else:
            self.model.fit(X, y)

        # For ml_for_analytics, we'll want to save these feature names somewhere easily accessible
        self.added_feature_names_ = ['prdicted_cluster_group_' + str(x) for x in range(self.n_clusters)]

        return self


    def transform(self, X, y=None):
        predictions = self.model.predict(X)
        if self.model_name == 'MiniBatchKMeans':

            # KMeans will return an int cluster prediction. We need to turn that into categorical variables our models can recognize (cluster=1: True, cluster=2: False, etc.)
            encoded_predictions = []
            for prediction in predictions:
                blank_prediction_row = [0 for x in range(self.n_clusters)]
                blank_prediction_row[prediction] = 1
                encoded_predictions.append(blank_prediction_row)

            predictions = encoded_predictions
        else:
            # We need to reshape our predictions to each be very clearly one row
            predictions = [[x] for x in predictions]
        if self.include_original_X:
            X = scipy.sparse.hstack((X, predictions), format='csr')
            return X
        else:
            return predictions


class AddSubpredictorPredictions(BaseEstimator, TransformerMixin):


    def __init__(self, trained_subpredictors, include_original_X=True):
        self.trained_subpredictors = trained_subpredictors
        self.include_original_X = include_original_X
        self.sub_names = [pred.named_steps['final_model'].output_column for pred in self.trained_subpredictors]


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if isinstance(X, dict):
            X = [X]
        predictions = []
        for predictor in self.trained_subpredictors:

            if predictor.named_steps['final_model'].type_of_estimator == 'regressor':
                predictions.append(predictor.predict(X))

            else:
                predictions.append(predictor.predict(X))

        if self.include_original_X:
            X_copy = []
            for row_idx, row in enumerate(X):

                row_copy = {}
                for k, v in row.items():
                    row_copy[k] = v

                for pred_idx, name in enumerate(self.sub_names):

                    row_copy[name + '_sub_prediction'] = predictions[pred_idx][row_idx]

                X_copy.append(row_copy)

            return X_copy

        else:
            return predictions

