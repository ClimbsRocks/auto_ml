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

import scipy

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


class BasicDataCleaning(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions=None):
        self.column_descriptions = column_descriptions
        self.vals_to_del = set([None, float('nan'), float('Inf')])
        self.vals_to_ignore = set(['regressor', 'classifier', 'output', 'ignore'])


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        clean_X = []
        deleted_values_sample = []
        deleted_info = {}

        for row in X:
            clean_row = {}

            for key, val in row.items():
                col_desc = self.column_descriptions.get(key)
                if col_desc == 'categorical':
                    clean_row[key] = str(val)
                elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
                    if val not in self.vals_to_del:
                        clean_row[key] = float(val)
                elif col_desc == 'date':
                    clean_row = add_date_features(val, clean_row, key)
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

def add_date_features(date_val, target_row, date_col):

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
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
    # model_map = {
    #     # Classifiers
    #     'LogisticRegression': LogisticRegression,
    #     'RandomForestClassifier': RandomForestClassifier,
    #     'RidgeClassifier': RidgeClassifier,
    #     'XGBClassifier': xgb.XGBClassifier,
    #     'GradientBoostingClassifier': GradientBoostingClassifier,

    #     # Regressors
    #     'LinearRegression': LinearRegression,
    #     'RandomForestRegressor': RandomForestRegressor,
    #     'Ridge': Ridge,
    #     'XGBRegressor': xgb.XGBRegressor,
    #     'ExtraTreesRegressor': ExtraTreesRegressor,
    #     'AdaBoostRegressor': AdaBoostRegressor,
    #     'RANSACRegressor': RANSACRegressor,
    #     'GradientBoostingRegressor': GradientBoostingRegressor
    # }

    # new_instance = model_map[model_name]
    # new_instance = new_instance()
    # # Super crude, but we'll just try to set all the available params on this new instance
    # try:
    #     new_instance.set_params(n_jobs=-1)
    # except:
    #     pass
    # try:
    #     new_instance.set_params(presort=False)
    # except:
    #     pass
    # # TODO: eventually, don't create new instances except for what the user requests
    # # Then have a params_to_set hash, where we've got all the params we're interested in setting

    # return new_instance




# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# This class provides two key methods for optimization:
# 1. Model selection (try a bunch of different mdoels, see which one is best)
# 2. Model hyperparameter optimization (or not). This class will allow you to use the base estimator, or optimize the estimator's hyperparameters.
class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model_name, X_train=None, y_train=None, perform_grid_search_on_model=False, ml_for_analytics=False, type_of_estimator='classifier'):

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.perform_grid_search_on_model = perform_grid_search_on_model
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator

        if self.type_of_estimator == 'classifier':
            self._scorer = rscv_brier_score_loss_wrapper
        else:
            self._scorer = rscv_rmse_scoring


    # It would be optimal to store large objects like this elsewhere, but storing it all inside FinalModelATC ensures that each instance will always be self-contained, which is helpful when saving and transferring to different environments.
    def get_search_params(self):
        randomized_search_params = {
            'LogisticRegression': {
                'C': scipy.stats.expon(.0001, 1000),
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
                'alpha': scipy.stats.expon(.0001, 1000),
                'class_weight': [None, 'balanced'],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
            },
            'Ridge': {
                'alpha': scipy.stats.expon(.0001, 1000),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
            },
            'XGBClassifier': {
                'max_depth': [1, 2, 5, 20, 50, 100],
                'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
                'subsample': [0.5, 1]
            },
            'XGBRegressor': {
                # Add in max_delta_step if classes are extremely imbalanced
                'max_depth': [1, 2, 5, 20, 50, 100],
                # 'lossl': ['ls', 'lad', 'huber', 'quantile']
                # 'booster': ['gbtree', 'gblinear', 'dart'],
                'objective': ['reg:linear', 'reg:gamma', 'rank:pairwise'],
                'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
                'subsample': [0.5, 1],

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
            'GradientBoostingRegressor': {
                # Add in max_delta_step if classes are extremely imbalanced
                'max_depth': [1, 2, 5, 20, 50, 100],
                # 'loss': ['ls', 'lad', 'huber', 'quantile']
                # 'booster': ['gbtree', 'gblinear', 'dart'],
                'loss': ['ls', 'lad', 'huber', 'quantile'],
                'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
                'subsample': [0.5, 1]
            },
            'GradientBoostingClassifier': {
                'loss': ['deviance', 'exponential'],
                'max_depth': [1, 2, 5, 20, 50, 100],
                'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
                'subsample': [0.5, 1]
            },
            'Lasso': {
                'selection': ['cyclic', 'random'],
                'tol': scipy.stats.expon(.0000001, .001),
                'positive': [True, False]
            },

            'ElasticNet': {
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'selection': ['cyclic', 'random'],
                'tol': scipy.stats.expon(.0000001, .001),
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
                'tol': scipy.stats.expon(.0000001, .001),
                'alpha_1': scipy.stats.expon(.000000001, .0001),
                'lambda_1': scipy.stats.expon(.000000001, .0001),
                'lambda_2': scipy.stats.expon(.000000001, .0001)
            },

            'ARDRegression': {
                'tol': scipy.stats.expon(.0000001, .001),
                'alpha_1': scipy.stats.expon(.000000001, .0001),
                'alpha_2': scipy.stats.expon(.000000001, .0001),
                'lambda_1': scipy.stats.expon(.000000001, .0001),
                'lambda_2': scipy.stats.expon(.000000001, .0001),
                'threshold_lambda': scipy.stats.expon(100, 1000000)
            },

            'SGDRegressor': {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'alpha': scipy.stats.expon(.000000001, .0001),
            },

            'PassiveAggressiveRegressor': {
                'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
                'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                'C': scipy.stats.expon(0.000001, 100000)
            },

            'SGDClassifier': {
                'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'alpha': scipy.stats.expon(.000000001, .0001),
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'class_weight': ['balanced', None]
            },

            'Perceptron': {
                'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'alpha': scipy.stats.expon(.000000001, .0001),
                'class_weight': ['balanced', None]
            },

            'PassiveAggressiveClassifier': {
                'loss': ['hinge', 'squared_hinge'],
                'class_weight': ['balanced', None],
                'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            }

        }

        return randomized_search_params[self.model_name]


    def fit(self, X, y):

        model_to_fit = get_model_from_name(self.model_name)

        if self.ml_for_analytics:
            self.feature_ranges = []

            # Grab the ranges for each feature
            if scipy.sparse.issparse(X):
                for col_idx in range(X.shape[1]):
                    col_vals = X.getcol(col_idx).toarray()
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


        # we can perform RandomizedSearchCV on just our final estimator.
        if self.perform_grid_search_on_model:

            gs_params = self.get_search_params()

            n_iter = 8
            # if self.model_name == 'XGBRegressor':
            #     n_iter = 20
            if self.model_name == 'LinearRegression':
                # There's just not much to optimize on a linear regression
                n_iter = 4

            # # print the settable parameter names
            # print(self.model_map[self.model_name].get_params().keys())
            self.rscv = RandomizedSearchCV(
                model_to_fit,
                gs_params,
                # Pick n_iter combinations of hyperparameters to fit on and score.
                # Larger numbers risk more overfitting, but also could be more accurate, at more computational expense.
                n_iter=n_iter,
                n_jobs=-1,
                # Have only two folds of cross-validation, rather than 3. This speeds up training time, and reduces the risk of overfitting.
                cv=2,
                # verbose=1,
                # If a combination of hyperparameters fails to fit, set it's score to a very low number that we will not choose.
                error_score=-1000000000,
                scoring=self._scorer
            )
            self.rscv.fit(X, y)
            self.model = self.rscv.best_estimator_

        # or, we can just use the default estimator
        else:
            # self.model = self.model_map[self.model_name]
            self.model = get_model_from_name(self.model_name)

            self.model.fit(X, y)

        return self


    def score(self, X, y):
        if self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X = X.todense()

        try:
            if self._scorer is not None:
                if self.type_of_estimator == 'regressor':
                    return self._scorer(self.model, X, y)
                elif self.type_of_estimator == 'classifier':
                    return self._scorer(self.model, X, y)

            else:
                return self.model.score(X, y)

        except ValueError:

            # XGBoost doesn't always handle sparse matrices well.
            X_dense = X.todense()

            if self._scorer is not None:
                return self._scorer(X_dense, y)
            else:
                return self.model.score(X_dense, y)


    def predict_proba(self, X):
        if self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X = X.todense()

        try:
            return self.model.predict_proba(X)
        except ValueError:
            # XGBoost doesn't always handle sparse matrices well.
            X_dense = X.todense()
            return self.model.predict_proba(X_dense)
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

        except Exception as e:
            raise(e)

    def predict(self, X):
        if self.model_name[:16] == 'GradientBoosting' and scipy.sparse.issparse(X):
            X = X.todense()

        # XGBoost doesn't always handle sparse matrices well.
        try:
            return self.model.predict(X)
        except ValueError:
            X_dense = X.todense()
            return self.model.predict(X_dense)


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

        # self.selector = self._model_map[self.type_of_estimator][self.feature_selection_model]
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


def brier_score_loss_wrapper(estimator, X, y):
    if isinstance(estimator, GradientBoostingClassifier):
        X = X.toarray()

    predictions = estimator.predict_proba(X)
    probas = [row[1] for row in predictions]
    score = brier_score_loss(y, probas)
    return -1 * score


def rscv_rmse_scoring(estimator, X, y, took_log_of_y=False):
    if isinstance(estimator, GradientBoostingRegressor):
        X = X.toarray()

    # XGBoost does not always handle sparse matrices well. This should work around that annoyance.
    try:
        predictions = estimator.predict(X)
    except ValueError:
        X_dense = X.todense()
        predictions = estimator.predict(X_dense)

    if took_log_of_y:
        for idx, val in enumerate(predictions):
            predictions[idx] = math.exp(val)
    rmse = mean_squared_error(y, predictions)**0.5
    return - 1 * rmse


def rscv_brier_score_loss_wrapper(estimator, X, y):
    if isinstance(estimator, GradientBoostingClassifier):
        X = X.toarray()

    # XGBoost does not always handle sparse matrices well. This should work around that annoyance.
    try:
        predictions = estimator.predict_proba(X)
    except ValueError:
        X_dense = X.todense()
        predictions = estimator.predict_proba(X_dense)

    # predictions = estimator.predict_proba(X)
    probas = [row[1] for row in predictions]
    score = brier_score_loss(y, probas)
    return -1 * score


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


    def __init__(self, column_descriptions, truncate_large_values=False):
        self.column_descriptions = column_descriptions
        self.cols_to_avoid = set([k for k, v in column_descriptions.items()])
        self.truncate_large_values = truncate_large_values


    def fit(self, X, y=None):
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

    # def set_model_map(self):
    #     self.model_map = {
    #         # Classifiers
    #         'LogisticRegression': LogisticRegression(n_jobs=-2),
    #         'RandomForestClassifier': RandomForestClassifier(n_jobs=-2),
    #         'RidgeClassifier': RidgeClassifier(),
    #         'XGBClassifier': xgb.XGBClassifier(),

    #         # Regressors
    #         'LinearRegression': LinearRegression(n_jobs=-2),
    #         'RandomForestRegressor': RandomForestRegressor(n_jobs=-2),
    #         'Ridge': Ridge(),
    #         'XGBRegressor': xgb.XGBRegressor(),
    #         'ExtraTreesRegressor': ExtraTreesRegressor(n_jobs=-1),
    #         'AdaBoostRegressor': AdaBoostRegressor(n_estimators=5),
    #         'RANSACRegressor': RANSACRegressor(),

    #         # Clustering
    #         'MiniBatchKMeans': MiniBatchKMeans(self.n_clusters)
    #     }


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
        # self.set_model_map()


    def fit(self, X, y=None):
        # self.model = self.model_map[self.model_name]
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
        self.sub_names = [pred.output_column for pred in self.trained_subpredictors]


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        predictions = []
        for predictor in self.trained_subpredictors:
            if predictor.type_of_estimator == 'regressor':
                predictions.append(predictor.predict(X))
            else:
                # TODO: Future- if it's a classifier, get both the predicted class, as well as the predict_proba
                predictions.append(predictor.predict(X))
        if self.include_original_X:
            X_copy = []
            for row_idx, row in enumerate(X):
                row_copy = row.copy()
                for pred_idx, name in enumerate(self.sub_names):
                    row_copy[name + '_sub_prediction'] = predictions[pred_idx][row_idx]
                X_copy.append(row_copy)

            return X_copy

        else:
            # TODO: this will break if we ever try to refactor into FeatureUnions again.
            return predictions

