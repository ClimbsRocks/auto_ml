import csv
import datetime
import math
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, RandomizedLasso, RandomizedLogisticRegression, RidgeClassifier, Ridge, Perceptron, RANSACRegressor
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss

import scipy

import xgboost as xgb


def split_output(X, output_column_name, verbose=False):
    y = []
    for row in X:
        y.append(
            row.pop(output_column_name)
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
        pass


    def fit(self, X, y=None):
        return self


    def turn_strings_to_floats(self, X, y=None):

        vals_to_del = set([None, float('nan'), float('Inf')])

        for row in X:
            for key, val in row.items():
                col_desc = self.column_descriptions.get(key)
                if col_desc == 'categorical':
                    row[key] = str(val)
                elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
                    if val in vals_to_del:
                        del row[key]
                    else:
                        row[key] = float(val)
                else:
                    # covers cases for dates, target, etc.
                    pass

        return X


    def transform(self, X, y=None):
        X = self.turn_strings_to_floats(X, y)

        return X




# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# This class provides two key methods for optimization:
# 1. Model selection (try a bunch of different mdoels, see which one is best)
# 2. Model hyperparameter optimization (or not). This class will allow you to use the base estimator, or optimize the estimator's hyperparameters.
class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model_name, X_train=None, y_train=None, perform_grid_search_on_model=False, model_map=None, ml_for_analytics=False, type_of_estimator='classifier'):

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.perform_grid_search_on_model = perform_grid_search_on_model
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator

        if model_map is not None:
            self.model_map = model_map
        else:
            self.set_model_map()


    def set_model_map(self):
        self.model_map = {
            # Classifiers
            'LogisticRegression': LogisticRegression(n_jobs=-2),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-2),
            'RidgeClassifier': RidgeClassifier(),
            'XGBClassifier': xgb.XGBClassifier(),

            # Regressors
            'LinearRegression': LinearRegression(n_jobs=-2),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=-2),
            'Ridge': Ridge(),
            'XGBRegressor': xgb.XGBRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(n_jobs=-1),
            'AdaBoostRegressor': AdaBoostRegressor(n_estimators=5),
            'RANSACRegressor': RANSACRegressor()
        }


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
                'max_depth': [1, 2, 5, 20, 50, 100]
                # 'learning_rate': np.random.uniform(0.0, 1.0)
            },
            'XGBRegressor': {
                'max_depth': [1, 2, 5, 20, 50, 100]

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
            }

        }

        return randomized_search_params[self.model_name]


    def fit(self, X, y):

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
            # TODO(PRESTON): add in RMSE
            if self.type_of_estimator == 'classifier':
                scorer = make_scorer(brier_score_loss, greater_is_better=True)
            else:
                # scorer = None
                # # scorer = 'mean_squared_error'
                scorer = rmse_scoring

            gs_params = self.get_search_params()

            n_iter = 5
            if self.model_name == 'LinearRegression':
                # There's just not much to optimize on a linear regression
                n_iter = 4
            self.rscv = RandomizedSearchCV(
                self.model_map[self.model_name],
                gs_params,
                # Pick n_iter combinations of hyperparameters to fit on and score.
                # Larger numbers risk more overfitting, but also could be more accurate, at more computational expense.
                n_iter=n_iter,
                n_jobs=-1,
                # verbose=1,
                # Print warnings, but do not raise errors if a combination of hyperparameters fails to fit.
                error_score=10,
                # TOOD(PRESTON): change to be RMSE by default
                scoring=scorer
            )
            self.rscv.fit(X, y)
            self.model = self.rscv.best_estimator_

        # or, we can just use the default estimator
        else:
            self.model = self.model_map[self.model_name]

            self.model.fit(X, y)

        return self


    def score(self, X, y):
        return self.model.score(X, y)


    def predict_proba(self, X):
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
        except Exception as e:
            raise(e)

    def predict(self, X):
        return self.model.predict(X)


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


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, type_of_estimator, feature_selection_model='SelectFromModel'):


        self.type_of_estimator = type_of_estimator
        self.feature_selection_model = feature_selection_model
        self._set_model_map()


    def _set_model_map(self):
        # TODO(PRESTON): eventually let threshold be user-configurable (or grid_searchable)
        # TODO(PRESTON): optimize the params used here
        self._model_map = {
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

    def fit(self, X, y=None):

        self.selector = self._model_map[self.type_of_estimator][self.feature_selection_model]

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
            self.selector.transform(X)
            return X


def rmse_scoring(estimator, X, y, took_log_of_y=False):
    predictions = estimator.predict(X)
    if took_log_of_y:
        for idx, val in enumerate(predictions):
            predictions[idx] = math.exp(val)
    rmse = mean_squared_error(y, predictions)**0.5
    return rmse


def get_all_attribute_names(list_of_dictionaries, cols_to_avoid):
    attribute_hash = {}
    for dictionary in list_of_dictionaries:
        for k, v in dictionary:
            attribute_hash[k] = True

    # All of the columns in column_descriptions should be avoided. They're probably either categorical or NLP data, both of which cannot be scaled.

    attribute_list = [k for k, v in attribute_hash.items() if k not in cols_to_avoid]
    return attribute_list


# Scale sparse data to the 90th and 10th percentile
# Only do so for values that actuall exist (do absolutely nothing with rows that do not have this data point)
def CustomSparseScaler(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions):
        self.column_descriptions = column_descriptions
        self.cols_to_avoid = set([k for k, v in column_descriptions.items()])
        pass


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
                    attributes_summary[k].append(v)

            for attribute in bucket:

                # Sort our collected data for that column
                attributes_summary[attribute] = sorted(attributes_summary[attribute])
                col_vals = attributes_summary[attribute]
                tenth_percentile = col_vals[int(0.1 * len(col_vals))]
                ninetieth_percentile = col_vals[int(0.9 * len(col_vals))]
                attributes_summary[attribute] = [tenth_percentile, ninetieth_percentile, ninetieth_percentile - tenth_percentile]
                del col_vals

        self.attributes_summary = attributes_summary


    # Perform basic min/max scaling, with the minor caveat that our min and max values are the 10th and 90th percentile values, to avoid outliers.
    def transform(self, X, y=None):
        for row in X:
            for k, v in row.items():
                if k not in self.cols_to_avoid:
                    min_val = self.attributes_summary[k][0]
                    max_val = self.attributes_summary[k][1]
                    attribute_range = self.attributes_summary[k][2]
                    row[k] = (v - min_val) / attribute_range
        print(X[10])
        print(X[100])

        return X


