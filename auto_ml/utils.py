from collections import OrderedDict
import csv
import datetime
import math
import os
import random
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

import pandas as pd

import xgboost as xgb


# The easiest way to check against a bunch of different bad values is to convert whatever val we have into a string, then check it against a set containing the string representation of a bunch of bad values
bad_vals_as_strings = set([str(float('nan')), str(float('inf')), str(float('-inf')), 'None', 'none', 'NaN', 'nan', 'NULL', 'null', '', 'inf', '-inf'])

# clean_val will try to turn a value into a float.
# If it fails, it will attempt to strip commas and then attempt to turn it into a float again
# Additionally, it will check to make sure the value is not in a set of bad vals (nan, None, inf, etc.)
# This function will either return a clean value, or raise an error if we cannot turn the value into a float or the value is a bad val
def clean_val(val):
    if str(val) in bad_vals_as_strings:
        raise(ValueError('clean_val failed'))
    else:
        try:
            float_val = float(val)
        except:
            # This will throw a ValueError if it fails
            # remove any commas in the string, and try to turn into a float again
            cleaned_string = val.replace(',', '')
            float_val = float(cleaned_string)
        return float_val

# Same as above, except this version returns float('nan') when it fails
# This plays more nicely with df.apply, and assumes we will be handling nans appropriately when doing DataFrameVectorizer later.
def clean_val_nan_version(val):
    if str(val) in bad_vals_as_strings:
        return float('nan')
    else:
        try:
            float_val = float(val)
        except:
            # This will throw a ValueError if it fails
            # remove any commas in the string, and try to turn into a float again
            cleaned_string = val.replace(',', '')
            try:
                float_val = float(cleaned_string)
            except:
                return float('nan')
        return float_val


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
        self.text_col_indicators = set(['text', 'nlp'])
        self.tfidfvec = TfidfVectorizer(
            # If we have any documents that cannot be decoded properly, just ignore them and keep going as planned with everything else
            decode_error='ignore'
            # Try to strip accents from characters. Using unicode is slightly slower but more comprehensive than 'ascii'
            , strip_accents='unicode'
            # Can also choose 'character', which will likely increase accuracy, at the cost of much more space, generally
            , analyzer='word'
            # Remove commonly found english words ('it', 'a', 'the') which do not typically contain much signal
            , stop_words='english'
            # Convert all characters to lowercase
            , lowercase=True
            # Only consider words that appear in fewer than max_df percent of all documents
            # In this case, ignore all words that appear in 90% of all documents
            , max_df=0.9
            # Consider only the most frequently occurring 3000 words, after taking into account all the other filtering going on
            , max_features=3000
        )

    def fit(self, X_df, y=None):

        # See if we should fit TfidfVectorizer or not
        for key in X_df.columns:
            col_desc = self.column_descriptions.get(key, False)
            if col_desc in self.text_col_indicators:
                    self.tfidfvec.fit(X_df[key])

        return self

    def transform(self, X, y=None):
        # Convert input to DataFrame if we were given a list of dictionaries
        if isinstance(X, dict) or isinstance(X, list):
            X = pd.DataFrame(X)

        # All of these are values we will not want to keep for training this particular estimator or subpredictor
        vals_to_drop = set(['ignore', 'output', 'regressor', 'classifier'])

        # It is much more efficient to drop a bunch of columns at once, rather than one at a time
        cols_to_drop = []


        for key in X.columns:
            col_desc = self.column_descriptions.get(key)
            if col_desc == 'categorical':
                # We will handle categorical data later, one-hot-encoding it inside DataFrameVectorizer
                pass

            elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
                # For all of our numerical columns, try to turn all of these values into floats
                # This function handles commas inside strings that represent numbers, and returns nan if we cannot turn this value into a float. nans are ignored in DataFrameVectorizer
                X[key] = X[key].apply(clean_val_nan_version)

            elif col_desc == 'date':
                X = add_date_features_df(X, key)

            elif col_desc in self.text_col_indicators:

                keys = self.tfidfvec.get_feature_names()

                tfvec = self.tfidfvec.transform(X.loc[:,key].values).toarray()
                #create sepearte dataframe and append next to each other along columns
                textframe = pd.DataFrame(tfvec)
                X = X.join(textframe)
                #once the transformed datafrane is added , remove original text
                X = X.drop(key, axis=1) 

            elif col_desc in vals_to_drop:
                cols_to_drop.append(key)

            else:
                # If we have gotten here, the value is not any that we recognize
                # This is most likely a typo that the user would want to be informed of, or a case while we're developing on auto_ml itself.
                # In either case, it's useful to log it.
                print('When transforming the data, we have encountered a value in column_descriptions that is not currently supported. The column has been dropped to allow the rest of the pipeline to run. Here\'s the name of the column:' )
                print(key)
                print('And here is the value for this column passed into column_descriptions:')
                print(col_desc)

        if len(cols_to_drop) > 0:
            X = X.drop(cols_to_drop, axis=1)

        return X


def minutes_into_day_parts(minutes_into_day):
    if minutes_into_day < 6 * 60:
        return 'late_night'
    elif minutes_into_day < 10 * 60:
        return 'morning'
    elif minutes_into_day < 11.5 * 60:
        return 'mid_morning'
    elif minutes_into_day < 14 * 60:
        return 'lunchtime'
    elif minutes_into_day < 18 * 60:
        return 'afternoon'
    elif minutes_into_day < 20.5 * 60:
        return 'dinnertime'
    elif minutes_into_day < 23.5 * 60:
        return 'early_night'
    else:
        return 'late_night'


def add_date_features_df(df, date_col):

    df[date_col + '_day_of_week'] = df[date_col].apply(lambda x: x.weekday()).astype(int)
    df[date_col + '_hour'] = df[date_col].apply(lambda x: x.hour).astype(int)

    df[date_col + '_minutes_into_day'] = df[date_col].apply(lambda x: x.hour * 60 + x.minute)

    df[date_col + '_is_weekend'] = df[date_col].apply(lambda x: x.weekday() in (5,6))
    df[date_col + '_day_part'] = df[date_col + '_minutes_into_day'].apply(minutes_into_day_parts)

    df = df.drop([date_col], axis=1)

    return df


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


# Used in CustomSparseScaler
def calculate_scaling_ranges(X, col, min_percentile=0.05, max_percentile=0.95):

    series_vals = X[col]
    good_vals_indexes = series_vals.notnull()

    series_vals = list(series_vals[good_vals_indexes])
    series_vals = sorted(series_vals)

    max_val_idx = int(max_percentile * len(series_vals)) - 1
    min_val_idx = int(min_percentile * len(series_vals))

    if len(series_vals) > 0:
        max_val = series_vals[max_val_idx]
        min_val = series_vals[min_val_idx]
    else:
        print('This column appears to have only nan values, and will be ignored:')
        print(col)
        return 'ignore'

    inner_range = max_val - min_val

    if inner_range == 0:
        # Used to do recursion here, which is prettier and uses less code, but since we've already got the filtered and sorted series_vals, it makes sense to use those to avoid duplicate computation
        # Grab the absolute largest max and min vals, and see if there is any difference in them, since our 95th and 5th percentile vals had no difference between them
        max_val = series_vals[len(series_vals) - 1]
        min_val = series_vals[0]
        inner_range = max_val - min_val

        if inner_range == 0:
            print('This column appears to have 0 variance (the max and min values are the same), and will be ignored:')
            print(col)
            return 'ignore'

    col_summary = {
        'max_val': max_val
        , 'min_val': min_val
        , 'inner_range': inner_range
    }

    return col_summary

# Scale sparse data to the 95th and 5th percentile
# Only do so for values that actuall exist (do absolutely nothing with rows that do not have this data point)
class CustomSparseScaler(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions, truncate_large_values=False, perform_feature_scaling=True):
        self.column_descriptions = column_descriptions

        self.numeric_col_descs = set([None, 'continuous', 'numerical', 'numeric', 'float', 'int'])
        # Everything in column_descriptions (except numeric_col_descs) is a non-numeric column, and thus, cannot be scaled
        self.cols_to_avoid = set([k for k, v in column_descriptions.items() if v not in self.numeric_col_descs])

        # Setting these here so that they can be grid searchable
        # Truncating large values is an interesting strategy. It forces all values to fit inside the 5th - 95th percentiles. 
        # Essentially, it turns any really large (or small) values into reasonably large (or small) values. 
        self.truncate_large_values = truncate_large_values
        self.perform_feature_scaling = perform_feature_scaling


    def fit(self, X, y=None):
        self.column_ranges = {}
        self.cols_to_ignore = []

        if self.perform_feature_scaling:

            for col in X.columns:
                if col not in self.cols_to_avoid:
                    col_summary = calculate_scaling_ranges(X, col, min_percentile=0.05, max_percentile=0.95)
                    if col_summary == 'ignore':
                        self.cols_to_ignore.append(col)
                    else:
                        self.column_ranges[col] = col_summary

        return self


    # Perform basic min/max scaling, with the minor caveat that our min and max values are the 10th and 90th percentile values, to avoid outliers.
    def transform(self, X, y=None):
        if len(self.cols_to_ignore) > 0:
            X = X.drop(self.cols_to_ignore, axis=1)

        for col, col_dict in self.column_ranges.items():
            min_val = col_dict['min_val']
            inner_range = col_dict['inner_range']
            X[col] = X[col].apply(lambda x: scale_val(x, min_val, inner_range, self.perform_feature_scaling))

        return X


def scale_val(val, min_val, total_range, truncate_large_values=False):
    scaled_value = (val - min_val) / total_range
    if truncate_large_values:
        if scaled_value < 0:
            scaled_value = 0
        elif scaled_value > 1:
            scaled_value = 1

    return scaled_value


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
        if isinstance(X, dict) or isinstance(X, list):
            X = pd.DataFrame(X)
        predictions = []
        for predictor in self.trained_subpredictors:

            if predictor.named_steps['final_model'].type_of_estimator == 'regressor':
                predictions.append(predictor.predict(X))

            else:
                predictions.append(predictor.predict(X))

        if self.include_original_X:
            for pred_idx, name in enumerate(self.sub_names):
                X[name + '_sub_prediction'] = predictions[pred_idx]
            return X
            
        else:
            return predictions

def safely_drop_columns(df, cols_to_drop):
    safe_cols_to_drop = []
    for col in cols_to_drop:
        if col in df.columns:
            safe_cols_to_drop.append(col)

    df = df.drop(safe_cols_to_drop, axis=1)
    return df
