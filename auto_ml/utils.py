import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, Ridge
from sklearn.metrics import mean_squared_error, make_scorer, brier_score_loss

import scipy


def split_output(X, output_column_name):
    y = []
    for row in X:
        y.append(
            row.pop(output_column_name)
        )

    return X, y


class BasicDataCleaning(BaseEstimator, TransformerMixin):


    def __init__(self, column_descriptions=None):
        self.column_descriptions = column_descriptions
        pass


    def fit(self, X, y=None):
        return self


    def turn_strings_to_floats(self, X, y=None):

        for row in X:
            for key, val in row.items():
                col_desc = self.column_descriptions.get(key)
                if col_desc == 'categorical':
                    row[key] = str(val)
                elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
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


    def __init__(self, model_name, X_train=None, y_train=None, perform_grid_search_on_model=False, model_map=None, ml_for_analytics=False):

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.perform_grid_search_on_model = perform_grid_search_on_model
        self.ml_for_analytics = ml_for_analytics

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
            'XGBRegressor': xgb.XGBRegressor()
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

            },
            'RidgeClassifier': {
                'alpha': scipy.stats.expon(.0001, 1000),
                'class_weight': [None, 'balanced'],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
            },
            'Ridge': {

            },
            'XGBClassifier': {
                'max_depth': [1, 2, 5, 20, 50, 100]
                # 'learning_rate': np.random.uniform(0.0, 1.0)
            },
            'XGBRegressor': {

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
            scorer = make_scorer(brier_score_loss, greater_is_better=True)

            gs_params = self.get_search_params()
            self.rscv = RandomizedSearchCV(
                self.model_map[self.model_name],
                gs_params,
                # Pick n_iter combinations of hyperparameters to fit on and score.
                # Larger numbers risk more overfitting, but also could be more accurate, at more computational expense.
                n_iter=5,
                n_jobs=-1,
                verbose=1,
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
