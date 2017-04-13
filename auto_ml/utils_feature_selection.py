from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression


import scipy
import itertools
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel


def get_feature_selection_model_from_name(type_of_estimator, model_name):
    model_map = {
        'classifier': {
            'SelectFromModel': SelectFromModel(RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15), threshold='20*mean'),
            'RFECV': RFECV(estimator=RandomForestClassifier(n_jobs=-1), step=0.1),
            'GenericUnivariateSelect': GenericUnivariateSelect(),
            'RandomizedSparse': RandomizedLogisticRegression(),
            'KeepAll': 'KeepAll'
        },
        'regressor': {
            'SelectFromModel': SelectFromModel(RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=15), threshold='0.7*mean'),
            'RFECV': RFECV(estimator=RandomForestRegressor(n_jobs=-1), step=0.1),
            'GenericUnivariateSelect': GenericUnivariateSelect(),
            'RandomizedSparse': RandomizedLasso(),
            'KeepAll': 'KeepAll'
        }
    }

    return model_map[type_of_estimator][model_name]


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):


    def __init__(self, type_of_estimator, column_descriptions, feature_selection_model='SelectFromModel'):

        self.column_descriptions = column_descriptions
        self.type_of_estimator = type_of_estimator
        self.feature_selection_model = feature_selection_model


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default


    def fit(self, X, y=None):


        self.selector = get_feature_selection_model_from_name(self.type_of_estimator, self.feature_selection_model)

        if self.selector == 'KeepAll':
            if scipy.sparse.issparse(X):
                num_cols = X.shape[0]
            else:
                num_cols = len(X[0])

            self.support_mask = [True for col_idx in range(num_cols) ]
        else:
            if self.feature_selection_model == 'SelectFromModel':
                num_cols = X.shape[1]
                num_rows = X.shape[0]
                if self.type_of_estimator == 'regressor':
                    self.estimator = RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=15)
                else:
                    self.estimator = RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15)

                self.estimator.fit(X, y)

                feature_importances = self.estimator.feature_importances_

                # Two ways of doing feature selection

                # 1. Any feature with a feature importance of at least 1/100th of our max feature
                max_feature_importance = max(feature_importances)
                threshold_by_relative_importance = 0.01 * max_feature_importance

                # 2. 1/4 the number of rows (so 100 rows means 25 columns)
                sorted_importances = sorted(feature_importances, reverse=True)
                max_cols = int(num_rows * 0.25)
                try:
                    threshold_by_max_cols = sorted_importances[max_cols]
                except IndexError:
                    threshold_by_max_cols = sorted_importances[-1]

                threshold = max(threshold_by_relative_importance, threshold_by_max_cols)
                self.support_mask = [True if x > threshold else False for x in feature_importances]

            else:
                self.selector.fit(X, y)
                self.support_mask = self.selector.get_support()

        # Get a mask of which indices it is we want to keep
        self.index_mask = [idx for idx, val in enumerate(self.support_mask) if val == True]
        return self


    def transform(self, X, y=None):

        if self.selector == 'KeepAll':
            return X

        if scipy.sparse.issparse(X):
            if X.getformat() == 'csr':
                # convert to a csc (column) matrix, rather than a csr (row) matrix
                X = X.tocsc()

            # Slice that column matrix to only get the relevant columns that we already calculated in fit:
            X = X[:, self.index_mask]

            # convert back to a csr matrix
            return X.tocsr()

        # If this is a dense matrix:
        else:
            pruned_X = [list(itertools.compress(row, self.support_mask)) for row in X]
            return pruned_X

