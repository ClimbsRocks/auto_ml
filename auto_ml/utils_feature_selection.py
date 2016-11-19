from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression


import scipy
import itertools
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, SelectFromModel


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
            'SelectFromModel': SelectFromModel(RandomForestRegressor(n_jobs=-1, max_depth=10, n_estimators=15), threshold='0.5*mean'),
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

