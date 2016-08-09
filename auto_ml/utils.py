from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


# originally implemented to be consistent with sklearn's API, but currently used outside of a pipeline
class SplitOutput(BaseEstimator, TransformerMixin):

    def __init__(self, output_column_name):
        self.output_column_name = output_column_name


    def transform(self, X, y=None):
        y = []
        for row in X:
            y.append(
                row.pop(self.output_column_name)
            )

        return X, y


    def fit(self, X, y=None):

        return self


# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# This class provides two key methods for optimization:
# 1. Model selection (try a bunch of different mdoels, see which one is best)
# 2. Model hyperparameter optimization (or not). This class will allow you to use the base estimator, or optimize the estimator's hyperparameters.
class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model_name, X_train=None, y_train=None, perform_grid_search_on_model=False, model_map=None):

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.perform_grid_search_on_model = perform_grid_search_on_model

        if model_map is not None:
            self.model_map = model_map
        else:
            self.set_model_map()


    def set_model_map(self):
        self.model_map = {
            'LogisticRegression': LogisticRegression(n_jobs=-2),
            'RandomForestClassifier': RandomForestClassifier(n_jobs=-2)
        }


    # It would be optimal to store large objects like this elsewhere, but storing it all inside FinalModelATC ensures that each instance will always be self-contained, which is helpful when saving and transferring to different environments.
    def get_grid_search_params(self):
        gs_params = {
            'LogisticRegression': {
                'C': [.001, .01, .1, 1],
                'class_weight': [None, 'balanced'],
                'solver': ['newton-cg', 'lbfgs', 'sag']
            },
            'RandomForestClassifier': {
                'criterion': ['entropy', 'gini'],
                'class_weight': [None, 'balanced']
            }

        }

        return gs_params[self.model_name]


    def fit(self, X, y):

        # we can perform GridSearchCV (or, eventually, RandomizedSearchCV) on just our final estimator.
        if self.perform_grid_search_on_model:

            gs_params = self.get_grid_search_params()
            self.model = GridSearchCV(self.model_map[self.model_name], gs_params, n_jobs=-1, verbose=1)

        # or, we can just use the default estimator
        else:
            self.model = self.model_map[self.model_name]

        self.model.fit(X, y)


    def score(self, X, y):
        return self.model.score(X, y)


    def predict_proba(self, X):
        return self.model.predict_proba(X)
