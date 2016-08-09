from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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


def instantiate_model(model_name='RandomForestClassifier'):
    print(model_name)

    model_map = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    return model_map[model_name]


class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model_name, X_train=None, y_train=None, perform_grid_search_on_model=False):

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.perform_grid_search_on_model = perform_grid_search_on_model

        self.model_map = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier()
        }


    def fit(self, X, y):
        print('self.model_name inside fit')
        print(self.model_name)
        self.model = self.model_map[self.model_name]
        self.model.fit(X, y)


    def score(self, X, y):
        return self.model.score(X, y)


    def predict_proba(self, X):
        return self.model.predict_proba(X)
