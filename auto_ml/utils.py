from sklearn.base import BaseEstimator, TransformerMixin
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


def instantiate_model(model_name='LogisticRegression'):
    return LogisticRegression()

# class InstantiateModel(BaseEstimator, TransformerMixin):

#     def __init__
