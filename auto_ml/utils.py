from sklearn.base import BaseEstimator, TransformerMixin


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
