from sklearn.base import BaseEstimator, TransformerMixin


class SplitOutput(BaseEstimator, TransformerMixin):

    def __init__(self, output_column_name):
        self.output_column_name = output_column_name


    def transform(self, X, y=[]):

        for row in X:
            y.append( 
                row.pop(self.output_column_name)
            )

        return X, y


    def fit(self, X, y=None):

        return self
