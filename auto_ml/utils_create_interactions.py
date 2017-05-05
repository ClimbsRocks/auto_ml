import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreateInteractions(BaseEstimator, TransformerMixin):

    def __init__(self, column_descriptions):
        self.column_descriptions = column_descriptions
        pass

    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default

    def fit(self, X, y=None):
        # TODO:
        # 1. create a list of columns we want interactions for (ignoring categorical columns, nlp, and date for now)
        # TODO POST-MVP:
            # interact categorical columns with each other. might be crazy cardinality
        col_types_to_ignore = ['ignore', 'date', 'categorical', 'output', 'nlp']
        columns_to_interact = []
        for col in X.columns:
            col_type = self.column_descriptions.get(col, 'ok')
            if col_type not in col_types_to_ignore and 'is_weekend' not in col:
                columns_to_interact.append(col)

        self.columns_to_interact = sorted(columns_to_interact)
        return self

    def transform(self, X, y=None):

        for idx, col in enumerate(self.columns_to_interact):
            # print('col')
            # print(col)
            X[col + '_squared'] = X[col] * X[col]
            for interaction_col in self.columns_to_interact[idx + 1:]:
                # print('interaction_col')
                # print(interaction_col)
                # if isinstance(X, pd.DataFrame):
                X[col + '-' + interaction_col] = X[col] - X[interaction_col]
                X[col + '+' + interaction_col] = X[col] + X[interaction_col]
                X[col + '*' + interaction_col] = X[col] * X[interaction_col]
                X[col + '/' + interaction_col] = X[col] / X[interaction_col]



        return X



