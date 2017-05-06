import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreateInteractions(BaseEstimator, TransformerMixin):

    def __init__(self, column_descriptions, interaction_cols_to_ignore=None, interaction_cols_to_keep=None):
        self.column_descriptions = column_descriptions
        if interaction_cols_to_ignore is None:
            interaction_cols_to_ignore = []
        self.interaction_cols_to_ignore = set(interaction_cols_to_ignore)
        if interaction_cols_to_keep is None:
            interaction_cols_to_keep = []
        self.interaction_cols_to_keep = set(interaction_cols_to_keep)
        self.is_first_transform = True

        pass

    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default

    def fit(self, X, y=None):
        print('Fitting our create_interactions module')
        # TODO:
        # 1. create a list of columns we want interactions for (ignoring categorical columns, nlp, and date for now)
        # TODO POST-MVP:
            # interact categorical columns with each other. might be crazy cardinality
        col_types_to_ignore = ['ignore', 'date', 'categorical', 'output', 'nlp']
        columns_to_interact = []

        if len(self.interaction_cols_to_keep) > 1:
            columns_to_check = list(self.interaction_cols_to_keep)
        else:
            columns_to_check = X.columns

        for col in columns_to_check:
            col_type = self.column_descriptions.get(col, 'ok')
            if col_type not in col_types_to_ignore and 'is_weekend' not in col and col not in self.interaction_cols_to_ignore:
                columns_to_interact.append(col)

        self.columns_to_interact = sorted(columns_to_interact)
        return self

    def transform(self, X, y=None):

        def transform_one_col(idx, col, is_first_transform):
            if is_first_transform:
                print('Creating interactions for ' + col)
            results = {}
        # for idx, col in enumerate(self.columns_to_interact):
            col_data = X[col]
            results[col + '_squared'] = col_data * col_data

            for interaction_col in self.columns_to_interact[idx + 1:]:
                interaction_col_data = X[interaction_col]
                # print('interaction_col')
                # print(interaction_col)
                # if isinstance(X, pd.DataFrame):
                results[col + '-' + interaction_col] = col_data - interaction_col_data
                results[col + '+' + interaction_col] = col_data + interaction_col_data
                results[col + '*' + interaction_col] = col_data * interaction_col_data
                try:
                    results[col + '/' + interaction_col] = col_data / interaction_col_data
                except ZeroDivisionError:
                    results[col + '/' + interaction_col] = None

            return results

        results = map(lambda x: transform_one_col(x[0], x[1], self.is_first_transform), [[idx, col] for idx, col in enumerate(self.columns_to_interact)])

        col_dict = {}
        for result in results:
            col_dict.update(result)

        if isinstance(X, pd.DataFrame):
            df_new_cols = pd.DataFrame.from_dict(col_dict, orient='columns')
            # Hstack this onto our existing X df
            X = pd.concat([X, df_new_cols], axis=1)

            if self.is_first_transform:
                print('Total number of columns after creating_interactions:')
                print(len(list(X.columns)))
        else:
            X.update(col_dict)

        self.is_first_transform = False

        return X



