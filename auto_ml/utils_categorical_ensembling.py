import pandas as pd

class CategoricalEnsembler(object):

    def __init__(self, trained_models, transformation_pipeline, categorical_column, default_category):
        self.trained_models = trained_models
        self.categorical_column = categorical_column
        self.transformation_pipeline = transformation_pipeline
        self.default_category = default_category
        self.is_categorical_ensembler = True


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default


    def predict(self, data):
        # For now, we are assuming that data is a list of dictionaries, so if we have a single dict, put it in a list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        predictions = []
        for row in data:
            category = row[self.categorical_column]
            if str(category) == 'nan':
                category = 'nan'
            try:
                model = self.trained_models[category]
            except KeyError as e:
                if self.default_category == '_RAISE_ERROR':
                    raise(e)
                model = self.trained_models[self.default_category]

            transformed_row = self.transformation_pipeline.transform(row)
            prediction = model.predict(transformed_row)
            predictions.append(prediction)

        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions

    def predict_proba(self, data):
        # For now, we are assuming that data is a list of dictionaries, so if we have a single dict, put it in a list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        predictions = []
        for row in data:
            category = row[self.categorical_column]
            if str(category) == 'nan':
                category = 'nan'

            try:
                model = self.trained_models[category]
            except KeyError as e:
                if self.default_category == '_RAISE_ERROR':
                    raise(e)
                model = self.trained_models[self.default_category]

            transformed_row = self.transformation_pipeline.transform(row)
            prediction = model.predict_proba(transformed_row)
            predictions.append(prediction)

        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions


# Remove nans from our categorical ensemble column
def clean_categorical_definitions(df, categorical_column):
    sum_of_nan_values = df[categorical_column].isnull().sum().sum()
    if sum_of_nan_values > 0:
        print('Found ' + str(sum_of_nan_values) + ' nan values in the categorical_column.')
        print('We will default to making these values a string "nan" instead, since that can be used as a key')
        print('If this is not the behavior you want, consider changing these categorical_column values yourself')

        df[categorical_column].fillna('nan', inplace=True)

    return df
