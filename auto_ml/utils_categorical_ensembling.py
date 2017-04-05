import pandas as pd

class CategoricalEnsembler(object):

    def __init__(self, trained_models, transformation_pipeline, categorical_column):
        self.trained_models = trained_models
        self.categorical_column = categorical_column
        self.transformation_pipeline = transformation_pipeline


    def predict(self, data):
        # For now, we are assuming that data is a list of dictionaries, so if we have a single dict, put it in a list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        predictions = []
        for row in data:
            category = row[self.categorical_column]
            model = self.trained_models[category]
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
            model = self.trained_models[category]
            transformed_row = self.transformation_pipeline.transform(row)
            prediction = model.predict_proba(transformed_row)
            predictions.append(prediction)

        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions
