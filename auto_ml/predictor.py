from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import utils

class Predictor(object):


    def __init__(self, type_of_algo, column_descriptions, verbose=True):
        self.type_of_algo = type_of_algo
        self.column_descriptions = column_descriptions
        self.verbose = verbose

        output_column = [key for key, value in column_descriptions.items() if value.lower() == 'output'][0]
        self.output_column = output_column


    def train(self, raw_training_data):

        # split out out output column so we have a proper X, y dataset
        output_splitter = utils.SplitOutput(self.output_column)
        X, y = output_splitter.transform(raw_training_data)

        ppl = Pipeline([
            ('dv', DictVectorizer(sparse=False)),
            ('model', LogisticRegression())
        ])

        ppl.fit(X, y)

        self.trained_pipeline = ppl


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)

    def predict_proba(self, prediction_data):

        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test):
        return self.trained_pipeline.score(X_test, y_test)

