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

        ppl = Pipeline([
            ('split_output', utils.SplitOutput())
            ('dv', DictVectorizer()),
            ('model', LogisticRegression())
        ])

        ppl.fit()

        self.trained_pipeline = trained_pipeline


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)

