

class Predictor(object):

    def __init__(self, type_of_algo):
        self.type_of_algo = type_of_algo

    def train(self, raw_training_data):


        self.trained_pipeline = trained_pipeline


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)
