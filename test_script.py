import csv
from auto_ml import Predictor

with open('numerai_datasets_early_aug/numerai_training_data.csv', 'rU') as input_file:
    training_rows = csv.DictReader(input_file)

    training_data = []

    for row in training_rows:
        training_data.append(row)

ml_predictor = Predictor(type_of_algo='classifier')

training_data.insert(0, {'target': 'output'})
ml_predictor.train(training_data)
# ml_predictor.predict(new_data)
