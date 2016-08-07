import csv
import random
from auto_ml import Predictor
from auto_ml import utils

# # open full dataset
# with open('numerai_datasets_early_aug/numerai_training_data.csv', 'rU') as input_file:
#     training_rows = csv.reader(input_file)

#     training_data = []

#     training_data_short = []

#     for row in training_rows:
#         if random.random() > 0.98:
#             training_data_short.append(row)
#         training_data.append(row)

# # write short dataset to file
# with open('numerai_datasets_early_aug/numerai_short.csv', 'w+') as write_file:
#     writer = csv.writer(write_file)
#     writer.writerow(["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","feature12","feature13","feature14","feature15","feature16","feature17","feature18","feature19","feature20","feature21","target"])
#     for row in training_data_short:
#         writer.writerow(row)

# load short dataset
with open('numerai_datasets_early_aug/numerai_short.csv', 'rU') as input_file:
    training_rows = csv.DictReader(input_file)

    training_data = []
    testing_data = []

    for row in training_rows:
        if random.random() > 0.8:
            testing_data.append(row)
        else:
            training_data.append(row)


ml_predictor = Predictor(type_of_algo='classifier', column_descriptions={'target': 'output'})

ml_predictor.train(training_data)

output_splitter = utils.SplitOutput('target')
X_test, y_test = output_splitter.transform(testing_data)

print(ml_predictor.predict_proba(X_test))
