import csv
import random
import sys
import pandas as pd
from auto_ml import Predictor
from auto_ml import utils
random.seed(10)

if len(sys.argv) > 1 and sys.argv[1] in set(['full', 'long', 'full_dataset', 'all_data', 'all']):

    # open full dataset
    with open('numerai_datasets_early_aug/numerai_training_data.csv', 'rU') as input_file:
        training_rows = csv.DictReader(input_file)

        training_data = []

        testing_data = []

        for row in training_rows:
            if random.random() > 0.8:
                testing_data.append(row)
            else:
                training_data.append(row)

    # # write short dataset to file
    # with open('numerai_datasets_early_aug/numerai_short.csv', 'w+') as write_file:
    #     writer = csv.writer(write_file)
    #     writer.writerow(["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","feature12","feature13","feature14","feature15","feature16","feature17","feature18","feature19","feature20","feature21","target"])
    #     for row in training_data_short:
    #         writer.writerow(row)

else:
    # load short dataset
    with open('numerai_training_data.csv', 'rU') as input_file:
        training_rows = csv.DictReader(input_file)

        training_data = []
        testing_data = []

        for row in training_rows:
            if random.random() > 0.8:
                testing_data.append(row)
            else:
                training_data.append(row)

# print type(pd)

print "Testing with dict objects"

ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions={'sentence':'text','target': 'output'})

X_test, y_test = utils.split_output_dataframe(testing_data, output_column_name='target')

# ml_predictor.train(training_data, optimize_entire_pipeline=True, optimize_final_model=True)
ml_predictor.train(training_data)

# ml_predictor.predict_proba(X_test)
print(ml_predictor.score(X_test, y_test))


print "Testing with dataframes"
training_data=pd.DataFrame.from_dict(training_data)
testing_data=pd.DataFrame.from_dict(testing_data)

ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions={'sentence':'text','target': 'output'})

X_test, y_test = utils.split_output_dataframe(testing_data, output_column_name='target')

# ml_predictor.train(training_data, optimize_entire_pipeline=True, optimize_final_model=True)
ml_predictor.train(training_data)

# ml_predictor.predict_proba(X_test)
print(ml_predictor.score(X_test, y_test))

