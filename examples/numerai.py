from auto_ml import Predictor
# import predictor.Predictor

import datetime
import dill
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

df_train = pd.read_csv(os.path.join('numerai_datasets', 'numerai_training_data.csv'))
# Split out 10% of our data to calibrate our probability predictions on
df_train, df_calibrate = train_test_split(df_train, test_size=0.1)

col_descs = {
    'target': 'output'
}


ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_descs)

ml_predictor.train(df_train, optimize_final_model=False, perform_feature_selection=False, perform_feature_scaling=False, X_test=df_calibrate, y_test=df_calibrate.target, calibrate_final_model=True, scoring='log_loss')

file_name = ml_predictor.save('numerai_model_' + str(datetime.datetime.now()))

with open(file_name, 'rb') as read_file:
    trained_model = dill.load(read_file)

df_tournament = pd.read_csv(os.path.join('numerai_datasets', 'numerai_tournament_data.csv'))

predictions = trained_model.predict_proba(df_tournament)

print(predictions)


