# """
# To get standard out, run nosetests as follows:
#   $ nosetests -s tests
# """
# import os
# import sys

# sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

# import dill
# from nose.tools import assert_equal, assert_not_equal, with_setup
# import numpy as np
# import random
# from sklearn.model_selection import train_test_split

# import utils_testing as utils


# def test_basic_ensemble_classifier():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
#     ml_predictor = utils.make_titanic_ensemble(df_titanic_train)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived, verbose=0)
#     # Very rough ensembles don't do as well on this problem as a standard GradientBoostingClassifier does
#     # Right now we're getting a score of -.22
#     # Make sure our score is good, but not unreasonably good
#     assert -0.225 < test_score < -0.17


# def test_saving_basic_ensemble_classifier():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
#     ml_predictor = utils.make_titanic_ensemble(df_titanic_train)

#     file_name = ml_predictor.save(str(random.random()))

#     with open(file_name, 'rb') as read_file:
#         saved_ml_pipeline = dill.load(read_file)
#     os.remove(file_name)


#     probas = saved_ml_pipeline.predict_proba(df_titanic_test)
#     probas = [proba[1] for proba in probas]
#     # print(probas)

#     test_score = utils.calculate_brier_score_loss(df_titanic_test.survived, probas)
#     print('test_score')
#     print(test_score)

#     # Very rough ensembles don't do as well on this problem as a standard GradientBoostingClassifier does
#     # Right now we're getting a score of -.22
#     # Make sure our score is good, but not unreasonably good
#     assert -0.225 < test_score < -0.17


# def test_get_basic_ensemble_predictions_one_at_a_time_classifier():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
#     ml_predictor = utils.make_titanic_ensemble(df_titanic_train)
#     file_name = ml_predictor.save(str(random.random()))

#     with open(file_name, 'rb') as read_file:
#         saved_ml_pipeline = dill.load(read_file)
#     os.remove(file_name)

#     df_titanic_test_dictionaries = df_titanic_test.to_dict('records')

#     # These predictions take a while. So we'll cut out 80% of our data to make this run much faster
#     df_titanic_test_dictionaries, df_titanic_test_dictionaries_ignored, df_titanic_test, df_titanic_test_ignored = train_test_split(df_titanic_test_dictionaries, df_titanic_test, train_size=0.05, random_state=0)

#     # 1. make sure the accuracy is the same

#     predictions = []
#     for row in df_titanic_test_dictionaries:
#         prediction = saved_ml_pipeline.predict_proba(row)
#         predictions.append(prediction)

#     first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
#     print('first_score')
#     print(first_score)
#     # Make sure our score is good, but not unreasonably good
#     assert -0.235 < first_score < -0.17

#     # 2. make sure the speed is reasonable (do it a few extra times)
#     # data_length = len(df_titanic_test_dictionaries)
#     # start_time = datetime.datetime.now()
#     # for idx in range(1000):
#     #     row_num = idx % data_length
#     #     saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
#     # end_time = datetime.datetime.now()
#     # duration = end_time - start_time

#     # print('duration.total_seconds()')
#     # print(duration.total_seconds())

#     # # It's very difficult to set a benchmark for speed that will work across all machines.
#     # # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
#     # # That's about 1 millisecond per prediction
#     # # Assuming we might be running on a test box that's pretty weak, multiply by 3
#     # # Also make sure we're not running unreasonably quickly
#     # assert 0.4 < duration.total_seconds() < 3


#     # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

#     predictions = []
#     for row in df_titanic_test_dictionaries:
#         predictions.append(saved_ml_pipeline.predict_proba(row))

#     second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
#     # Make sure our score is good, but not unreasonably good
#     assert -0.235 < second_score < -0.17


# # All these tests hang on scikit-learn's GSCV multiprocessing bug

# def test_ml_ensemble_classifier():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
#     ml_predictor = utils.make_titanic_ensemble(df_titanic_train, method='ml')

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived, verbose=0)
#     # Very rough ensembles don't do as well on this problem as a standard GradientBoostingClassifier does
#     # Right now we're getting a score of -.22
#     # Make sure our score is good, but not unreasonably good
#     print('test_score')
#     print(test_score)
#     assert -0.225 < test_score < -0.17


# def test_saving_ml_ensemble_classifier():
#     np.random.seed(0)

#     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
#     ml_predictor = utils.make_titanic_ensemble(df_titanic_train, method='ml')

#     file_name = ml_predictor.save(str(random.random()))

#     with open(file_name, 'rb') as read_file:
#         saved_ml_pipeline = dill.load(read_file)
#     os.remove(file_name)


#     probas = saved_ml_pipeline.predict_proba(df_titanic_test)
#     probas = [proba[1] for proba in probas]
#     # print(probas)

#     test_score = utils.calculate_brier_score_loss(df_titanic_test.survived, probas)
#     print('test_score')
#     print(test_score)

#     # Very rough ensembles don't do as well on this problem as a standard GradientBoostingClassifier does
#     # Right now we're getting a score of -.22
#     # Make sure our score is good, but not unreasonably good
#     assert -0.225 < test_score < -0.17


# # def test_get_ml_ensemble_predictions_one_at_a_time_classifier():
# #     np.random.seed(0)

# #     df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
# #     ml_predictor = utils.make_titanic_ensemble(df_titanic_train, method='ml')
# #     file_name = ml_predictor.save(str(random.random()))

# #     with open(file_name, 'rb') as read_file:
# #         saved_ml_pipeline = dill.load(read_file)
#     # os.remove(file_name)

# #     df_titanic_test_dictionaries = df_titanic_test.to_dict('records')

# #     # These predictions take a while. So we'll cut out 80% of our data to make this run much faster
# #     df_titanic_test_dictionaries, df_titanic_test_dictionaries_ignored, df_titanic_test, df_titanic_test_ignored = train_test_split(df_titanic_test_dictionaries, df_titanic_test, train_size=0.05, random_state=0)

# #     # 1. make sure the accuracy is the same

# #     predictions = []
# #     for row in df_titanic_test_dictionaries:
# #         prediction = saved_ml_pipeline.predict_proba(row)
# #         # print(prediction)
# #         # print(type(prediction))
# #         # print(prediction[0])
# #         # print(type(prediction[0]))
# #         predictions.append(prediction)

# #     print('predictions inside our failing test')
# #     print(predictions)

# #     first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
# #     print('first_score')
# #     print(first_score)
# #     # Make sure our score is good, but not unreasonably good
# #     assert -0.235 < first_score < -0.17

# #     # 2. make sure the speed is reasonable (do it a few extra times)
# #     # data_length = len(df_titanic_test_dictionaries)
# #     # start_time = datetime.datetime.now()
# #     # for idx in range(1000):
# #     #     row_num = idx % data_length
# #     #     saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
# #     # end_time = datetime.datetime.now()
# #     # duration = end_time - start_time

# #     # print('duration.total_seconds()')
# #     # print(duration.total_seconds())

# #     # # It's very difficult to set a benchmark for speed that will work across all machines.
# #     # # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
# #     # # That's about 1 millisecond per prediction
# #     # # Assuming we might be running on a test box that's pretty weak, multiply by 3
# #     # # Also make sure we're not running unreasonably quickly
# #     # assert 0.4 < duration.total_seconds() < 3


# #     # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

# #     predictions = []
# #     for row in df_titanic_test_dictionaries:
# #         predictions.append(saved_ml_pipeline.predict_proba(row))

# #     second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
# #     # Make sure our score is good, but not unreasonably good
# #     assert -0.235 < second_score < -0.17


