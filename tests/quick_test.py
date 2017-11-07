# """
# nosetests -sv --nologcapture tests/quick_test.py
# nosetests --verbosity=2 --detailed-errors --nologcapture --processes=4 --process-restartworker --process-timeout=1000 tests/quick_test.py
# """

# import datetime
# import os
# import random
# import sys
# sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
# os.environ['is_test_suite'] = 'True'
# os.environ['KERAS_BACKEND'] = 'theano'

# from auto_ml import Predictor
# from auto_ml.utils_models import load_ml_model

# from nose.tools import assert_equal, assert_not_equal, with_setup
# from sklearn.metrics import accuracy_score

# import dill
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split


# def get_boston_regression_dataset():
#     boston = load_boston()
#     df_boston = pd.DataFrame(boston.data)
#     df_boston.columns = boston.feature_names
#     df_boston['MEDV'] = boston['target']
#     df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.33, random_state=42)
#     return df_boston_train, df_boston_test


# def regression_test():
#     # a random seed of 42 has ExtraTreesRegressor getting the best CV score, and that model doesn't generalize as well as GradientBoostingRegressor.
#     np.random.seed(0)
#     model_name = 'LGBMRegressor'

#     df_boston_train, df_boston_test = get_boston_regression_dataset()
#     many_dfs = []
#     for i in range(100):
#         many_dfs.append(df_boston_train)
#     df_boston_train = pd.concat(many_dfs)


#     column_descriptions = {
#         'MEDV': 'output'
#         , 'CHAS': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

#     ml_predictor.train(df_boston_train, model_names=[model_name], perform_feature_scaling=False)

#     test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

#     print('test_score')
#     print(test_score)

#     lower_bound = -3.2
#     if model_name == 'DeepLearningRegressor':
#         lower_bound = -7.8
#     if model_name == 'LGBMRegressor':
#         lower_bound = -4.95
#     if model_name == 'XGBRegressor':
#         lower_bound = -3.4

#     assert lower_bound < test_score < -2.8


# def get_titanic_binary_classification_dataset(basic=True):
#     try:
#         df_titanic = pd.read_csv(os.path.join('tests', 'titanic.csv'))
#     except Exception as e:
#         print('Error')
#         print(e)
#         dataset_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv'
#         df_titanic = pd.read_csv(dataset_url)
#         # Do not write the index that pandas automatically creates
#         df_titanic.to_csv(os.path.join('tests', 'titanic.csv'), index=False)

#     df_titanic = df_titanic.drop(['boat', 'body'], axis=1)

#     if basic == True:
#         df_titanic = df_titanic.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)

#     df_titanic_train, df_titanic_test = train_test_split(df_titanic, test_size=0.33, random_state=42)
#     return df_titanic_train, df_titanic_test


# def classification_test():
#     np.random.seed(0)
#     # model_name = 'GradientBoostingClassifier'
#     model_name = 'LGBMClassifier'

#     df_titanic_train, df_titanic_test = get_titanic_binary_classification_dataset()

#     column_descriptions = {
#         'survived': 'output'
#         , 'embarked': 'categorical'
#         , 'pclass': 'categorical'
#         , 'sex': 'categorical'
#     }

#     ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

#     ml_predictor.train(df_titanic_train, model_names=model_name)

#     test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

#     print('test_score')
#     print(test_score)

#     lower_bound = -0.16
#     if model_name == 'DeepLearningClassifier':
#         lower_bound = -0.245
#     if model_name == 'LGBMClassifier':
#         lower_bound = -0.225

#     assert lower_bound < test_score < -0.135

# if __name__ == '__main__':
#     regression_test()
