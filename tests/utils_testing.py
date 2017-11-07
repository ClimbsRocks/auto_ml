import sys, os
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

def get_boston_regression_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.33, random_state=42)
    return df_boston_train, df_boston_test

def get_titanic_binary_classification_dataset(basic=True):
    try:
        df_titanic = pd.read_csv(os.path.join('tests', 'titanic.csv'))
    except Exception as e:
        print('Error')
        print(e)
        dataset_url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv'
        df_titanic = pd.read_csv(dataset_url)
        # Do not write the index that pandas automatically creates
        df_titanic.to_csv(os.path.join('tests', 'titanic.csv'), index=False)

    df_titanic = df_titanic.drop(['boat', 'body'], axis=1)

    if basic == True:
        df_titanic = df_titanic.drop(['name', 'ticket', 'cabin', 'home.dest'], axis=1)

    df_titanic_train, df_titanic_test = train_test_split(df_titanic, test_size=0.33, random_state=42)
    return df_titanic_train, df_titanic_test


def train_basic_binary_classifier(df_titanic_train):
    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train)

    return ml_predictor


def train_basic_regressor(df_boston_train):
    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, verbose=False)
    return ml_predictor

def calculate_rmse(actuals, preds):
    return mean_squared_error(actuals, preds)**0.5 * -1

def calculate_brier_score_loss(actuals, probas):
    return -1 * brier_score_loss(actuals, probas)



def get_twitter_sentiment_multilabel_classification_dataset():

    file_name = os.path.join('tests', 'twitter_sentiment.csv')

    try:
        df_twitter = pd.read_csv(open(file_name,'rU'), encoding='latin-1', engine='python')
    except Exception as e:
        print('Error')
        print(e)
        dataset_url = 'https://raw.githubusercontent.com/ClimbsRocks/sample_datasets/master/twitter_airline_sentiment.csv'
        df_twitter = pd.read_csv(dataset_url, encoding='latin-1')
        # Do not write the index that pandas automatically creates

        df_twitter.to_csv(file_name, index=False, encoding='latin-1')

    # Grab only 10% of the dataset- runs much faster this way
    df_twitter = df_twitter.sample(frac=0.1)

    df_twitter['tweet_created'] = pd.to_datetime(df_twitter.tweet_created)

    df_twitter_train, df_twitter_test = train_test_split(df_twitter, test_size=0.33, random_state=42)
    return df_twitter_train, df_twitter_test


def train_basic_multilabel_classifier(df_twitter_train):
    column_descriptions = {
        'airline_sentiment': 'output'
        , 'airline': 'categorical'
        , 'text': 'ignore'
        , 'tweet_location': 'categorical'
        , 'user_timezone': 'categorical'
        , 'tweet_created': 'date'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_twitter_train)

    return ml_predictor
