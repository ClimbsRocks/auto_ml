# This set of tests id specifically designed to make sure auto_ml is user friendly- throwing useful warnings where possible about what specific actions the user can take to avoid an error, instead of throwing the non-obvious error messages that the underlying libraries will choke on.
import warnings
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from auto_ml import Predictor

import utils_testing as utils


def test_unmarked_date_column():
    df_twitter_train, df_twitter_test = utils.get_twitter_sentiment_multilabel_classification_dataset()

    column_descriptions = {
        'airline_sentiment': 'output'
        , 'airline': 'categorical'
        , 'text': 'nlp'
        , 'tweet_location': 'categorical'
        , 'user_timezone': 'categorical'
        # tweet_created is our date column. We want to test that "forgetting" to mark it as a date column throws a user warning before throwing the error
        # , 'tweet_created': 'date'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    with warnings.catch_warnings(record=True) as w:

        try:
            ml_predictor.train(df_twitter_train)
        except TypeError as e:
            pass

        print('Here are the caught warnings:')
        print(w)

        assert len(w) == 1

