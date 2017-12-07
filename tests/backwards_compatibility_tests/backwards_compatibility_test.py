import datetime
import os
import sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.abspath(os.path.dirname(os.path.dirname(__file__)))] + sys.path

from auto_ml import Predictor, __version__ as auto_ml_version
from auto_ml.utils_models import load_ml_model
import dill
import numpy as np
import utils_testing as utils


if 'backwards_compatibility' in os.environ.get('TESTS_TO_RUN', 'blank'):
    def test_backwards_compatibility_with_version_2_1_6():
        np.random.seed(0)
        print('auto_ml_version')
        print(auto_ml_version)
        if auto_ml_version <= '2.9.0':
            raise(TypeError)

        df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

        saved_ml_pipeline = load_ml_model(os.path.join('tests', 'backwards_compatibility_tests', 'trained_ml_model_v_2_1_6.dill'))

        df_titanic_test_dictionaries = df_titanic_test.to_dict('records')

        # 1. make sure the accuracy is the same

        predictions = []
        for row in df_titanic_test_dictionaries:
            predictions.append(saved_ml_pipeline.predict_proba(row)[1])

        print('predictions')
        print(predictions)

        first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
        print('first_score')
        print(first_score)
        # Make sure our score is good, but not unreasonably good

        lower_bound = -0.215

        assert lower_bound < first_score < -0.17

        # 2. make sure the speed is reasonable (do it a few extra times)
        data_length = len(df_titanic_test_dictionaries)
        start_time = datetime.datetime.now()
        for idx in range(1000):
            row_num = idx % data_length
            saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        print('duration.total_seconds()')
        print(duration.total_seconds())

        # It's very difficult to set a benchmark for speed that will work across all machines.
        # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
        # That's about 1 millisecond per prediction
        # Assuming we might be running on a test box that's pretty weak, multiply by 3
        # Also make sure we're not running unreasonably quickly
        assert 0.2 < duration.total_seconds() < 15


        # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

        predictions = []
        for row in df_titanic_test_dictionaries:
            predictions.append(saved_ml_pipeline.predict_proba(row)[1])

        print('predictions')
        print(predictions)
        print('df_titanic_test_dictionaries')
        print(df_titanic_test_dictionaries)
        second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
        print('second_score')
        print(second_score)
        # Make sure our score is good, but not unreasonably good

        assert lower_bound < second_score < -0.17


def train_old_model():
    print('auto_ml_version')
    print(auto_ml_version)
    if auto_ml_version > '2.1.6':
        raise(TypeError)

    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output'
        , 'sex': 'categorical'
        , 'embarked': 'categorical'
        , 'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)

    ml_predictor.train(df_titanic_train)

    file_name = ml_predictor.save('trained_ml_model_v_2_1_6.dill')

    saved_ml_pipeline = load_ml_model(file_name)

    df_titanic_test_dictionaries = df_titanic_test.to_dict('records')

    # 1. make sure the accuracy is the same

    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    # Make sure our score is good, but not unreasonably good

    lower_bound = -0.16

    assert -0.16 < first_score < -0.135

    # 2. make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_titanic_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print('duration.total_seconds()')
    print(duration.total_seconds())

    # It's very difficult to set a benchmark for speed that will work across all machines.
    # On my 2013 bottom of the line 15" MacBook Pro, this runs in about 0.8 seconds for 1000 predictions
    # That's about 1 millisecond per prediction
    # Assuming we might be running on a test box that's pretty weak, multiply by 3
    # Also make sure we're not running unreasonably quickly
    assert 0.2 < duration.total_seconds() < 15


    # 3. make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)

    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    # Make sure our score is good, but not unreasonably good

    assert -0.16 < second_score < -0.135

if __name__ == '__main__':
    train_old_model()
