from collections import Iterable
import os

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

from auto_ml import utils_models
from auto_ml.utils_models import get_name_from_model

keras_installed = False
try:
    # Suppress some level of logs
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
    keras_installed = True
except:
    pass


# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# It also gives us more granular control over things like turning the input for GradientBoosting into dense matrices, or appending a set of dummy 1's to the end of sparse matrices getting predictions from XGBoost.

class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model, model_name=None, ml_for_analytics=False, type_of_estimator='classifier', output_column=None, name=None, scoring_method=None, training_features=None, column_descriptions=None, feature_learning=False, uncertainty_model=None, uc_results = None):

        self.model = model
        self.model_name = model_name
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator
        self.name = name
        self.training_features = training_features
        self.column_descriptions = column_descriptions
        self.feature_learning = feature_learning
        self.uncertainty_model = uncertainty_model
        self.uc_results = uc_results


        if self.type_of_estimator == 'classifier':
            self._scorer = scoring_method
        else:
            self._scorer = scoring_method


    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default


    def fit(self, X, y):
        self.model_name = get_name_from_model(self.model)

        X_fit = X

        if self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression', 'Perceptron', 'PassiveAggressiveClassifier', 'SGDClassifier', 'RidgeClassifier', 'LogisticRegression']:
            if scipy.sparse.issparse(X_fit):
                X_fit = X_fit.todense()

            if self.model_name[:12] == 'DeepLearning':
                if keras_installed:

                    # For Keras, we need to tell it how many input nodes to expect, which is our num_cols
                    num_cols = X_fit.shape[1]

                    model_params = self.model.get_params()
                    del model_params['build_fn']

                    if self.type_of_estimator == 'regressor':
                        self.model = KerasRegressor(build_fn=utils_models.make_deep_learning_model, num_cols=num_cols, feature_learning=self.feature_learning, **model_params)
                    elif self.type_of_estimator == 'classifier':
                        self.model = KerasClassifier(build_fn=utils_models.make_deep_learning_classifier, num_cols=num_cols, feature_learning=self.feature_learning, **model_params)
                else:
                    print('WARNING: We did not detect that Keras was available.')
                    raise TypeError('A DeepLearning model was requested, but Keras was not available to import')

        try:
            if self.model_name[:12] == 'DeepLearning':

                print('\nWe will stop training early if we have not seen an improvement in training accuracy in 25 epochs')
                from keras.callbacks import EarlyStopping
                early_stopping = EarlyStopping(monitor='loss', patience=25, verbose=1)
                self.model.fit(X_fit, y, callbacks=[early_stopping])

            else:
                self.model.fit(X_fit, y)

        except TypeError as e:
            if scipy.sparse.issparse(X_fit):
                X_fit = X_fit.todense()
            self.model.fit(X_fit, y)

        except KeyboardInterrupt as e:
            print('Stopping training at this point because we heard a KeyboardInterrupt')
            print('If the model is functional at this point, we will output the model in its latest form')
            print('Note that not all models can be interrupted and still used, and that this feature generally is an unofficial beta-release feature that is known to fail on occasion')
            pass

        return self

    def remove_categorical_values(self, features):
        clean_features = set([])
        for feature in features:
            if '=' not in feature:
                clean_features.add(feature)
            else:
                clean_features.add(feature[:feature.index('=')])

        return clean_features

    def verify_features(self, X, raw_features_only=False):

        if self.column_descriptions is None:
            print('This feature is not enabled by default. Depending on the shape of the training data, it can add hundreds of KB to the saved file size.')
            print('Please pass in `ml_predictor.train(data, verify_features=True)` when training a model, and we will enable this function, at the cost of a potentially larger file size.')
            warnings.warn('Please pass verify_features=True when invoking .train() on the ml_predictor instance.')
            return None

        print('\n\nNow verifying consistency between training features and prediction features')
        if isinstance(X, dict):
            prediction_features = set(X.keys())
        elif isinstance(X, pd.DataFrame):
            prediction_features = set(X.columns)

        # If the user passed in categorical features, we will effectively one-hot-encode them ourselves here
        # Note that this assumes we're using the "=" as the separater in DictVectorizer/DataFrameVectorizer
        date_col_names = []
        categorical_col_names = []
        for key, value in self.column_descriptions.items():
            if value == 'categorical' and 'day_part' not in key:
                try:
                    # This covers the case that the user passes in a value in column_descriptions that is not present in their prediction data
                    column_vals = X[key].unique()
                    for val in column_vals:
                        prediction_features.add(key + '=' + str(val))

                    categorical_col_names.append(key)
                except:
                    print('\nFound a column in your column_descriptions that is not present in your prediction data:')
                    print(key)

            elif 'day_part' in key:
                # We have found a date column. Make sure this date column is in our prediction data
                # It is outside the scope of this function to make sure that the same date parts are available in both our training and testing data
                raw_date_col_name = key[:key.index('day_part') - 1]
                date_col_names.append(raw_date_col_name)

            elif value == 'output':
                try:
                    prediction_features.remove(key)
                except KeyError:
                    pass

        # Now that we've added in all the one-hot-encoded categorical columns (name=val1, name=val2), remove the base name from our prediction data
        prediction_features = prediction_features - set(categorical_col_names)

        # Get only the unique raw_date_col_names
        date_col_names = set(date_col_names)

        training_features = set(self.training_features)

        # Remove all of the transformed date column feature names from our training data
        features_to_remove = []
        for feature in training_features:
            for raw_date_col_name in date_col_names:
                if raw_date_col_name in feature:
                    features_to_remove.append(feature)
        training_features = training_features - set(features_to_remove)

        # Make sure the raw_date_col_name is in our training data after we have removed all the transformed feature names
        training_features = training_features | date_col_names

        # MVP means ignoring text features
        print_nlp_warning = False
        nlp_example = None
        for feature in training_features:
            if 'nlp_' in feature:
                print_nlp_warning = True
                nlp_example = feature
                training_features.remove(feature)

        if print_nlp_warning == True:
            print('\n\nWe found an NLP column in the training data')
            print('verify_features() currently does not support checking all of the values within an NLP column, so if the text of your NLP column has dramatically changed, you will have to check that yourself.')
            print('Here is one example of an NLP feature in the training data:')
            print(nlp_example)

        training_not_prediction = training_features - prediction_features

        if raw_features_only == True:
            training_not_prediction = self.remove_categorical_values(training_not_prediction)

        if len(training_not_prediction) > 0:

            print('\n\nHere are the features this model was trained on that were not present in this prediction data:')
            print(sorted(list(training_not_prediction)))
        else:
            print('All of the features this model was trained on are included in the prediction data')

        prediction_not_training = prediction_features - training_features
        if raw_features_only == True:
            prediction_not_training = self.remove_categorical_values(prediction_not_training)

        if len(prediction_not_training) > 0:

            # Separate out those values we were told to ignore by column_descriptions
            ignored_features = []
            for feature in prediction_not_training:
                if self.column_descriptions.get(feature, 'False') == 'ignore':
                    ignored_features.append(feature)
            prediction_not_training = prediction_not_training - set(ignored_features)

            print('\n\nHere are the features available in the prediction data that were not part of the training data:')
            print(sorted(list(prediction_not_training)))

            if len(ignored_features) > 0:
                print('\n\nAdditionally, we found features in the prediction data that we were told to ignore in the training data')
                print(sorted(list(ignored_features)))

        else:
            print('All of the features in the prediction data were in this model\'s training data')

        print('\n\n')
        return {
            'training_not_prediction': training_not_prediction
            , 'prediction_not_training': prediction_not_training
        }


    def score(self, X, y, verbose=False):
        # At the time of writing this, GradientBoosting does not support sparse matrices for predictions
        if (self.model_name[:16] == 'GradientBoosting' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X = X.todense()

        if self._scorer is not None:
            if self.type_of_estimator == 'regressor':
                return self._scorer.score(self, X, y)
            elif self.type_of_estimator == 'classifier':
                return self._scorer.score(self, X, y)


        else:
            return self.model.score(X, y)


    def predict_proba(self, X, verbose=False):

        if (self.model_name[:16] == 'GradientBoosting' or self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X = X.todense()

        try:
            predictions = self.model.predict_proba(X)

        except AttributeError as e:
            # print('This model has no predict_proba method. Returning results of .predict instead.')
            try:
                predictions = self.model.predict(X)
            except TypeError as e:
                if scipy.sparse.issparse(X):
                    X = X.todense()
                predictions = self.model.predict(X)

        except TypeError as e:
            if scipy.sparse.issparse(X):
                X = X.todense()
            predictions = self.model.predict_proba(X)

        # If this model does not have predict_proba, and we have fallen back on predict, we want to make sure we give results back in the same format the user would expect for predict_proba, namely each prediction is a list of predicted probabilities for each class.
        # Note that this DOES NOT WORK for multi-label problems, or problems that are not reduced to 0,1
        # If this is not an iterable (ignoring strings, which might be iterable), then we will want to turn our predictions into tupled predictions
        if not (hasattr(predictions[0], '__iter__') and not isinstance(predictions[0], str)):
            tupled_predictions = []
            for prediction in predictions:
                if prediction == 1:
                    tupled_predictions.append([0,1])
                else:
                    tupled_predictions.append([1,0])
            predictions = tupled_predictions


        # This handles an annoying edge case with libraries like Keras that, for a binary classification problem, with return a single predicted probability in a list, rather than the probability of both classes in a list
        if len(predictions[0]) == 1:
            tupled_predictions = []
            for prediction in predictions:
                tupled_predictions.append([1 - prediction[0], prediction[0]])
            predictions = tupled_predictions

        if X.shape[0] == 1:
            return predictions[0]
        else:
            return predictions

    def predict(self, X, verbose=False):

        if (self.model_name[:16] == 'GradientBoosting' or self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X_predict = X.todense()

        else:
            X_predict = X

        prediction = self.model.predict(X_predict)
        # Handle cases of getting a prediction for a single item.
        # It makes a cleaner interface just to get just the single prediction back, rather than a list with the prediction hidden inside.

        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
            if isinstance(prediction, float) or isinstance(prediction, int) or isinstance(prediction, str):
                return prediction

        if len(prediction) == 1:
            return prediction[0]
        else:
            return prediction

    # transform is initially designed to be used with feature_learning
    def transform(self, X):
        predicted_features = self.predict(X)
        predicted_features = list(predicted_features)

        if scipy.sparse.issparse(X):
            X = scipy.sparse.hstack([X, predicted_features], format='csr')
        else:
            print('Figuring out what type X is')
            print(type(X))
            print('If you see this message, please file a bug at https://github.com/ClimbsRocks/auto_ml')

        return X

    def predict_uncertainty(self, X):
        if self.uncertainty_model is None:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('This model was not trained to predict uncertainties')
            print('Please follow the documentation to tell this model at training time to learn how to predict uncertainties')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            raise ValueError('This model was not trained to predict uncertainties')

        base_predictions = self.predict(X)

        if isinstance(base_predictions, Iterable):
            base_predictions_col = [[val] for val in base_predictions]
            base_predictions_col = np.array(base_predictions_col)
        else:
            base_predictions_col = [base_predictions]

        X_combined = scipy.sparse.hstack([X, base_predictions_col], format='csr')

        uncertainty_predictions = self.uncertainty_model.predict_proba(X_combined)

        results = {
            'base_prediction': base_predictions
            , 'uncertainty_prediction': uncertainty_predictions
        }



        if isinstance(base_predictions, Iterable):

            results['uncertainty_prediction'] = [row[1] for row in results['uncertainty_prediction']]

            results = pd.DataFrame.from_dict(results, orient='columns')

            if self.uc_results is not None:
                calibration_results = {}
                # grab the relevant properties from our uc_results, and make them each their own list in calibration_results
                for key, value in self.uc_results[1].items():
                    calibration_results[key] = []

                for proba in results['uncertainty_prediction']:
                    max_bucket_proba = 0
                    bucket_num = 1
                    while proba > max_bucket_proba:
                        calibration_result = self.uc_results[bucket_num]
                        max_bucket_proba = self.uc_results[bucket_num]['max_proba']
                        bucket_num += 1

                    for key, value in calibration_result.items():
                        calibration_results[key].append(value)
                # TODO: grab the uncertainty_calibration data for DataFrames
                df_calibration_results = pd.DataFrame.from_dict(calibration_results, orient='columns')
                del df_calibration_results['max_proba']

                results = pd.concat([results, df_calibration_results], axis=1)

        else:
            if self.uc_results is not None:
                # TODO: grab the uncertainty_calibration data for dictionaries
                for bucket_name, bucket_result in self.uc_results.items():
                    if proba > bucket_result['max_proba']:
                        break
                    results.update(bucket_result)
                    del results['max_proba']




        return results


    def score_uncertainty(self, X, y, verbose=False):
        return self.uncertainty_model.score(X, y, verbose=False)






