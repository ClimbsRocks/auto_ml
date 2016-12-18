import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
try:
    from auto_ml.utils_scoring import ClassificationScorer, RegressionScorer
    from auto_ml.utils_models import get_model_from_name, get_name_from_model
except ImportError:
    from ..auto_ml.utils_scoring import ClassificationScorer, RegressionScorer
    from ..auto_ml.utils_models import get_model_from_name, get_name_from_model
# This is the Air Traffic Controller (ATC) that is a wrapper around sklearn estimators.
# In short, it wraps all the methods the pipeline will look for (fit, score, predict, predict_proba, etc.)
# However, it also gives us the ability to optimize this stage in conjunction with the rest of the pipeline.
# It also gives us more granular control over things like turning the input for GradientBoosting into dense matrices, or appending a set of dummy 1's to the end of sparse matrices getting predictions from XGBoost.
# TODO: make sure we can actually get the params from GridSearchCV.
    # Might have to do something tricky, like have a hold-all function that does nothing but get the params from GridSearchCV inside __init__
        # So, self.model might just be a dictionary or something
        # Or, a function that takes in anything as kwargs, and sets them on a dictionary, then returns that dictionary
    # And then that function does nothing but return those params
    # And we create a model using that inside fit

class FinalModelATC(BaseEstimator, TransformerMixin):


    def __init__(self, model, model_name=None, ml_for_analytics=False, type_of_estimator='classifier', output_column=None, name=None, scoring_method=None, training_features=None, column_descriptions=None):

        self.model = model
        self.model_name = model_name
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator
        self.name = name
        self.training_features = training_features
        self.column_descriptions = column_descriptions


        if self.type_of_estimator == 'classifier':
            self._scorer = scoring_method
        else:
            self._scorer = scoring_method


    def fit(self, X, y):
        self.model_name = get_name_from_model(self.model)

        # if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
        #     ones = [[1] for x in range(X.shape[0])]
        #     # Trying to force XGBoost to play nice with sparse matrices
        #     X_fit = scipy.sparse.hstack((X, ones))

        # else:

        X_fit = X


        if self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression', 'Perceptron', 'PassiveAggressiveClassifier', 'SGDClassifier', 'RidgeClassifier', 'LogisticRegression', ]:
            if scipy.sparse.issparse(X_fit):
                X_fit = X_fit.todense()

        #     num_cols = X_fit.shape[1]
        #     kwargs = {
        #         'num_cols':num_cols
        #         , 'nb_epoch': 20
        #         , 'batch_size': 10
        #         , 'verbose': 1
        #     }
        #     model_params = self.model.get_params()
        #     del model_params['build_fn']
        #     for k, v in model_params.items():
        #         if k not in kwargs:
        #             kwargs[k] = v
        #     if self.type_of_estimator == 'regressor':
        #         self.model = KerasRegressor(build_fn=make_deep_learning_model, **kwargs)

        try:
            self.model.fit(X_fit, y)
        except TypeError as e:
            if scipy.sparse.issparse(X_fit):
                X_fit = X_fit.todense()
            self.model.fit()

        return self

    def verify_features(self, X):

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
                prediction_features.remove(key)

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
        if len(training_not_prediction) > 0:

            print('\n\nHere are the features this model was trained on that were not present in this prediction data:')
            print(sorted(list(training_not_prediction)))
        else:
            print('All of the features this model was trained on are included in the prediction data')

        prediction_not_training = prediction_features - training_features
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

        # if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
        #     ones = [[1] for x in range(X.shape[0])]
        #     # Trying to force XGBoost to play nice with sparse matrices
        #     X = scipy.sparse.hstack((X, ones))

        if (self.model_name[:16] == 'GradientBoosting' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
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

        if X.shape[0] == 1:
            return predictions[0]
        else:
            return predictions

    def predict(self, X, verbose=False):

        # if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
        #     ones = [[1] for x in range(X.shape[0])]
        #     # Trying to force XGBoost to play nice with sparse matrices
        #     X_predict = scipy.sparse.hstack((X, ones))

        if (self.model_name[:16] == 'GradientBoosting' or self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X_predict = X.todense()

        else:
            X_predict = X

        prediction = self.model.predict(X_predict)
        # Handle cases of getting a prediction for a single item.
        # It makes a cleaner interface just to get just the single prediction back, rather than a list with the prediction hidden inside.
        if len(prediction) == 1:
            return prediction[0]
        else:
            return prediction
