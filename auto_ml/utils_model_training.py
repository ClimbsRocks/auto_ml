import scipy
from sklearn.base import BaseEstimator, TransformerMixin
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


    def __init__(self, model, model_name=None, ml_for_analytics=False, type_of_estimator='classifier', output_column=None, name=None, scoring_method=None):

        self.model = model
        self.model_name = model_name
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator
        self.name = name


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


    def score(self, X, y):
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


    def predict_proba(self, X):

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

    def predict(self, X):

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
