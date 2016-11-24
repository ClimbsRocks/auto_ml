import scipy
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from auto_ml.utils_scoring import brier_score_loss_wrapper, rmse_scoring
except ImportError:
    from ..auto_ml.utils_scoring import brier_score_loss_wrapper, rmse_scoring
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


    def __init__(self, model, model_name, ml_for_analytics=False, type_of_estimator='classifier', output_column=None, name=None):

        self.model = model
        self.model_name = model_name
        self.ml_for_analytics = ml_for_analytics
        self.type_of_estimator = type_of_estimator
        self.name = name


        if self.type_of_estimator == 'classifier':
            self._scorer = brier_score_loss_wrapper
        else:
            self._scorer = rmse_scoring


    def fit(self, X, y):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X_fit = scipy.sparse.hstack((X, ones))

        else:
            X_fit = X


        if self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']:
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

        self.model.fit(X_fit, y)

        return self


    def score(self, X, y):
        # At the time of writing this, GradientBoosting does not support sparse matrices for predictions
        if (self.model_name[:16] == 'GradientBoosting' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X = X.todense()

        if self._scorer is not None:
            if self.type_of_estimator == 'regressor':
                return self._scorer(self, X, y)
            elif self.type_of_estimator == 'classifier':
                return self._scorer(self, X, y)


        else:
            return self.model.score(X, y)


    def predict_proba(self, X):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X = scipy.sparse.hstack((X, ones))

        if (self.model_name[:16] == 'GradientBoosting' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
            X = X.todense()

        try:
            predictions = self.model.predict_proba(X)
            if X.shape[0] == 1:
                return predictions[0]
            else:
                return predictions
        except AttributeError as e:
            # print('This model has no predict_proba method. Returning results of .predict instead.')
            raw_predictions = self.model.predict(X)
            tupled_predictions = []
            for prediction in raw_predictions:
                if prediction == 1:
                    tupled_predictions.append([0,1])
                else:
                    tupled_predictions.append([1,0])
            predictions = tupled_predictions
            # return tupled_predictions
            if X.shape[0] == 1:
                return predictions[0]
            else:
                return predictions


    def predict(self, X):

        if self.model_name[:3] == 'XGB' and scipy.sparse.issparse(X):
            ones = [[1] for x in range(X.shape[0])]
            # Trying to force XGBoost to play nice with sparse matrices
            X_predict = scipy.sparse.hstack((X, ones))

        elif (self.model_name[:16] == 'GradientBoosting' or self.model_name[:12] == 'DeepLearning' or self.model_name in ['BayesianRidge', 'LassoLars', 'OrthogonalMatchingPursuit', 'ARDRegression']) and scipy.sparse.issparse(X):
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
