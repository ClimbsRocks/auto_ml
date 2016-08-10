from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import utils

from sklearn.grid_search import GridSearchCV


class Predictor(object):


    def __init__(self, type_of_algo, column_descriptions, verbose=True):
        self.type_of_algo = type_of_algo
        self.column_descriptions = column_descriptions
        self.verbose = verbose
        self.trained_pipeline = None

        # figure out which column has a value 'output'
        output_column = [key for key, value in column_descriptions.items() if value.lower() == 'output'][0]
        self.output_column = output_column


    def _construct_pipeline(self, user_input_func=None, optimize_final_model=False):

        pipeline_list = []
        if user_input_func is not None:
            pipeline_list.append(('user_func', FunctionTransformer(func=user_input_func, pass_y=False, validate=False) ))

        pipeline_list.append(('basic_transform', utils.BasicDataCleaning()))
        pipeline_list.append(('dv', DictVectorizer(sparse=True)))
        pipeline_list.append(('final_model', utils.FinalModelATC(model_name='LogisticRegression', perform_grid_search_on_model=optimize_final_model)))

        constructed_pipeline = Pipeline(pipeline_list)
        return constructed_pipeline


    def train(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False, print_analytics_output=False):

        # split out out output column so we have a proper X, y dataset
        output_splitter = utils.SplitOutput(self.output_column)
        X, y = output_splitter.transform(raw_training_data)

        ppl = self._construct_pipeline(user_input_func, optimize_final_model)

        # print('Pipeline')
        # print(ppl)

        if optimize_entire_pipeline:
            self.grid_search_params = {
                'final_model__model_name': ['RandomForestClassifier', 'LogisticRegression']
            }

            # Note that this is computationally expensive and extremely time consuming.
            if optimize_final_model:
                # We can alternately try the raw, default, non-optimized algorithm used in our final_model stage, and also test optimizing that algorithm, in addition to optimizing the entire pipeline.
                # We could choose to bake this into the broader pipeline GridSearchCV, but that risks becoming too cumbersome, and might be impossible since we're trying so many different models that have different parameters.
                self.grid_search_params['final_model__perform_grid_search_on_model'] = [True, False]

            gs = GridSearchCV(
                # Fit on the pipeline.
                ppl,
                self.grid_search_params,
                # Train across all cores.
                n_jobs=-1,
                # Be verbose (lots of printing).
                verbose=10,
                # Print warnings when we fail to fit a given combination of parameters, but do not raise an error.
                error_score=10,
                # TODO(PRESTON): change scoring to be RMSE by default
                scoring=None
            )

            gs.fit(X, y)
            self.trained_pipeline = gs

        else:
            ppl.fit(X, y)

            self.trained_pipeline = ppl



    def ml_for_analytics(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False, print_analytics_output=False):

        # split out out output column so we have a proper X, y dataset
        output_splitter = utils.SplitOutput(self.output_column)
        X, y = output_splitter.transform(raw_training_data)

        ppl = self._construct_pipeline(user_input_func, optimize_final_model)

        if optimize_entire_pipeline:
            self.grid_search_params = {
                'final_model__model_name': ['RandomForestClassifier', 'LogisticRegression']
            }

            # Note that this is computationally expensive and extremely time consuming.
            if optimize_final_model:
                # We can alternately try the raw, default, non-optimized algorithm used in our final_model stage, and also test optimizing that algorithm, in addition to optimizing the entire pipeline.
                # We could choose to bake this into the broader pipeline GridSearchCV, but that risks becoming too cumbersome, and might be impossible since we're trying so many different models that have different parameters.
                self.grid_search_params['final_model__perform_grid_search_on_model'] = [True, False]

            gs = GridSearchCV(
                # Fit on the pipeline.
                ppl,
                self.grid_search_params,
                # Train across all cores.
                n_jobs=-1,
                # Be verbose (lots of printing).
                verbose=10,
                # Print warnings when we fail to fit a given combination of parameters, but do not raise an error.
                error_score=10,
                # TODO(PRESTON): change scoring to be RMSE by default
                scoring=None
            )

            gs.fit(X, y)
            self.trained_pipeline = gs.best_estimator_

        else:
            ppl.fit(X, y)
            self.trained_pipeline = ppl


        feature_names = self.trained_pipeline.named_steps['dv'].get_feature_names()
        print("self.trained_pipeline.named_steps['dv'].get_feature_names()")
        print(feature_names)


        # TODO(PRESTON)
        # Figure out how to access the FinalModelATC from our pipeline
        # Figure out how to access the model from FinalModelATC
        # Figure out how to get the coefficients from the best regression and random forest
        # Figure out how to get that particular model's features from DictVectorizer (we will be doing a lot of feature engineering and feature selection in very near term versions of this repo)
            # Might have to wrap DictVectorizer in a class that writes the results to the pipeline object or something?
        # consider letting them pass this in as a flag for train. would probably be much easier to calculate these things if we know to beforehand
        # look into putting some logic into FinalModelATC that keeps the best model/parameters around in an easy way for analytics.
        # Figure out the reasonable range for whatever features we do have left, for regression printing


    def print_training_summary(self):
        pass
        # Print some nice summary output of all the training we did.
        # maybe allow the user to pass in a flag to write info to a file


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)

    def predict_proba(self, prediction_data):

        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test):
        return self.trained_pipeline.score(X_test, y_test)

