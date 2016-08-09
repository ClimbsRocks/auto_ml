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

        output_column = [key for key, value in column_descriptions.items() if value.lower() == 'output'][0]
        self.output_column = output_column


    def train(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False):

        # split out out output column so we have a proper X, y dataset
        output_splitter = utils.SplitOutput(self.output_column)
        X, y = output_splitter.transform(raw_training_data)

        X_train = 'this is a test'

        ppl = Pipeline([
            ('user_func', FunctionTransformer(func=user_input_func, pass_y=False, validate=False)),
            ('dv', DictVectorizer(sparse=True)),
            ('final_model', utils.FinalModelATC(model_name='LogisticRegression', perform_grid_search_on_model=optimize_final_model))
        ])

        if optimize_entire_pipeline:
            grid_search_params = {
                'final_model__model_name': ['RandomForestClassifier', 'LogisticRegression']
            }

            # Note that this is computationally expensive and extremely time consuming.
            if optimize_final_model:
                # We can alternately try the raw, default, non-optimized algorithm used in our final_model stage, and also test optimizing that algorithm, in addition to optimizing the entire pipeline.
                # We could choose to bake this into the broader pipeline GridSearchCV, but that risks becoming too cumbersome, and might be impossible since we're trying so many different models that have different parameters.
                grid_search_params['final_model__perform_grid_search_on_model'] = [True, False]

            gs = GridSearchCV(
                # Fit on the pipeline.
                ppl,
                grid_search_params,
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


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)

    def predict_proba(self, prediction_data):

        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test):
        return self.trained_pipeline.score(X_test, y_test)

