import os
import warnings

from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, brier_score_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import utils

# with warnings.catch_warnings():
warnings.filterwarnings("ignore", category=UserWarning)


class Predictor(object):


    def __init__(self, type_of_algo, column_descriptions, verbose=True):
        self.type_of_algo = type_of_algo
        self.column_descriptions = column_descriptions
        self.verbose = verbose
        self.trained_pipeline = None
        self._scorer = None

        # figure out which column has a value 'output'
        output_column = [key for key, value in column_descriptions.items() if value.lower() == 'output'][0]
        self.output_column = output_column


    def _construct_pipeline(self, user_input_func=None, ml_for_analytics=False, model_name='LogisticRegression', optimize_final_model=False, perform_feature_selection=True, impute_missing_values=True):

        pipeline_list = []
        if user_input_func is not None:
            pipeline_list.append(('user_func', FunctionTransformer(func=user_input_func, pass_y=False, validate=False) ))

        # These parts will be included no matter what.
        pipeline_list.append(('basic_transform', utils.BasicDataCleaning(column_descriptions=self.column_descriptions)))
        pipeline_list.append(('dv', DictVectorizer(sparse=True)))

        if perform_feature_selection:
            pipeline_list.append(('feature_selection', utils.FeatureSelectionTransformer(type_of_model=self.type_of_algo, feature_selection_model='SelectFromModel') ))

        # if any xgboost estimators are part of our pipeline, we will need to use our DMatrixTransformer to get the data into the right format for XGBoost
        # for estimator_name in self.gs_params.get('final_model__model_name'):
        #     if estimator_name.lower()[:3] == 'xgb':
        #         pipeline_list.append(('xgb_dmatrix_transformer', utils.DMatrixTransformer() ))

        pipeline_list.append(('final_model', utils.FinalModelATC(model_name=model_name, perform_grid_search_on_model=optimize_final_model, type_of_model=self.type_of_algo)))

        constructed_pipeline = Pipeline(pipeline_list)
        return constructed_pipeline


    def _construct_pipeline_search_params(self, optimize_entire_pipeline=True, optimize_final_model=False, ml_for_analytics=False, perform_feature_selection=True):

        gs_params = {}

        if optimize_final_model:
            gs_params['final_model__perform_grid_search_on_model'] = [True, False]

        if ml_for_analytics:
            gs_params['final_model__ml_for_analytics'] = [True]

        else:
            if optimize_entire_pipeline:
                gs_params['final_model__model_name'] = self._get_estimator_names(ml_for_analytics=ml_for_analytics)

        if perform_feature_selection:
            # We also have support built in for RFECV, but that typically takes way too long
            # We've also built in support for 'RandomizedSparse' feature selection methods, but they don't always support sparse matrices, so we are ignoring them by default.
            gs_params['feature_selection__feature_selection_model'] = ['SelectFromModel', 'GenericUnivariateSelect', 'KeepAll'] #, 'RandomizedSparse', 'RFECV']

        return gs_params


    def train(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False, write_gs_param_results_to_file=True):

        if write_gs_param_results_to_file:
            gs_param_file_name = 'most_recent_pipeline_grid_search_result.csv'
            try:
                os.remove(gs_param_file_name)
            except:
                pass

        # split out out output column so we have a proper X, y dataset
        X, y = utils.split_output(raw_training_data, self.output_column)

        self.gs_params = self._construct_pipeline_search_params(optimize_entire_pipeline=optimize_entire_pipeline, optimize_final_model=optimize_final_model)

        ppl = self._construct_pipeline(user_input_func, optimize_final_model)

        if self.type_of_algo == 'classifier':
            # scoring = 'roc_auc'
            scoring = make_scorer(brier_score_loss, greater_is_better=True)

        else:
            scoring = None
            # scoring = 'mean_squared_error'
            # scoring = utils.rmse_scoring

        # We will be performing GridSearchCV every time, even if the space we are searching over is null
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
            scoring=scoring
        )

        gs.fit(X, y)

        self.trained_pipeline = gs.best_estimator_

        # write the results for each param combo to file for user analytics.
        if write_gs_param_results_to_file:
            utils.write_gs_param_results_to_file(gs, gs_param_file_name)

        return self

    def _get_estimator_names(self, ml_for_analytics=False):
        if self.type_of_algo == 'regressor':
            base_estimators = ['LinearRegression', 'RandomForestRegressor', 'Ridge', 'XGBRegressor']
            if ml_for_analytics:
                return base_estimators
            else:
                # base_estimators.append()
                return base_estimators

        elif self.type_of_algo == 'classifier':
            base_estimators = ['LogisticRegression', 'RandomForestClassifier', 'RidgeClassifier', 'XGBClassifier']
            if ml_for_analytics:
                return base_estimators
            else:
                # base_estimators.append()
                return base_estimators

        else:
            raise('TypeError: type_of_algo must be either "classifier" or "regressor".')


    def ml_for_analytics(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False, write_gs_param_results_to_file=True, perform_feature_selection=True):

        if write_gs_param_results_to_file:
            gs_param_file_name = 'most_recent_pipeline_grid_search_result.csv'
            try:
                os.remove(gs_param_file_name)
            except:
                pass

        # split out out output column so we have a proper X, y dataset
        X, y = utils.split_output(raw_training_data, self.output_column)

        if self.type_of_algo == 'classifier':
            try:
                y_ints = []
                for val in y:
                    y_ints.append(int(val))
                y = y_ints
            except:
                pass

        else:
            indices_to_delete = []
            y_floats = []
            bad_vals = []
            for idx, val in enumerate(y):
                try:
                    float_val = float(val)
                    y_floats.append(float_val)
                except:
                    indices_to_delete.append(idx)
                    bad_vals.append(val)

            y = y_floats

            if len(indices_to_delete) > 0:
                print('The y values given included some bad values')
                print('The rows at these indices have been deleted because their y value could not be turned into a float')
                print(indices_to_delete)
                print('These were the bad values')
                print(bad_vals)
                indices_to_delete = set(indices_to_delete)
                X = [row for idx, row in enumerate(X) if idx not in indices_to_delete]
                print(len(X))
                print(len(y))


        ppl = self._construct_pipeline(user_input_func, optimize_final_model=optimize_final_model, ml_for_analytics=True, perform_feature_selection=perform_feature_selection)

        estimator_names = self._get_estimator_names(ml_for_analytics=True)

        if self.type_of_algo == 'classifier':
            # scoring = 'roc_auc'
            scoring = make_scorer(brier_score_loss, greater_is_better=True)
            self._scorer = scoring
        else:
            scoring = None
            # scoring = 'mean_squared_error'
            # scoring = utils.rmse_scoring
            self._scorer = scoring


        for model_name in estimator_names:

            self.grid_search_params = self._construct_pipeline_search_params(optimize_entire_pipeline=optimize_entire_pipeline, optimize_final_model=optimize_final_model, ml_for_analytics=True, perform_feature_selection=perform_feature_selection)

            self.grid_search_params['final_model__model_name'] = [model_name]

            gs = GridSearchCV(
                # Fit on the pipeline.
                ppl,
                param_grid=self.grid_search_params,
                # Train across all cores.
                n_jobs=-1,
                # Be verbose (lots of printing).
                # verbose=10,
                # Print warnings when we fail to fit a given combination of parameters, but do not raise an error.
                error_score=10,
                # TODO(PRESTON): change scoring to be RMSE by default
                scoring=scoring
            )

            gs.fit(X, y)
            self.trained_pipeline = gs.best_estimator_

            if model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
                self._print_ml_analytics_results_regression()
            elif model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
                self._print_ml_analytics_results_random_forest()

            # write the results for each param combo to file for user analytics.
            if write_gs_param_results_to_file:
                utils.write_gs_param_results_to_file(gs, gs_param_file_name)

    def _print_ml_analytics_results_random_forest(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)

        if self.trained_pipeline.named_steps.get('feature_selection', False):

            selected_indices = self.trained_pipeline.named_steps['feature_selection'].support_mask
            feature_names_before_selection = self.trained_pipeline.named_steps['dv'].get_feature_names()
            trained_feature_names = [name for idx, name in enumerate(feature_names_before_selection) if selected_indices[idx]]

        else:
            trained_feature_names = self.trained_pipeline.named_steps['dv'].get_feature_names()

        trained_feature_importances = self.trained_pipeline.named_steps['final_model'].model.feature_importances_

        feature_infos = zip(trained_feature_names, trained_feature_importances)

        sorted_feature_infos = sorted(feature_infos, key=lambda x: x[1])

        print('Here are the feature_importances from the tree-based model:')
        print('The printed list will only contain at most the top 50 features.')
        for feature in sorted_feature_infos[:50]:
            print(feature[0] + ': ' + str(round(feature[1], 4)))


    def _print_ml_analytics_results_regression(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)

        if self.trained_pipeline.named_steps.get('feature_selection', False):

            selected_indices = self.trained_pipeline.named_steps['feature_selection'].support_mask
            feature_names_before_selection = self.trained_pipeline.named_steps['dv'].get_feature_names()
            trained_feature_names = [name for idx, name in enumerate(feature_names_before_selection) if selected_indices[idx]]

        else:
            trained_feature_names = self.trained_pipeline.named_steps['dv'].get_feature_names()

        trained_coefficients = self.trained_pipeline.named_steps['final_model'].model.coef_[0]

        feature_ranges = self.trained_pipeline.named_steps['final_model'].feature_ranges

        # TODO(PRESTON): readability. Can probably do this in a single zip statement.
        feature_summary = []
        for col_idx, feature_name in enumerate(trained_feature_names):

            potential_impact = feature_ranges[col_idx] * trained_coefficients[col_idx]
            summary_tuple = (feature_name, trained_coefficients[col_idx], potential_impact)
            feature_summary.append(summary_tuple)

        sorted_feature_summary = sorted(feature_summary, key=lambda x: abs(x[2]))

        print('The following is a list of feature names and their coefficients. This is followed by calculating a reasonable range for each feature, and multiplying by that feature\'s coefficient, to get an idea of the scale of the possible impact from this feature.')
        print('This printed list will contain at most the top 50 features.')
        for summary in sorted_feature_summary[:50]:
            print(summary[0] + ': ' + str(round(summary[1], 4)))
            print('The potential impact of this feature is: ' + str(round(summary[2], 4)))


    def print_training_summary(self):
        pass
        # Print some nice summary output of all the training we did.
        # maybe allow the user to pass in a flag to write info to a file


    def predict(self, prediction_data):

        return self.trained_pipeline.predict(prediction_data)

    def predict_proba(self, prediction_data):

        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test):
        if self._scorer is not None:
            return self._scorer(self.trained_pipeline, X_test, y_test)
        else:
            return self.trained_pipeline.score(X_test, y_test)

