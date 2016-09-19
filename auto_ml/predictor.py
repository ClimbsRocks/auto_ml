import math
import os
import sys
import warnings

try:
    import cPickle as pickle
except:
    import pickle

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, brier_score_loss, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from auto_ml import utils
from auto_ml import date_feature_engineering

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Predictor(object):


    def __init__(self, type_of_estimator, column_descriptions, verbose=True):
        if type_of_estimator.lower() in ['regressor','regression', 'regressions', 'regressors', 'number', 'numeric', 'continuous']:
            self.type_of_estimator = 'regressor'
        elif type_of_estimator.lower() in ['classifier', 'classification', 'categorizer', 'categorization', 'categories', 'labels', 'labeled', 'label']:
            self.type_of_estimator = 'classifier'
        else:
            print('Invalid value for "type_of_estimator". Please pass in either "regressor" or "classifier". You passed in: ' + type_of_estimator)
            raise ValueError('Invalid value for "type_of_estimator". Please pass in either "regressor" or "classifier". You passed in: ' + type_of_estimator)
        self.column_descriptions = column_descriptions
        self.verbose = verbose
        self.trained_pipeline = None
        self._scorer = None
        self.date_cols = []
        # Later on, if this is a regression problem, we will probably take the natural log of our y values for training, but we will still want to return the predictions in their normal scale (not the natural log values)
        self.took_log_of_y = False
        self.take_log_of_y = False

        self._validate_input_col_descriptions(column_descriptions)

        self.grid_search_pipelines = []


    def _validate_input_col_descriptions(self, column_descriptions):
        found_output_column = False
        self.subpredictors = []
        subpredictor_vals = set(['regressor', 'classifier'])
        expected_vals = set(['categorical', 'ignore'])

        for key, value in column_descriptions.items():
            value = value.lower()
            column_descriptions[key] = value
            if value == 'output':
                self.output_column = key
                found_output_column = True
            elif value == 'date':
                self.date_cols.append(key)
            elif value in expected_vals:
                pass
            elif value in subpredictor_vals:
                self.subpredictors.append(key)
            else:
                raise ValueError('We are not sure how to process this column of data: ' + str(value) + '. Please pass in "output", "categorical", "date", or for more advances subpredictor ensembling, "regressor" or "classifier"')
        if found_output_column is False:
            print('Here is the column_descriptions that was passed in:')
            print(column_descriptions)
            raise ValueError('In your column_descriptions, please make sure exactly one column has the value "output", which is the value we will be training models to predict.')


    def _construct_pipeline(self, user_input_func=None, model_name='LogisticRegression', perform_feature_selection=True, impute_missing_values=True, ml_for_analytics=True, perform_feature_scaling=True):

        pipeline_list = []

        if len(self.subpredictors) > 0:
            pipeline_list.append(('subpredictors', utils.AddSubpredictorPredictions(trained_subpredictors=self.subpredictors)))
        if user_input_func is not None:
            pipeline_list.append(('user_func', FunctionTransformer(func=user_input_func, pass_y=False, validate=False) ))

        if len(self.date_cols) > 0:
            pipeline_list.append(('date_feature_engineering', date_feature_engineering.FeatureEngineer(date_cols=self.date_cols)))

        # These parts will be included no matter what.
        pipeline_list.append(('basic_transform', utils.BasicDataCleaning(column_descriptions=self.column_descriptions)))

        if perform_feature_scaling:
            pipeline_list.append(('scaler', utils.CustomSparseScaler(self.column_descriptions)))

        pipeline_list.append(('dv', DictVectorizer(sparse=True, sort=False)))

        if perform_feature_selection:
            # pipeline_list.append(('pca', TruncatedSVD()))
            pipeline_list.append(('feature_selection', utils.FeatureSelectionTransformer(type_of_estimator=self.type_of_estimator, feature_selection_model='SelectFromModel') ))

        if self.add_cluster_prediction or self.compute_power >=7:
            pipeline_list.append(('add_cluster_prediction', utils.AddPredictedFeature(model_name='MiniBatchKMeans', type_of_estimator=self.type_of_estimator, include_original_X=True)))

        pipeline_list.append(('final_model', utils.FinalModelATC(model_name=model_name, perform_grid_search_on_model=self.optimize_final_model, type_of_estimator=self.type_of_estimator, ml_for_analytics=ml_for_analytics)))

        constructed_pipeline = Pipeline(pipeline_list)
        return constructed_pipeline


    def _construct_pipeline_search_params(self, user_defined_model_names=None):

        gs_params = {}

        if self.optimize_final_model or self.compute_power >= 5:
            gs_params['final_model__perform_grid_search_on_model'] = [True, False]

        if self.compute_power >= 6:
            gs_params['scaler__truncate_large_values'] = [True, False]

        if user_defined_model_names:
            model_names = user_defined_model_names
        else:
            model_names = self._get_estimator_names()
        gs_params['final_model__model_name'] = model_names

        # Only optimize our feature selection methods this deeply if the user really, really wants to.
        if self.compute_power >= 10:
            # We've also built in support for 'RandomizedSparse' feature selection methods, but they don't always support sparse matrices, so we are ignoring them by default.
            gs_params['feature_selection__feature_selection_model'] = ['SelectFromModel', 'GenericUnivariateSelect', 'KeepAll', 'RFECV'] # , 'RandomizedSparse'

        return gs_params


    def _get_estimator_names(self):
        if self.type_of_estimator == 'regressor':
            # base_estimators = ['LinearRegression']
            base_estimators = ['Ridge', 'XGBRegressor']
            if self.compute_power < 7:
                return base_estimators
            else:
                base_estimators.append('RANSACRegressor')
                base_estimators.append('RandomForestRegressor')
                base_estimators.append('LinearRegression')
                base_estimators.append('AdaBoostRegressor')
                base_estimators.append('ExtraTreesRegressor')
                return base_estimators

        elif self.type_of_estimator == 'classifier':
            base_estimators = ['RidgeClassifier', 'XGBClassifier']
            if self.compute_power < 7:
                return base_estimators
            else:
                base_estimators.append('LogisticRegression')
                base_estimators.append('RandomForestClassifier')
                return base_estimators

        else:
            raise('TypeError: type_of_estimator must be either "classifier" or "regressor".')

    def _prepare_for_training(self, raw_training_data):
        if self.write_gs_param_results_to_file:
            self.gs_param_file_name = 'most_recent_pipeline_grid_search_result.csv'
            try:
                os.remove(self.gs_param_file_name)
            except:
                pass

        # split out out output column so we have a proper X, y dataset
        X, y = utils.split_output(raw_training_data, self.output_column)

        # TODO: modularize into clean_y_vals function
        if self.type_of_estimator == 'classifier':
            try:
                y_ints = []
                for val in y:
                    y_ints.append(int(val))
                y = y_ints
            except:
                pass
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
                print('The y values given included some bad values that the machine learning algorithms will not be able to train on.')
                print('The rows at these indices have been deleted because their y value could not be turned into a float:')
                print(indices_to_delete)
                print('These were the bad values')
                print(bad_vals)
                indices_to_delete = set(indices_to_delete)
                X = [row for idx, row in enumerate(X) if idx not in indices_to_delete]

        return X, y

    def _make_sub_column_descriptions(self, column_descriptions, sub_name):
        # TODO: make this work for multiple subpredictors. right now it will grab all 'regressor' or 'classifier' values at once, instead of only grabbing on.
        subpredictor_types = set(['classifier', 'regressor'])
        dup_descs = {}
        # subpredictor_type_of_estimator = 'regressor'

        for key, val in column_descriptions.items():

            # Obviously, skip the parent ensembler's output column
            if val != 'output':
                if key == sub_name:
                    # set the right sub_name to be output for this subpredictor
                    dup_descs[key] = 'output'
                    sub_type_of_estimator = val
                elif val in subpredictor_types:
                    dup_descs[key] = 'ignore'

                # Include all other subpredictor names, so that we know to ignore them later on inside this subpredictor
                else:
                    dup_descs[key] = val

        return dup_descs, sub_type_of_estimator

    def make_sub_x_and_y_test(self, X_test, sub_name):
        vals_to_ignore = set([None, float('nan'), float('Inf'), 'ignore'])
        clean_X_test = []
        clean_y = []
        for row in X_test:
            y_val = row.pop(sub_name, None)
            if y_val not in vals_to_ignore:
                clean_X_test.append(row)
                clean_y.append(y_val)
        return clean_X_test, clean_y


    def _train_subpredictor(self, sub_name, X_subpredictors, sub_idx, sub_model_names=None, sub_ml_analytics=False, sub_compute_power=5):

        sub_column_descriptions, sub_type_of_estimator = self._make_sub_column_descriptions(self.column_descriptions, sub_name)
        if sub_model_names is None and sub_type_of_estimator == 'classifier':
            # sub_model_names = ['XGBClassifier']
            sub_model_names = ['GradientBoostingClassifier']
        elif sub_model_names is None and sub_type_of_estimator == 'regressor':
            # sub_model_names = ['XGBRegressor']
            sub_model_names = ['GradientBoostingRegressor']

        ml_predictor = Predictor(type_of_estimator=sub_type_of_estimator, column_descriptions=sub_column_descriptions)
        # TODO: grab proper y_test values for this particular subpredictor
        if self.X_test is not None and self.y_test is not None:
            sub_X_test, sub_y_test = self.make_sub_x_and_y_test(self.X_test, sub_name)
        else:
            sub_X_test = None
            sub_y_test = None

        # NOTE that we will be mutating the input X here by stripping off the y values.
        ml_predictor.train(raw_training_data=X_subpredictors
            , perform_feature_selection=True
            , X_test=sub_X_test
            , y_test=sub_y_test
            , ml_for_analytics=sub_ml_analytics
            , compute_power=sub_compute_power
            , take_log_of_y=False
            , add_cluster_prediction=True
            , model_names=sub_model_names
            , write_gs_param_results_to_file=False
            , optimize_final_model=self.optimize_final_model
        )
        self.subpredictors[sub_idx] = ml_predictor



    def train(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=False, write_gs_param_results_to_file=True, perform_feature_selection=True, verbose=True, X_test=None, y_test=None, print_training_summary_to_viewer=True, ml_for_analytics=True, only_analytics=False, compute_power=3, take_log_of_y=True, model_names=None, add_cluster_prediction=False, num_weak_estimators=0):

        self.user_input_func = user_input_func
        self.optimize_final_model = optimize_final_model
        self.optimize_entire_pipeline = optimize_entire_pipeline
        self.perform_feature_selection = perform_feature_selection
        self.write_gs_param_results_to_file = write_gs_param_results_to_file
        self.compute_power = compute_power
        self.ml_for_analytics = ml_for_analytics
        self.only_analytics = only_analytics
        self.X_test = X_test
        self.y_test = y_test
        self.print_training_summary_to_viewer = print_training_summary_to_viewer
        if self.type_of_estimator == 'regressor':
            self.take_log_of_y = take_log_of_y
        self.add_cluster_prediction = add_cluster_prediction
        self.num_weak_estimators = num_weak_estimators
        self.model_names = model_names

        # Put in place the markers that will tell us later on to train up a subpredictor for this problem
        if self.num_weak_estimators > 0:
            for idx in range(self.num_weak_estimators):
                self.column_descriptions['weak_estimator_' + str(idx)] = self.type_of_estimator
                self.subpredictors.append('weak_estimator_' + str(idx))
            self.weak_estimator_store = {
                'regressor': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'
                # , 'LassoLars'
                # , 'OrthogonalMatchingPursuit'
                # , 'BayesianRidge'
                # , 'ARDRegression'
                , 'SGDRegressor'
                , 'PassiveAggressiveRegressor'
                ],
                'classifier': ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier']
            }

        if verbose:
            print('Welcome to auto_ml! We\'re about to go through and make sense of your data using machine learning')

        X, y = self._prepare_for_training(raw_training_data)

        # print('To compare to our current predictions, we are only training our master ensemble on two thirds of the data')
        # X_ensemble, X_subpredictors, y_ensemble, y_subpredictors = train_test_split(X, y, test_size=0.33)
        # X = X_ensemble
        # y = y_ensemble
        # Once we have removed the applicable y-values, look into creating any subpredictors we might need
        if len(self.subpredictors) > 0:
            print('We are going to be training up several subpredictors before training up our final ensembled predictor')

            # Split out a percentage of our dataset to use ONLY for training subpredictors.
            # We will batch train the subpredictors once at the start, before GridSearchCV, to avoide computationally expensive repetitive training of these same models.
            # However, this means we'll have to train our subpredictors on a different dataset than we train our larger ensemble predictor on.
            # X_ensemble is the X data we'll be using to train our ensemble (the bulk of our data), and y_ensemble is, of course, the relevant y data for training our ensemble.
            # X_subpredictors is the smaller subset of data we'll be using to train our subpredictors on. y_subpredictors doesn't make any sense- it's the y values for our ensemble, but split to mirror the data we're using to train our subpredictors. Thus, we'll ignore it.
            X_ensemble, X_subpredictors, y_ensemble, y_subpredictors = train_test_split(X, y, test_size=0.33)
            X = X_ensemble
            y = y_ensemble

            for sub_idx, sub_name in enumerate(self.subpredictors):
                print('Now training a subpredictor for ' + sub_name)

                # Print out analytics for the subpredictor if we are printing them for the parent.
                sub_ml_analytics = self.ml_for_analytics
                sub_compute_power = self.compute_power

                sub_model_names = None
                if sub_name[:14] == 'weak_estimator':
                    weak_estimator_list = self.weak_estimator_store[self.type_of_estimator]
                    # Cycle through the weak estimator names in order.
                    name_index = sub_idx % len(weak_estimator_list)
                    weak_estimator_name = weak_estimator_list[name_index]
                    sub_model_names = [weak_estimator_name]

                    # If this is a weak predictor, ignore the analytics.
                    sub_ml_analytics = False
                    sub_compute_power = 1
                    # Now we have to give it the data to train on!
                    for row_idx, row in enumerate(X_subpredictors):
                        row[sub_name] = y_subpredictors[row_idx]

                self._train_subpredictor(sub_name, X_subpredictors, sub_idx, sub_model_names=sub_model_names, sub_ml_analytics=sub_ml_analytics, sub_compute_power=sub_compute_power)

        if self.take_log_of_y:
            y = [math.log(val) for val in y]
            self.took_log_of_y = True

        if verbose:
            print('Successfully performed basic preparations and y-value cleaning')

        ppl = self._construct_pipeline(user_input_func, perform_feature_selection=perform_feature_selection, ml_for_analytics=self.ml_for_analytics)

        if verbose:
            print('Successfully constructed the pipeline')

        if model_names:
            estimator_names = model_names
        else:
            estimator_names = self._get_estimator_names()

        if self.type_of_estimator == 'classifier':
            # scoring = make_scorer(brier_score_loss, greater_is_better=True)
            scoring = utils.brier_score_loss_wrapper
            self._scorer = scoring
        else:
            scoring = utils.rmse_scoring
            self._scorer = scoring

        if verbose:
            print('Created estimator_names and scoring')


        self.perform_grid_search_by_model_names(estimator_names, ppl, scoring, X, y)

        # Once we have trained all the pipelines, select the best one based on it's performance on (top priority first):
        # 1. Holdout data
        # 2. CV data

        # First, sort all of the tuples that hold our scores in their first position(s), and our actual trained pipeline in their final position
        # Since a more positive score is better, we want to make sure that the first item in our sorted list is the highest score, thus, reverse=True
        sorted_gs_pipeline_results = sorted(self.grid_search_pipelines, key=lambda x: x[0], reverse=True)

        # Next, grab the thing at position 0 in our sorted list, which is itself a list of the scores(s), and the pipeline itself
        best_result_list = sorted_gs_pipeline_results[0]
        # Our best grid search result is the thing at the end of that list.
        best_trained_gs = best_result_list[-1]
        # And the pipeline is the best estimator within that grid search object.
        self.trained_pipeline = best_trained_gs.best_estimator_

        del self.X_test
        del self.grid_search_pipelines

    def perform_grid_search_by_model_names(self, estimator_names, ppl, scoring, X, y):

        ppl = self._construct_pipeline(self.user_input_func, perform_feature_selection=self.perform_feature_selection, ml_for_analytics=self.ml_for_analytics)

        for model_name in estimator_names:

            self.grid_search_params = self._construct_pipeline_search_params(user_defined_model_names=estimator_names)

            self.grid_search_params['final_model__model_name'] = [model_name]
            if self.verbose:
                grid_search_verbose = 5
            else:
                grid_search_verbose = 0

            gs = GridSearchCV(
                # Fit on the pipeline.
                ppl,
                cv=2,
                param_grid=self.grid_search_params,
                # Train across all cores.
                n_jobs=-1,
                # Be verbose (lots of printing).
                verbose=grid_search_verbose,
                # Print warnings when we fail to fit a given combination of parameters, but do not raise an error.
                # Set the score on this partition to some very negative number, so that we do not choose this estimator.
                error_score=-1000000000,
                # TODO(PRESTON): change scoring to be RMSE by default
                scoring=scoring
                # ,pre_dispatch='1*n_jobs'
            )

            if self.verbose:
                print('\n\n********************************************************************************************')
                print('About to fit the GridSearchCV on the pipeline for the model ' + model_name + ' to predict ' + self.output_column)

            gs.fit(X, y)
            self.trained_pipeline = gs.best_estimator_

            if self.ml_for_analytics and model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
                self._print_ml_analytics_results_regression()
            elif self.ml_for_analytics and model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
                self._print_ml_analytics_results_random_forest()

            # write the results for each param combo to file for user analytics.
            if self.write_gs_param_results_to_file:
                utils.write_gs_param_results_to_file(gs, self.gs_param_file_name)

            # We will save the info for this pipeline grid search, along with it's scores on the CV data, and the holdout data
            pipeline_results = []

            if self.X_test and self.y_test:
                print('The results from the X_test and y_test data passed into ml_for_analytics (which were not used for training- true holdout data) are:')
                holdout_data_score = self.score(self.X_test, self.y_test)
                print(holdout_data_score)

                pipeline_results.append(holdout_data_score)

            if self.print_training_summary_to_viewer:
                self.print_training_summary(gs)

            pipeline_results.append(gs.best_score_)
            pipeline_results.append(gs)
            self.grid_search_pipelines.append(pipeline_results)


    def _get_xgb_feat_importances(self, clf):

        try:
            # Handles case when clf has been created by calling
            # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
            fscore = clf.booster().get_fscore()
        except:
            # Handles case when clf has been created by calling xgb.train.
            # Thus, clf is an instance of xgb.Booster.
            fscore = clf.get_fscore()

        trained_feature_names = self._get_trained_feature_names()

        feat_importances = []

        # Somewhat annoying. XGBoost only returns importances for the features it finds useful.
        # So we have to go in, get the index of the feature from the "feature name" by removing the f before the feature name, and grabbing the rest of that string, which is actually the index of that feature name.
        fscore_list = [[int(k[1:]), v] for k, v in fscore.items()]


        feature_infos = []
        sum_of_all_feature_importances = 0.0

        for idx_and_result in fscore_list:
            idx = idx_and_result[0]
            # Use the index that we grabbed above to find the human-readable feature name
            feature_name = trained_feature_names[idx]
            feat_importance = idx_and_result[1]

            # If we sum up all the feature importances and then divide by that sum, we will be able to have each feature importance as it's relative feature imoprtance, and the sum of all of them will sum up to 1, just as it is in scikit-learn.
            sum_of_all_feature_importances += feat_importance
            feature_infos.append([feature_name, feat_importance])

        sorted_feature_infos = sorted(feature_infos, key=lambda x: x[1])

        print('Here are the feature_importances from the tree-based model:')
        print('The printed list will only contain at most the top 50 features.')
        for feature in sorted_feature_infos[-50:]:
            print(feature[0] + ': ' + str(round(feature[1] / sum_of_all_feature_importances, 4)))

    def _print_ml_analytics_results_random_forest(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)

        # XGB's Classifier has a proper .feature_importances_ property, while the XGBRegressor does not.
        if self.trained_pipeline.named_steps['final_model'].model_name in ['XGBRegressor', 'XGBClassifier']:
            self._get_xgb_feat_importances(self.trained_pipeline.named_steps['final_model'].model)

        else:
            trained_feature_names = self._get_trained_feature_names()

            trained_feature_importances = self.trained_pipeline.named_steps['final_model'].model.feature_importances_

            feature_infos = zip(trained_feature_names, trained_feature_importances)

            sorted_feature_infos = sorted(feature_infos, key=lambda x: x[1])

            print('Here are the feature_importances from the tree-based model:')
            print('The printed list will only contain at most the top 50 features.')
            for feature in sorted_feature_infos[-50:]:
                print(feature[0] + ': ' + str(round(feature[1], 4)))

    def _get_trained_feature_names(self):
        if self.trained_pipeline.named_steps.get('feature_selection', False):

            selected_indices = self.trained_pipeline.named_steps['feature_selection'].support_mask
            feature_names_before_selection = self.trained_pipeline.named_steps['dv'].get_feature_names()
            trained_feature_names = [name for idx, name in enumerate(feature_names_before_selection) if selected_indices[idx]]

        else:
            trained_feature_names = self.trained_pipeline.named_steps['dv'].get_feature_names()

        # Every transformer that adds a feature after DictVectorizer must start with "add", and must have a .added_feature_names_ attribute.
        # Get all the feature names that were added by ensembled predictors of any type
        for step in self.trained_pipeline.named_steps:
            if step[:3] == 'add':
                trained_feature_names = trained_feature_names + self.trained_pipeline.named_steps[step].added_feature_names_

        return trained_feature_names


    def _print_ml_analytics_results_regression(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)

        trained_feature_names = self._get_trained_feature_names()

        if self.type_of_estimator == 'classifier':
            trained_coefficients = self.trained_pipeline.named_steps['final_model'].model.coef_[0]
        else:
            trained_coefficients = self.trained_pipeline.named_steps['final_model'].model.coef_

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
        for summary in sorted_feature_summary[-50:]:
            print(summary[0] + ': ' + str(round(summary[1], 4)))
            print('The potential impact of this feature is: ' + str(round(summary[2], 4)))


    def print_training_summary(self, gs):
        print('The best CV score from GridSearchCV (by default averaging across k-fold CV) for ' + self.output_column + ' is:')
        if self.took_log_of_y:
            print('    Note that this score is calculated using the natural logs of the y values.')
        print(gs.best_score_)
        print('The best params were')
        print(gs.best_params_)
        # Print some nice summary output of all the training we did.
        # maybe allow the user to pass in a flag to write info to a file


    def predict(self, prediction_data):

        # if self.model_name[:13] == 'GradientBoosting' and scipy.issparse(prediction_data):
        #     prediction_data


        # TODO(PRESTON): investigate if we need to handle input of a single dictionary differently than a list of dictionaries.
        predicted_vals = self.trained_pipeline.predict(prediction_data)
        if self.took_log_of_y:
            for idx, val in predicted_vals:
                predicted_vals[idx] = math.exp(val)
        return predicted_vals

    def predict_proba(self, prediction_data):

        # TODO(PRESTON): investigate if we need to handle input of a single dictionary differently than a list of dictionaries.
        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test):
        if self._scorer is not None:
            try:
                if self.type_of_estimator == 'regressor':
                    return self._scorer(self.trained_pipeline, X_test, y_test, self.took_log_of_y)
                elif self.type_of_estimator == 'classifier':
                    return self._scorer(self.trained_pipeline, X_test, y_test)

            except:

                return self._scorer(self.trained_pipeline, X_test, y_test)
        else:
            return self.trained_pipeline.score(X_test, y_test)

    def save(self, file_name='auto_ml_saved_pipeline.pkl', verbose=True):
        with open(file_name, 'wb') as open_file_name:
            pickle.dump(self.trained_pipeline, open_file_name, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print('\n\nWe have saved the trained pipeline to a filed called "auto_ml_saved_pipeline.pkl"')
            print('It is saved in the directory: ')
            print(os.getcwd())
            print('To use it to get predictions, please follow the following flow (adjusting for your own uses as necessary:\n\n')
            print('`with open("auto_ml_saved_pipeline.pkl", "rb") as read_file:`')
            print('`    trained_ml_pipeline = pickle.load(read_file)`')
            print('`trained_ml_pipeline.predict(list_of_dicts_with_same_data_as_training_data)`\n\n')

            print('Note that this pickle file can only be loaded in an environment with the same modules installed, and running the same Python version.')
            print('This version of Python is:')
            print(sys.version_info)

            print('\n\nWhen passing in new data to get predictions on, columns that were not present (or were not found to be useful) in the training data will be silently ignored.')
            print('It is worthwhile to make sure that you feed in all the most useful data points though, to make sure you can get the highest quality predictions.')
            print('\nThese are the most important features that were fed into the model:')

            if self.ml_for_analytics and self.trained_pipeline.named_steps['final_model'].model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
                self._print_ml_analytics_results_regression()
            elif self.ml_for_analytics and self.trained_pipeline.named_steps['final_model'].model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
                self._print_ml_analytics_results_random_forest()

        return os.getcwd() + file_name


