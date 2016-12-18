import datetime
import math
import multiprocessing
import os
import random
import sys
import warnings

import dill

import pandas as pd
import pathos

# Ultimately, we (the authors of auto_ml) are responsible for building a project that's robust against warnings.
# The classes of warnings below are ones we've deemed acceptable. The user should be able to sit at a high level of abstraction, and not be bothered with the internals of how we're handing these things.
# Ignore all warnings that are UserWarnings or DeprecationWarnings. We'll fix these ourselves as necessary.
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# from sklearn.model_selection import
import scipy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, brier_score_loss, make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# This is ugly, but allows auto_ml to work whether it's installed using pip, or the whole project is installed using git clone https://github.com/ClimbsRocks/auto_ml
try:
# from auto_ml import date_feature_engineering
    from auto_ml import DataFrameVectorizer
    from auto_ml import utils
    from auto_ml import utils_data_cleaning
    from auto_ml import utils_ensemble
    from auto_ml import utils_feature_selection
    from auto_ml import utils_model_training
    from auto_ml import utils_models
    from auto_ml import utils_scaling
    from auto_ml import utils_scoring
except ImportError:
    from .. auto_ml import date_feature_engineering
    from .. auto_ml import DataFrameVectorizer
    from .. auto_ml import utils
    from .. auto_ml import utils_data_cleaning
    from .. auto_ml import utils_ensemble
    from .. auto_ml import utils_feature_selection
    from .. auto_ml import utils_model_training
    from .. auto_ml import utils_models
    from .. auto_ml import utils_scaling
    from .. auto_ml import utils_scoring


# XGBoost can be a pain to install. It's also a super powerful and effective package.
# So we'll make it optional here. If a user wants to install XGBoost themselves, we fully support XGBoost!
# But, if they just want to get running out of the gate, without dealing with any installation other than what's done for them automatically, we won't force them to go through that.
# The same logic will apply to deep learning with Keras and TensorFlow
global xgb_installed
xgb_installed = False
try:
    import xgboost as xgb
    xgb_installed = True
except NameError:
    pass
except ImportError:
    pass



class Predictor(object):


    def __init__(self, type_of_estimator, column_descriptions, verbose=True, name=None):
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
        # Later on, if this is a regression problem, we will possibly take the natural log of our y values for training, but we will still want to return the predictions in their normal scale (not the natural log values)
        self.took_log_of_y = False
        self.take_log_of_y = False

        self._validate_input_col_descriptions()

        self.grid_search_pipelines = []

        self.is_ensemble = False
        # Set this here for later use if this is an ensembled subpredictor
        self.name = name


    def _validate_input_col_descriptions(self):
        found_output_column = False
        self.cols_to_ignore = []
        expected_vals = set(['categorical', 'text', 'nlp'])

        for key, value in self.column_descriptions.items():
            value = value.lower()
            self.column_descriptions[key] = value
            if value == 'output':
                self.output_column = key
                found_output_column = True
            elif value == 'date':
                self.date_cols.append(key)
            elif value == 'ignore':
                self.cols_to_ignore.append(key)
            elif value in expected_vals:
                pass
            else:
                raise ValueError('We are not sure how to process this column of data: ' + str(value) + '. Please pass in "output", "categorical", "ignore", "nlp", or "date".')
        if found_output_column is False:
            print('Here is the column_descriptions that was passed in:')
            print(self.column_descriptions)
            raise ValueError('In your column_descriptions, please make sure exactly one column has the value "output", which is the value we will be training models to predict.')

        # We will be adding one new categorical variable for each date col
        # Be sure to add it here so the rest of the pipeline knows to handle it as a categorical column
        for date_col in self.date_cols:
            self.column_descriptions[date_col + '_day_part'] = 'categorical'


    # We use _construct_pipeline at both the start and end of our training.
    # At the start, it constructs the pipeline from scratch
    # At the end, it takes FeatureSelection out after we've used it to restrict DictVectorizer
    def _construct_pipeline(self, model_name='LogisticRegression', impute_missing_values=True, trained_pipeline=None):

        pipeline_list = []

        if self.ensembler is not None:
            if trained_pipeline is not None:
                pipeline_list.append(('add_ensemble_predictions', trained_pipeline.named_steps['add_ensemble_predictions']))
            else:
                pipeline_list.append(('add_ensemble_predictions', utils_ensemble.AddEnsembledPredictions(ensembler=self.ensembler, type_of_estimator=self.type_of_estimator)))

        if self.user_input_func is not None:
            if trained_pipeline is not None:
                pipeline_list.append(('user_func', trained_pipeline.named_steps['user_func']))
            else:
                pipeline_list.append(('user_func', FunctionTransformer(func=self.user_input_func, pass_y=False, validate=False)))

        # These parts will be included no matter what.
        if trained_pipeline is not None:
            pipeline_list.append(('basic_transform', trained_pipeline.named_steps['basic_transform']))
        else:
            pipeline_list.append(('basic_transform', utils_data_cleaning.BasicDataCleaning(column_descriptions=self.column_descriptions)))

        if self.perform_feature_scaling is True or (self.compute_power >= 7 and self.perform_feature_scaling is not False):
            if trained_pipeline is not None:
                pipeline_list.append(('scaler', trained_pipeline.named_steps['scaler']))
            else:
                pipeline_list.append(('scaler', utils_scaling.CustomSparseScaler(self.column_descriptions)))


        if trained_pipeline is not None:
            pipeline_list.append(('dv', trained_pipeline.named_steps['dv']))
        else:
            pipeline_list.append(('dv', DataFrameVectorizer.DataFrameVectorizer(sparse=True, sort=True, column_descriptions=self.column_descriptions)))


        if self.perform_feature_selection == True or (self.compute_power >= 9 and self.perform_feature_selection is None):
            if trained_pipeline is not None:
                # This is the step we are trying to remove from the trained_pipeline, since it has already been combined with dv using dv.restrict
                pass
            else:
                # pipeline_list.append(('pca', TruncatedSVD()))
                pipeline_list.append(('feature_selection', utils_feature_selection.FeatureSelectionTransformer(type_of_estimator=self.type_of_estimator, column_descriptions=self.column_descriptions, feature_selection_model='SelectFromModel') ))

        if trained_pipeline is not None:
            pipeline_list.append(('final_model', trained_pipeline.named_steps['final_model']))
        else:
            final_model = utils_models.get_model_from_name(model_name)
            pipeline_list.append(('final_model', utils_model_training.FinalModelATC(model=final_model, type_of_estimator=self.type_of_estimator, ml_for_analytics=self.ml_for_analytics, name=self.name, scoring_method=self._scorer)))
            # final_model = utils_models.get_model_from_name(model_name)
            # pipeline_list.append(('final_model', utils_model_training.FinalModelATC(model_name=model_name, type_of_estimator=self.type_of_estimator, ml_for_analytics=self.ml_for_analytics, name=self.name)))

        constructed_pipeline = Pipeline(pipeline_list)
        return constructed_pipeline


    def _construct_pipeline_search_params(self, user_defined_model_names=None):

        gs_params = {}

        if self.compute_power >= 6:
            gs_params['scaler__truncate_large_values'] = [True, False]

        if user_defined_model_names is not None:
            model_names = user_defined_model_names
        else:
            model_names = self._get_estimator_names()

        final_model__models = []
        for model_name in model_names:
            final_model__models.append(utils_models.get_model_from_name(model_name))


        # gs_params['final_model__model_name'] = model_names
        gs_params['final_model__model'] = final_model__models

        if self.compute_power >= 7:
            gs_params['scaler__perform_feature_scaling'] = [True, False]


        # Only optimize our feature selection methods this deeply if the user really, really wants to. This is super computationally expensive.
        if self.compute_power >= 10:
            # We've also built in support for 'RandomizedSparse' feature selection methods, but they don't always support sparse matrices, so we are ignoring them by default.
            gs_params['feature_selection__feature_selection_model'] = ['SelectFromModel', 'GenericUnivariateSelect', 'KeepAll', 'RFECV'] # , 'RandomizedSparse'

        return gs_params


    def _get_estimator_names(self):
        if self.type_of_estimator == 'regressor':
            if xgb_installed:
                base_estimators = ['XGBRegressor']
            else:
                base_estimators = ['GradientBoostingRegressor']
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
            if xgb_installed:
                base_estimators = ['XGBClassifier']
            else:
                base_estimators = ['GradientBoostingClassifier']
            if self.compute_power < 7:
                return base_estimators
            else:
                base_estimators.append('LogisticRegression')
                base_estimators.append('RandomForestClassifier')
                return base_estimators

        else:
            raise('TypeError: type_of_estimator must be either "classifier" or "regressor".')

    def _prepare_for_training(self, X_df):

        # If we're writing training results to file, create the new empty file name here
        if self.write_gs_param_results_to_file:
            self.gs_param_file_name = 'most_recent_pipeline_grid_search_result.csv'
            try:
                os.remove(self.gs_param_file_name)
            except:
                pass

        # Drop all rows that have an empty value for our output column
        # User logging so they can adjust if they pass in a bunch of bad values:
        bad_rows = X_df[pd.isnull(X_df[self.output_column])]
        if bad_rows.shape[0] > 0:
            print('We encountered a number of missing values for this output column')
            print('Specifically, here is the output column:')
            print(self.output_column)
            print('And here is the number of missing (nan, None, etc.) values for this column:')
            print(bad_rows.shape[0])
            print('We will remove these values, and continue with training on the cleaned dataset')
        X_df = X_df.dropna(subset=[self.output_column])


        # Remove the output column from the dataset, and store it into the y varaible
        y = list(X_df.pop(self.output_column))

        # If this is a classifier, try to turn all the y values into proper ints
        # Some classifiers play more nicely if you give them category labels as ints rather than strings, so we'll make our jobs easier here if we can.
        if self.type_of_estimator == 'classifier':
            # The entire column must be turned into floats. If any value fails, don't convert anything in the column to floats
            try:
                y_ints = []
                for val in y:
                    y_ints.append(int(val))
                y = y_ints
            except:
                pass
        else:
            # If this is a regressor, turn all the values into floats if possible, and remove this row if they cannot be turned into floats
            indices_to_delete = []
            y_floats = []
            bad_vals = []
            for idx, val in enumerate(y):
                try:
                    float_val = utils_data_cleaning.clean_val(val)
                    y_floats.append(float_val)
                except ValueError as err:
                    indices_to_delete.append(idx)
                    bad_vals.append(val)

            y = y_floats

            # Even more verbose logging here since these values are not just missing, they're strings for a regression problem
            if len(indices_to_delete) > 0:
                print('The y values given included some bad values that the machine learning algorithms will not be able to train on.')
                print('The rows at these indices have been deleted because their y value could not be turned into a float:')
                print(indices_to_delete)
                print('These were the bad values')
                print(bad_vals)
                # indices_to_delete = set(indices_to_delete)
                X_df = X_df.drop(X_df.index(indices_to_delete))
                # X_df = [row for idx, row in enumerate(X_df) if idx not in indices_to_delete]

        return X_df, y


    def _consolidate_feature_selection_steps(self, trained_pipeline):
        # First, restrict our DictVectorizer or DataFrameVectorizer
        # This goes through and has DV only output the items that have passed our support mask
        # This has a number of benefits: speeds up computation, reduces memory usage, and combines several transforms into a single, easy step
        # It also significantly reduces the size of dv.vocabulary_ which can get quite large

        dv = trained_pipeline.named_steps['dv']

        # If we do not have feature selection in the pipeline, just return the pipeline as is
        try:
            feature_selection = trained_pipeline.named_steps['feature_selection']
        except KeyError:
            return trained_pipeline
        feature_selection_mask = feature_selection.support_mask
        dv.restrict(feature_selection_mask)

        # We have overloaded our _construct_pipeline method to work both to create a new pipeline from scratch at the start of training, and to go through a trained pipeline in exactly the same order and steps to take a dedicated FeatureSelection model out of an already trained pipeline
        # In this way, we ensure that we only have to maintain a single centralized piece of logic for the correct order a pipeline should follow
        trained_pipeline_without_feature_selection = self._construct_pipeline(trained_pipeline=trained_pipeline)

        return trained_pipeline_without_feature_selection


    def train_ensemble(self, data, ensemble_training_list, X_test=None, y_test=None, ensemble_method='median', data_for_final_ensembling=None, find_best_method=False, verbose=2, include_original_X=True, scoring=None):

        self.scoring = scoring

        if y_test is not None:
            y_test = list(y_test)
        self.ensemble_predictors = []

        self.is_ensemble = True
        self.ml_for_analytics = True

        if self.type_of_estimator == 'classifier':
            scoring = utils_scoring.ClassificationScorer(self.scoring)
            self._scorer = scoring
        else:
            scoring = utils_scoring.RegressionScorer(self.scoring)
            self._scorer = scoring

        # Make it optional for the person to pass in type_of_estimator
        for training_params in ensemble_training_list:
            if training_params.get('type_of_estimator', None) is None:
                training_params['type_of_estimator'] = self.type_of_estimator
            if training_params.get('scoring', None) is None:
                training_params['scoring'] = self.scoring

        # ################################
        # If we're using machine learning to assemble our final ensemble, and we don't have data for it from the user, split out data here
        # ################################
        if ensemble_method in ['machine learning', 'ml', 'machine_learning'] and data_for_final_ensembling is None:



            # Just grab the last 20% of the dataset in the order it was given to us
            ensemble_idx = int(0.7 * len(data))
            data_for_final_ensembling = data[ensemble_idx:]
            data = data[:ensemble_idx]



        # ################################
        # Train one subpredictor, with logging, and only a subset of the data as chosen by the user
        # ################################
        def train_one_ensemble_subpredictor(training_params):

            print('\n\n************************')
            print('Training a new subpredictor for the ensemble!')
            name = training_params.pop('name')
            print('The name you gave for this estimator is:')
            print(name)
            print('\n\n')
            type_of_estimator = training_params.pop('type_of_estimator')
            col_descs = training_params.pop('column_descriptions')
            ml_predictor = Predictor(type_of_estimator, col_descs, name=name)

            this_rounds_data = data

            data_selection_func = training_params.pop('data_selection_func', None)
            if callable(data_selection_func):
                try:
                    # TODO: figure out how to see if this function is expecting to take in the name argument or not
                    this_rounds_data = data_selection_func(data, name)
                except TypeError:
                    this_rounds_data = data_selection_func(data)
            else:
                this_rounds_data = data


            training_params['raw_training_data'] = this_rounds_data

            ml_predictor.train(**training_params)

            return ml_predictor

            # self.ensemble_predictors.append(ml_predictor)


        # ################################
        # Train subpredictors in parallel
        # ################################
        pool = pathos.multiprocessing.ProcessPool()

        # Since we may have already closed the pool, try to restart it
        try:
            pool.restart()
        except AssertionError as e:
            pass
        trained_ensemble_predictors = pool.map(train_one_ensemble_subpredictor, ensemble_training_list, chunksize=100)
        trained_ensemble_predictors = list(trained_ensemble_predictors)

        # self.ensemble_predictors = [predictor.trained_pipeline for predictor in self.ensemble_predictors]
        # Once we have gotten all we need from the pool, close it so it's not taking up unnecessary memory
        pool.close()
        try:
            pool.join()
        except AssertionError:
            pass

        # ################################
        # Print scoring information for each trained subpredictor
        # ################################
        if X_test is not None and verbose >= 3:
            print('Scoring each of the trained subpredictors on the holdout data')

            def score_predictor(predictor, X_test, y_test):
                try:
                    print(predictor.name)
                except AttributeError:
                    pass


                predictor.score(X_test, y_test)

            pool = pathos.multiprocessing.ProcessPool()

            # Since we may have already closed the pool, try to restart it
            try:
                pool.restart()
            except AssertionError as e:
                pass
            pool.map(lambda predictor: score_predictor(predictor, X_test, y_test), trained_ensemble_predictors, chunksize=100)
            # pool.map(lambda predictor: score_predictor(predictor, X_test, y_test), self.ensemble_predictors, chunksize=100)
            # Once we have gotten all we need from the pool, close it so it's not taking up unnecessary memory
            pool.close()
            try:
                pool.join()
            except AssertionError:
                pass

        # Now that we've handled scoring (for which we will want the full ml_predictor objects with name and advanced_scoring, etc.), only grab the trained pipelines, which are much, much smaller objects
        for predictor in trained_ensemble_predictors:
            trained_pipeline = predictor.trained_pipeline
            # trained_pipeline.named_steps['final_model']['name'] = predictor.name
            self.ensemble_predictors.append(trained_pipeline)

        # ################################
        # Ensemble together our trained subpredictors, either using simple averaging, or training a new machine learning model to pick from amongst them
        # ################################

        if ensemble_method in ['machine learning', 'ml', 'machine_learning']:
            ensembler = utils_ensemble.Ensemble(ensemble_predictors=self.ensemble_predictors, type_of_estimator=self.type_of_estimator, method=ensemble_method)


            ml_predictor = Predictor(type_of_estimator=self.type_of_estimator, column_descriptions=self.column_descriptions, name=self.name)

            print('\n\n\n')
            print('We have trained up a bunch of individual estimators on this problem. Now it is time to train one final estimator that will ensemble all these predictions together for us')
            print('Using machine learning to ensemble together a bunch of trained estimators!')
            data_for_final_ensembling = data_for_final_ensembling.reset_index()
            if self.type_of_estimator == 'regressor':
                model_names = ['RandomForestRegressor', 'LinearRegression', 'ExtraTreesRegressor', 'Ridge', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'Lasso', 'ElasticNet', 'LassoLars', 'OrthogonalMatchingPursuit', 'BayesianRidge', 'SGDRegressor']
                if xgb_installed:
                    model_names.append('XGBRegressor')
            else:
                model_names = ['RandomForestClassifier', 'GradientBoostingClassifier', 'RidgeClassifier', 'LogisticRegression']
                if xgb_installed:
                    model_names.append('XGBClassifier')
                # model_names = ['LogisticRegression']
            ml_predictor.train(raw_training_data=data_for_final_ensembling, ensembler=ensembler, perform_feature_selection=False, model_names=model_names, _include_original_X=include_original_X, scoring=self.scoring)


            # predictions_on_ensemble_data = ensembler._get_all_predictions(data_for_final_ensembling)
            # data_for_final_ensembling = pd.concat([data_for_final_ensembling, predictions_on_ensemble_data], axis=1)

            self.trained_pipeline = ml_predictor.trained_pipeline

            if find_best_method == True:
                ensembler.find_best_ensemble_method(df=X_test, actuals=y_test)

        else:

            # Create an instance of an Ensemble object that will get predictions from all the trained subpredictors
            self.trained_pipeline = utils_ensemble.Ensemble(ensemble_predictors=self.ensemble_predictors, type_of_estimator=self.type_of_estimator, method=ensemble_method)

            if find_best_method == True:
                self.trained_pipeline.find_best_ensemble_method(df=X_test, actuals=y_test)




    def train(self, raw_training_data, user_input_func=None, optimize_entire_pipeline=False, optimize_final_model=None, write_gs_param_results_to_file=True, perform_feature_selection=None, verbose=True, X_test=None, y_test=None, print_training_summary_to_viewer=True, ml_for_analytics=True, only_analytics=False, compute_power=3, take_log_of_y=None, model_names=None, perform_feature_scaling=True, ensembler=None, calibrate_final_model=False, _include_original_X=False, _scorer=None, scoring=None, verify_features=False):

        self.user_input_func = user_input_func
        self.optimize_final_model = optimize_final_model
        self.optimize_entire_pipeline = optimize_entire_pipeline
        self.write_gs_param_results_to_file = write_gs_param_results_to_file
        self.compute_power = compute_power
        self.ml_for_analytics = ml_for_analytics
        self.only_analytics = only_analytics
        self.X_test = X_test
        self.y_test = y_test
        self.print_training_summary_to_viewer = print_training_summary_to_viewer
        if self.type_of_estimator == 'regressor':
            self.take_log_of_y = take_log_of_y
        self.model_names = model_names
        self.perform_feature_scaling = perform_feature_scaling
        self.ensembler = ensembler
        self.calibrate_final_model = calibrate_final_model
        self.scoring = scoring

        if verbose:
            print('Welcome to auto_ml! We\'re about to go through and make sense of your data using machine learning')

        # We accept input as either a DataFrame, or as a list of dictionaries. Internally, we use DataFrames. So if the user gave us a list, convert it to a DataFrame here.
        if isinstance(raw_training_data, list):
            X_df = pd.DataFrame(raw_training_data)
            del raw_training_data
        else:
            X_df = raw_training_data

        # Unless the user has told us to, don't perform feature selection unless we have a pretty decent amount of data
        if perform_feature_selection is None and self.compute_power < 9:
            if len(X_df.columns) < 50 or len(X_df) < 100000:
                perform_feature_selection = False
            else:
                perform_feature_selection = True

        self.perform_feature_selection = perform_feature_selection

        # To keep this as light in memory as possible, immediately remove any columns that the user has already told us should be ignored
        if len(self.cols_to_ignore) > 0:
            X_df = utils.safely_drop_columns(X_df, self.cols_to_ignore)

        X_df, y = self._prepare_for_training(X_df)
        self.X_df = X_df
        self.y = y


        if self.take_log_of_y:
            y = [math.log(val) for val in y]
            self.took_log_of_y = True

        if model_names is not None:
            estimator_names = model_names
        else:
            estimator_names = self._get_estimator_names()


        if self.type_of_estimator == 'classifier':
            if len(set(y)) > 2 and self.scoring is None:
                self.scoring = 'accuracy_score'
            else:
                scoring = utils_scoring.ClassificationScorer(self.scoring)
            self._scorer = scoring
        else:
            scoring = utils_scoring.RegressionScorer(self.scoring)
            self._scorer = scoring


        self.perform_grid_search_by_model_names(estimator_names, self._scorer, X_df, y)

        # If we ran GridSearchCV, we will have to pick the best model
        # If we did not, the best trained pipeline will already be saved in self.trained_pipeline
        if len(self.grid_search_pipelines) > 1:
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
            try:
                self.trained_pipeline = best_trained_gs.best_estimator_
            except:
                # We are also supporting the use case of the user training multiple model types, without optimzing any of them (so no GridSearchCV), but using X_test and y_test to determine the best model
                # In that case, the thing at the end of the best_result_list will be the trained pipeline we're interested in
                self.trained_pipeline = best_trained_gs

        # DictVectorizer will now perform DictVectorizer and FeatureSelection in a very efficient combination of the two steps.
        self.trained_pipeline = self._consolidate_feature_selection_steps(self.trained_pipeline)

        # verify_features is not enabled by default. It adds a significant amount to the file size of the saved pipelines.
        # If you are interested in submitting a PR to reduce the saved file size, there are definitely some optimizations you can make!
        if verify_features == True:
            # Save the features we used for training to our FinalModelATC instance.
            # This lets us provide useful information to the user when they call .predict(data, verbose=True)
            trained_feature_names = self._get_trained_feature_names()
            # print('trained_feature_names')
            # print(trained_feature_names)
            self.trained_pipeline.set_params(final_model__training_features=trained_feature_names)
            # We will need to know which columns are categorical/ignored/nlp when verifying features
            self.trained_pipeline.set_params(final_model__column_descriptions=self.column_descriptions)
            # self.trained_pipeline.named_steps['final_model']['training_features'] = trained_feature_names


        # Calibrate the probability predictions from our final model
        if self.calibrate_final_model is True and X_test is not None and y_test is not None:
            print('Now calibrating the final model so the probability predictions line up with the observed probabilities in the X_test and y_test datasets you passed in.')
            print('Note: the validation scores printed above are truly validation scores: they were scored before the model was calibrated to this data.')
            print('However, now that we are calibrating on the X_test and y_test data you gave us, it is no longer accurate to call this data validation data, since the model is being calibrated to it. As such, you must now report a validation score on a different dataset, or report the validation score used above before the model was calibrated to X_test and y_test. ')

            trained_model = self.trained_pipeline.named_steps['final_model'].model

            if len(X_test) < 1000:
                calibration_method = 'sigmoid'
            else:
                calibration_method = 'isotonic'

            calibrated_classifier = CalibratedClassifierCV(trained_model, method=calibration_method, cv='prefit')

            # We need to make sure X_test has been processed the exact same way y_test has.
            # So grab all the steps of the pipeline up to, but not including, the final_model
            # and run X_test through that transformer pipeline
            self.trained_transformer_pipeline = self.trained_pipeline
            transformer_pipeline = []
            for step in self.trained_transformer_pipeline.steps:
                if step[0] != 'final_model':
                    transformer_pipeline.append(step)

            self.trained_transformer_pipeline = Pipeline(transformer_pipeline)

            X_test = self.trained_transformer_pipeline.transform(X_test)

            try:
                calibrated_classifier = calibrated_classifier.fit(X_test, y_test)
            except TypeError as e:
                if scipy.sparse.issparse(X_test):
                    X_test = X_test.toarray()

                    calibrated_classifier = calibrated_classifier.fit(X_test, y_test)
                else:
                    raise(e)


            # Now insert the calibrated model back into our final_model step
            self.trained_pipeline.named_steps['final_model'].model = calibrated_classifier



        # Delete values that we no longer need that are just taking up space.
        del self.X_test
        del self.y_test
        del self.grid_search_pipelines
        del X_df


    # This is broken out into it's own function for each estimator on purpose
    # When we go to perform hyperparameter optimization, the hyperparameters for a GradientBoosting model will not at all align with the hyperparameters for an SVM. Doing all of that in one giant GSCV would throw errors. So we train each model in it's own grid search.
    # This also lets us test on X_test and y_test for each model
    def perform_grid_search_by_model_names(self, estimator_names, scoring, X_df, y):

        for model_name in estimator_names:

            ppl = self._construct_pipeline(model_name=model_name)

            self.grid_search_params = self._construct_pipeline_search_params(user_defined_model_names=estimator_names)

            # self.grid_search_params['final_model__model_name'] = [model_name]

            if self.optimize_final_model is True or (self.compute_power >= 5 and self.optimize_final_model is not False):
                raw_search_params = utils_models.get_search_params(model_name)
                for param_name, param_list in raw_search_params.items():
                    self.grid_search_params['final_model__model__' + param_name] = param_list

            if self.verbose:
                grid_search_verbose = 5
            else:
                grid_search_verbose = 0

            # Only fit GridSearchCV if we actually have hyperparameters to optimize.
            # Oftentimes, we'll just want to train a pipeline using all the default values, or using the user-provided values.
            # In those cases, fitting GSCV is unnecessarily computationally expensive.
            self.fit_grid_search = False
            for key, val in self.grid_search_params.items():

                # if it is a list, and has a length > 1, we will want to fit grid search
                if hasattr(val, '__len__') and (not isinstance(val, str)) and len(val) > 1 and key != 'final_model__model':
                    self.fit_grid_search = True

            # Here is where we will want to build in the logic for handling cases of no X_test, and no GSCV, but multiple models. Just add them to the GSCV params, and run GSCV, and we should be set.
            self.continue_after_single_gscv = False

            if self.fit_grid_search == False and (self.X_test is None and self.y_test is None) and len(estimator_names) > 1:

                final_model_models = map(lambda estimator_name: utils_models.get_model_from_name(estimator_name), estimator_names)
                self.grid_search_params['final_model__model'] = list(final_model_models)
                self.fit_grid_search = True
                self.continue_after_single_gscv = True

            if self.fit_grid_search == True:

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
                    scoring=scoring.score,
                    # ,pre_dispatch='1*n_jobs'
                )

                if self.verbose:
                    print('\n\n********************************************************************************************')
                    if self.continue_after_single_gscv:
                        print('About to run GridSearchCV on the pipeline for several models to predict ' + self.output_column)
                    else:
                        print('About to run GridSearchCV on the pipeline for the model ' + model_name + ' to predict ' + self.output_column)

                gs.fit(X_df, y)

                self.trained_pipeline = gs.best_estimator_

                # write the results for each param combo to file for user analytics.
                if self.write_gs_param_results_to_file:
                    utils.write_gs_param_results_to_file(gs, self.gs_param_file_name)

                if self.print_training_summary_to_viewer:
                    self.print_training_summary(gs)

                # We will save the info for this pipeline grid search, along with it's scores on the CV data, and the holdout data
                pipeline_results = []
                pipeline_results.append(gs.best_score_)
                pipeline_results.append(gs)

                if self.continue_after_single_gscv == True:

                    # Print ml_for_analytics here, since we break out of the loop before we can do it below
                    if 'final_model__model' in gs.best_params_:
                        model_name = gs.best_params_['final_model__model']

                    if self.ml_for_analytics and model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
                        self._print_ml_analytics_results_linear_model()

                    elif self.ml_for_analytics and model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor', 'GradientBoostingRegressor', 'GradientBoostingClassifier']:
                        self._print_ml_analytics_results_random_forest()

                    break

            # The case where we just want to run the training straight through, not fitting GridSearchCV
            else:
                if self.verbose:
                    print('\n\n********************************************************************************************')
                    if self.name is not None:
                        print(self.name)
                    print('About to fit the pipeline for the model ' + model_name + ' to predict ' + self.output_column)
                    print('Started at:')
                    start_time = datetime.datetime.now().replace(microsecond=0)
                    print(start_time)

                ppl.fit(X_df, y)
                self.trained_pipeline = ppl

                if self.verbose:
                    print('Finished training the pipeline!')
                    print('Total training time:')
                    print(datetime.datetime.now().replace(microsecond=0) - start_time)

            if self.ml_for_analytics and model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
                self._print_ml_analytics_results_linear_model()
            elif self.ml_for_analytics and model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor', 'GradientBoostingRegressor', 'GradientBoostingClassifier']:
                self._print_ml_analytics_results_random_forest()

            if (self.X_test) is not None and (self.y_test) is not None:
                if len(self.X_test) > 0 and len(self.y_test) > 0 and len(self.X_test) == len(self.y_test):
                    print('Calculating score on holdout data')
                    holdout_data_score = self.score(self.X_test, self.y_test)
                    print('The results from the X_test and y_test data passed into ml_for_analytics (which were not used for training- true holdout data)')
                    print(self.output_column + ':')
                    print(holdout_data_score)

                try:
                    # We want our score on the holdout data to be the first thing in our pipeline results tuple. This is what we will be selecting our best model from.
                    pipeline_results.prepend(holdout_data_score)
                except:
                    # If we do not have pipeline_results defined already, that means we did not fit grid search, but are relying on X_test/y_test to determine our best model
                    pipeline_results = []
                    pipeline_results.append(holdout_data_score)
                    pipeline_results.append(self.trained_pipeline)

                    # If we don't have pipeline_results (if we did not fit GSCV), then pass
                    pass

            try:
                self.grid_search_pipelines.append(pipeline_results)
            except Exception as e:
                pass

            # if self.fit_grid_search:


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
            print(str(feature[0]) + ': ' + str(round(feature[1] / sum_of_all_feature_importances, 4)))


    def _print_ml_analytics_results_random_forest(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)
        if self.name is not None:
            print(self.name)
        print('predicting ' + self.output_column)

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

        trained_feature_names = self.trained_pipeline.named_steps['dv'].get_feature_names()

        return trained_feature_names


    def _print_ml_analytics_results_linear_model(self):
        print('\n\nHere are the results from our ' + self.trained_pipeline.named_steps['final_model'].model_name)

        trained_feature_names = self._get_trained_feature_names()

        if self.type_of_estimator == 'classifier':
            trained_coefficients = self.trained_pipeline.named_steps['final_model'].model.coef_[0]
        else:
            trained_coefficients = self.trained_pipeline.named_steps['final_model'].model.coef_

        # feature_ranges = self.trained_pipeline.named_steps['final_model'].feature_ranges

        # TODO(PRESTON): readability. Can probably do this in a single zip statement.
        feature_summary = []
        for col_idx, feature_name in enumerate(trained_feature_names):
            # Ignoring potential_impact for now, since we're performing feature scaling by default
            # potential_impact = feature_ranges[col_idx] * trained_coefficients[col_idx]
            # summary_tuple = (feature_name, trained_coefficients[col_idx], potential_impact)
            summary_tuple = (feature_name, trained_coefficients[col_idx])
            feature_summary.append(summary_tuple)

        sorted_feature_summary = sorted(feature_summary, key=lambda x: abs(x[1]))

        print('The following is a list of feature names and their coefficients. By default, features are scaled to the range [0,1] in a way that is robust to outliers, so the coefficients are usually directly comparable to each other.')
        print('This printed list will contain at most the top 50 features.')
        for summary in sorted_feature_summary[-50:]:

            print(str(summary[0]) + ': ' + str(round(summary[1], 4)))
            # Again, we're ignoring feature_ranges for now since we apply feature scaling by default
            # print('The potential impact of this feature is: ' + str(round(summary[2], 4)))


    def print_training_summary(self, gs):
        print('The best CV score from GridSearchCV (by default averaging across k-fold CV) for ' + self.output_column + ' is:')
        if self.took_log_of_y:
            print('    Note that this score is calculated using the natural logs of the y values.')
        print(gs.best_score_)
        print('The best params were')

        # Remove 'final_model__model' from what we print- it's redundant with model name, and is difficult to read quickly in a list since it's a python object.
        if 'final_model__model' in gs.best_params_:
            printing_copy = {}
            for k, v in gs.best_params_.items():
                if k != 'final_model__model':
                    printing_copy[k] = v
                else:
                    printing_copy[k] = utils_models.get_name_from_model(v)
        else:
            printing_copy = gs.best_params_

        print(printing_copy)

        if self.verbose:
            print('Here are all the hyperparameters that were tried:')
            raw_scores = gs.grid_scores_
            sorted_scores = sorted(raw_scores, key=lambda x: x[1], reverse=True)
            for score in sorted_scores:
                for k, v in score[0].items():
                    if k == 'final_model__model':
                        score[0][k] = utils_models.get_name_from_model(v)
                print(score)
        # Print some nice summary output of all the training we did.
        # maybe allow the user to pass in a flag to write info to a file


    def predict(self, prediction_data):
        prediction_data = prediction_data.copy()
        if isinstance(prediction_data, list):
            prediction_data = pd.DataFrame(prediction_data)

        # If we are predicting a single row, we have to turn that into a list inside the first function that row encounters.
        # For some reason, turning it into a list here does not work.
        predicted_vals = self.trained_pipeline.predict(prediction_data)
        if self.took_log_of_y:
            for idx, val in predicted_vals:
                predicted_vals[idx] = math.exp(val)

        # del prediction_data
        return predicted_vals


    def predict_proba(self, prediction_data):
        prediction_data = prediction_data.copy()
        if isinstance(prediction_data, list):
            prediction_data = pd.DataFrame(prediction_data)

        return self.trained_pipeline.predict_proba(prediction_data)


    def score(self, X_test, y_test, advanced_scoring=True, verbose=2):

        if isinstance(X_test, list):
            X_test = pd.DataFrame(X_test)
        y_test = list(y_test)

        if self._scorer is not None:
            if self.type_of_estimator == 'regressor':
                return self._scorer.score(self.trained_pipeline, X_test, y_test, self.took_log_of_y, advanced_scoring=advanced_scoring, verbose=verbose, name=self.name)

            elif self.type_of_estimator == 'classifier':
                # TODO: can probably refactor accuracy score now that we've turned scoring into it's own class
                if self._scorer == accuracy_score:
                    predictions = self.trained_pipeline.predict(X_test)
                    return self._scorer.score(y_test, predictions)
                elif advanced_scoring:
                    score, probas = self._scorer.score(self.trained_pipeline, X_test, y_test, advanced_scoring=advanced_scoring)
                    utils_scoring.advanced_scoring_classifiers(probas, y_test, name=self.name)
                    return score
                else:
                    return self._scorer.score(self.trained_pipeline, X_test, y_test, advanced_scoring=advanced_scoring)
        else:
            return self.trained_pipeline.score(X_test, y_test)


    def save(self, file_name='auto_ml_saved_pipeline.dill', verbose=True):
        with open(file_name, 'wb') as open_file_name:
            dill.dump(self.trained_pipeline, open_file_name)

        if verbose:
            print('\n\nWe have saved the trained pipeline to a filed called "' + file_name + '"')
            print('It is saved in the directory: ')
            print(os.getcwd())
            print('To use it to get predictions, please follow the following flow (adjusting for your own uses as necessary:\n\n')
            print('`with open("' + file_name + '", "rb") as read_file:`')
            print('`    trained_ml_pipeline = dill.load(read_file)`')
            print('`trained_ml_pipeline.predict(list_of_dicts_with_same_data_as_training_data)`\n\n')

            print('Note that this pickle/dill file can only be loaded in an environment with the same modules installed, and running the same Python version.')
            print('This version of Python is:')
            print(sys.version_info)

            print('\n\nWhen passing in new data to get predictions on, columns that were not present (or were not found to be useful) in the training data will be silently ignored.')
            print('It is worthwhile to make sure that you feed in all the most useful data points though, to make sure you can get the highest quality predictions.')
            # print('\nThese are the most important features that were fed into the model:')

            # if self.ml_for_analytics and self.trained_pipeline.named_steps['final_model'].model_name in ('LogisticRegression', 'RidgeClassifier', 'LinearRegression', 'Ridge'):
            #     self._print_ml_analytics_results_linear_model()
            # elif self.ml_for_analytics and self.trained_pipeline.named_steps['final_model'].model_name in ['RandomForestClassifier', 'RandomForestRegressor', 'XGBClassifier', 'XGBRegressor']:
            #     self._print_ml_analytics_results_random_forest()

        return os.path.join(os.getcwd(), file_name)


