Properly Formal API Documenation
================================


auto_ml
-------

.. py:class:: Predictor(type_of_estimator, column_descriptions)

  :param type_of_estimator: Whether you want a classifier or regressor
  :type type_of_estimator: 'regressor' or 'classifier'
  :param column_descriptions: A key/value map noting which column is ``'output'``, along with any columns that are ``'nlp'``, ``'date'``, ``'ignore'``, or ``'categorical'``. See below for more details.
  :type column_descriptions: dictionary, where each attribute name represents a column of data that will be present in at least some of the rows of training data, and each value describes that column as being either ['categorical', 'output', 'nlp', 'date', 'ignore']. Note that 'continuous' data does not need to be labeled as such (all columns are assumed to be continuous unless labeled otherwise).

.. py:method:: ml_predictor.train(raw_training_data, user_input_func=None)

  :rtype: None. This is purely to fit the entire pipeline to the data. It doesn't return anything- it saves the fitted pipeline as a property of the ``Predictor`` instance.

  :param raw_training_data: The data to train on. See below for more information on formatting of this data.
  :type raw_training_data: DataFrame, or a list of dictionaries, where each dictionary has both the input data as well as the target data the ml estimator is trying to predict.

  :param user_input_func: A function that you can define that will be called as the first step in the pipeline. The function will be passed the entire X dataset, must not alter the order or length of the X dataset, and must return the entire X dataset. You can perform any feature engineering you would like in this function. See below for more details.
  :type user_input_func: function

  :param compute_power: The higher the number, the more options for hyperparameters we'll try to train, which could lead to a more accurate model, but will definitely lead to more compute time.
  :type compute_power: int, from 1 - 10

  :param ml_for_analytics: Whether or not to print out results for which coefficients the trained model found useful. If ``True``, you will see results that an analyst might find interesting printed to the shell.
  :type ml_for_analytics: Boolean

  :param user_input_func: A user-provided function that is used to perform feature engineering. This function will be passed X as it's only parameter, and must return a list of the exact same length and order as the X list passed in. Highly useful if you want to make sure your feature engineering is applied evenly across train, test, and prediction data in an easy and consistent way. For more information, please consult the docs for scikit-learn's ``FunctionTransformer``.
  :type user_input_func: function

  :param optimize_final_model: Whether or not to perform RandomizedSearchCV on the final model. Increases computation time significantly, but on a large enough dataset, will likely increase accuracy. Even if ``True``, we will try running a model without optimizing the hyperparameters of the final model just to see if that's better by avoiding overfitting.
  :type optimize_final_model: Boolean

  :param perform_feature_selection: Whether or not to run feature selection over our features before training the final model. Feature selection means picking only the most useful features, so we don't confuse the model with too much useless noise. Feature selection typically speeds up computation time by reducing the dimensionality of our dataset, and tends to combat overfitting as well.
  :type perform_feature_selection: Boolean

  :param X_test: Validation data. If you give validation data to auto_ml, it will report out on the results of the validation data automatically, and more frequently (once for each model that we try). Must be accompanied by y_test (the true observed values for the validation data). Typically, we recommend passing in 20% of your overall dataset as validation data.
  :type X_test: DataFrame, or list of dictionaries, same format as raw_training_data

  :param y_test: The true values for the validation data X_test. This is to compare the accuracy of our trained models to the observed reality.
  :type y_test: list, of length len(X_test)

  :param take_log_of_y: For regression problems, accuracy is oftentimes improved by taking the natural log of y values during training. This is oftentimes a pain, because then predicted values must be exponented accordingly to get back to the scale the user expects. auto_ml can handle all this automatically if you pass in ``take_log_of_y=True``.
  :type take_log_of_y: Boolean

  :param model_names: Which models to try. Acceptable values are ['Ridge', 'XGBRegressor', 'RANSACRegressor', 'RandomForestRegressor', 'LinearRegression', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'RidgeClassifier', 'XGBClassifier', 'LogisticRegression', 'RandomForestClassifier']. Note that this parameter must be a list of strings, not a single string.
  :type model_names: list of strings


.. py:method:: ml_predictor.predict(prediction_rows)

  :param prediction_rows: A single row, or a DataFrame or list of dictionaries with many rows. For production environments, the code is optimized to run quickly on a single row passed in as a dictionary, though batched predictions on thousands of rows at a time are generally more efficient if you're getting predictions for a larger dataset.

  :rtype: list of predicted values, of the same length as the ``prediction_rows`` passed in. Each row will hold a single value. For 'regressor' estimators, each value will be a number. For 'classifier' estimators, each row will be a sting of the predicted label (category), matching the categories passed in to the training data. If a single dictionary is passed in, the return value will be the predicted value, not nested in a list (so just a single number or predicted class).


.. py:method:: ml_predictor.predict_proba(prediction_rows)

  :param prediction_rows: Same as for predict above.

  :rtype:  Only works for 'classifier' estimators. Same as above, except each row in the returned list will now itself be a list, of length (number of categories in training data). The items in this row's list will represent the probability of each category.


.. py:method:: ml_predictor.score(X_test, y_test)

  :rtype: number representing the trained estimator's score on the validation data. Note that you can also pass X_test and y_test into .train() to have scores on validation data reported out for each algorithm we try, and each subpredictor we build.

.. py:method:: ml_predictor.save(file_name='auto_ml_saved_pipeline.pkl', verbose=True)

  :param file_name: [OPTIONAL] The name of the file you would like the trained pipeline to be saved to.
  :type file_name: string
  :param verbose: If ``True``, will log information about the file, the system this was trained on, and which features to make sure to feed in at prediction time.
  :type verbose: Boolean
  :rtype: the name of the file the trained ml_predictor is saved to.
