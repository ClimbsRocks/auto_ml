Properly Formal API Documenation
================================


auto_ml
-------

.. py:class:: Predictor(type_of_estimator, column_descriptions)

  :param type_of_estimator: Whether you want a classifier or regressor
  :type type_of_estimator: 'regressor' or 'classifier'
  :param column_descriptions: A key/value map noting which column is ``'output'``, along with any columns that are ``'nlp'``, ``'date'``, ``'ignore'``, or ``'categorical'``. See below for more details.
  :type column_descriptions: dictionary, where each attribute name represents a column of data in the training data, and each value describes that column as being either ['categorical', 'output', 'nlp', 'date', 'ignore']. Note that 'continuous' data does not need to be labeled as such (all columns are assumed to be continuous unless labeled otherwise).

.. py:method:: ml_predictor.train(raw_training_data, user_input_func=None)

  :rtype: None. This is purely to fit the entire pipeline to the data. It doesn't return anything- it saves the fitted pipeline as a property of the ``Predictor`` instance.

  :param raw_training_data: The data to train on. See below for more information on formatting of this data.
  :type raw_training_data: DataFrame, or a list of dictionaries, where each dictionary represents a row of data. Each row should have both the training features, and the output value we are trying to predict.

  :param user_input_func: [default- None] A function that you can define that will be called as the first step in the pipeline, for both training and predictions. The function will be passed the entire X dataset. The function must not alter the order or length of the X dataset, and must return the entire X dataset. You can perform any feature engineering you would like in this function. Using this function ensures that you perform the same feature engineering for both training and prediction. For more information, please consult the docs for scikit-learn's ``FunctionTransformer``.
  :type user_input_func: function

  :param ml_for_analytics: [default- True] Whether or not to print out results for which coefficients the trained model found useful. If ``True``, auto_ml will print results that an analyst might find interesting.
  :type ml_for_analytics: Boolean

  :param optimize_final_model: [default- False] Whether or not to perform GridSearchCV on the final model. True increases computation time significantly, but will likely increase accuracy.
  :type optimize_final_model: Boolean

  :param perform_feature_selection: [default- True for large datasets, False for small datasets] Whether or not to run feature selection before training the final model. Feature selection means picking only the most useful features, so we don't confuse the model with too much useless noise. Feature selection typically speeds up computation time by reducing the dimensionality of our dataset, and tends to combat overfitting as well.
  :type perform_feature_selection: Boolean

  :param take_log_of_y: For regression problems, accuracy is sometimes improved by taking the natural log of y values during training, so they all exist on a comparable scale.
  :type take_log_of_y: Boolean

  :param model_names: Which model(s) to try. Includes many scikit-learn models, deep learning with Keras/TensorFlow, and Microsoft's LightGBM. Currently available options from scikit-learn are ['ARDRegression', 'AdaBoostClassifier', 'AdaBoostRegressor', 'BayesianRidge', 'ElasticNet', 'ExtraTreesClassifier', 'ExtraTreesRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor', 'Lasso', 'LassoLars', 'LinearRegression', 'LogisticRegression', 'MiniBatchKMeans', 'OrthogonalMatchingPursuit', 'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor', 'Perceptron', 'RANSACRegressor', 'RandomForestClassifier', 'RandomForestRegressor', 'Ridge', 'RidgeClassifier', 'SGDClassifier', 'SGDRegressor']. If you have installed XGBoost, LightGBM, or Keras, you can also include ['DeepLearningClassifier', 'DeepLearningRegressor', 'LGBMClassifier', 'LGBMRegressor', 'XGBClassifier', 'XGBRegressor'].
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
