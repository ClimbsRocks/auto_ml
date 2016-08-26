Properly Formal API Documenation
================================

For those who prefer code over words.


auto_ml
-------

.. py:class:: Predictor(type_of_estimator, column_descriptions)

  :param type_of_estimator: Whether you want a classifier or regressor
  :type type_of_estimator: 'regressor' or 'classifier'
  :param column_descriptions: A key/value map noting which column is ``'output'``, along with any columns that are ``'nlp'`` or ``'categorical'``. See below for more details.
  :type column_descriptions: dictionary, where each attribute name represents a column of data that will be present in at least some of the rows of training data, and each value describes that column as being either ['categorical', 'output', 'nlp', or 'continuous']. Note that 'continuous' data does not need to be labeled as such (all columns are assumed to be continuous unless labeled otherwise), and 'nlp' support is not included yet.

.. py:method:: ml_predictor.train(raw_training_data, user_input_func=None)

  :rtype: None. This is purely to fit the entire pipeline to the data. It doesn't return anything- it saves the fitted pipeline as a property of the ``Predictor`` instance.
  :param raw_training_data: The data to train on. See below for more information on formatting of this data.
  :type raw_training_data: List of dictionaries, where each dictionary has both the input data as well as the target data the ml estimator is trying to predict.
  :param user_input_func: A function that you can define that will be called as the first step in the pipeline. The function will be passed the entire X dataset, must not alter the order or length of the X dataset, and must return the entire X dataset. You can perform any feature engineering you would like in this function. See below for more details.
  :type user_input_func: function

.. py:method:: ml_predictor.predict(prediction_rows)

  :rtype: list of predicted values, of the same length as the ``prediction_rows`` passed in. Each row will hold a single value. For 'regressor' estimators, each value will be a number. For 'classifier' estimators, each row will be a sting of the predicted label (category), matching the categories passed in to the training data.

.. py:method:: ml_predictor.predict_proba(prediction_rows)

  :rtype: list of predicted values, of the same length as the ``prediction_rows`` passed in. Only works for 'classifier' estimators. Each row in the returned list will now itself be a list, of length (number of categories in training data). The items in this row's list will represent the probability of each category.

.. py:method:: ml_predictor.score(X_test, y_test)

  :rtype: number representing the trained estimator's score on the validation data. Note that you can also pass X_test and y_test into .train() to have scores on validation data reported out for each algorithm we try, and each subpredictor we build.
