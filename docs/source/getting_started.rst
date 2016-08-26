Getting Started
===============

Installation
------------

``pip install auto_ml``


Core Functionality Example
--------------------------

.. code-block:: python

  from auto_ml import Predictor

  col_desc_dictionary = {col_to_predict: 'output'}

  ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)
  # Can pass in type_of_estimator='regressor' as well

  ml_predictor.train(list_of_dictionaries)
  # Wait for the machine to learn all the complex and beautiful patterns in your data...

  ml_predictor.predict(new_data)
  # Where new_data is also a list of dictionaries