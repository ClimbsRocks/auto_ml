Getting Started
===============

Installation
------------

``pip install auto_ml``


Core Functionality Example
--------------------------

.. code-block:: python

  from auto_ml import Predictor

  # If you pass in any categorical data as a number, tell us here and we'll take care of it.
  col_desc_dictionary = {col_to_predict: 'output', state_code: 'categorical'}

  ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)
  # Can pass in type_of_estimator='regressor' as well

  ml_predictor.train(list_of_dictionaries)
  # Wait for the machine to learn all the complex and beautiful patterns in your data...

  ml_predictor.predict(new_data)
  # Where new_data is also a list of dictionaries


That's it.

Seriously.

Sure, there's a ton of complexity hiding under the surface. And yes, I've got a ton more options that you can pass in to customize your experience with ``auto_ml``, but none of that's at all necessary to get some awesome results from this library.


Advice
------

Before you go any further, try running the code. Load up some dictionaries in Python, where each dictionary is a row of data. Make a ``column_descriptions`` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into ``auto_ml``, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.