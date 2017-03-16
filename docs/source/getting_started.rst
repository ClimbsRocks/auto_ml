Getting Started
===============

Installation
------------

``pip install auto_ml``


Core Functionality Example
--------------------------

.. code-block:: python

  from auto_ml import Predictor
  from auto_ml.utils import get_boston_dataset

  df_train, df_test = get_boston_dataset()

  column_descriptions = {
      'MEDV': 'output'
      , 'CHAS': 'categorical'
  }

  ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

  ml_predictor.train(df_train)

  ml_predictor.score(df_test, df_test.MEDV)


That's it.

Seriously.

Sure, there's a ton of complexity hiding under the surface. And yes, I've got a ton more options that you can pass in to customize your experience with ``auto_ml``, but none of that's at all necessary to get some awesome results from this library.


Advice
------

Before you go any further, try running the code. Load up some data (either a DataFrame, or a list of dictionaries, where each dictionary is a row of data). Make a `column_descriptions` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into `auto_ml`, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.
