Using machine learning for analytics
====================================

Intended Audience:

#. Analysts. Yes, as an analyst who knows just a tiny bit of Python, you can run machine learning that will make your analytics both more accurate, and much cooler sounding.
#. Engineers looking to improve their models by figuring out what feature engineering to build out next.
#. Anyone interested in making business decisions, not just engineering decisions.


This is one of my favorite parts of this project: once the machines have learned all the complex patterns in the data, we can ask them what they've learned!


The code to make this work
--------------------------

It's super simple. When you train, simply pass in ``ml_for_analytics=True``, like so: ``ml_predictor.train(training_data, ml_for_analytics=True)``

Here's the whole code block that will get you analytics results in your console:

.. code-block:: python

  from auto_ml import Predictor

  # If you pass in any categorical data as a number, tell us here and we'll take care of it.
  col_desc_dictionary = {col_to_predict: 'output', state_code: 'categorical'}

  ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)
  # Can pass in type_of_estimator='regressor' as well

  ml_predictor.train(list_of_dictionaries, ml_for_analytics=True)
  # Wait for the machine to learn all the complex and beautiful patterns in your data...

  # And this time, in your shell, it will print out the results for what it found was useful in making predictions!

  ml_predictor.predict(new_data)
  # Where new_data is also a list of dictionaries


Tangent time- what do you mean analytics from machine learning?
---------------------------------------------------------------
One of my favorite analogies for this (and really, for machine learning in general), is to think of a loan officer at a regional bank in, say, the 1940's or some other pre-computer era. She's been there for 30 years. She's seen thousands of loans cross her desk, and over time she's figured out what makes a loan likely to default, or likely to be healthy.

As a bright-eyed and bushy-tailed (mabye not bushy-tailed, you probably kept your furry tendencies in the closet back then) new loan officer, you probably wanted to learn her secrets! What was it that mattered? How could you read a loan application and figure out whether to give them a huge chunk of money or not?

It's the exact same process with machine learning. You feed the machine a ton of loan applications (in digital form now, not analog). You tell it which ones were good and which ones were bad. It learns the patterns in the data. And based on those patterns, it's able to learn in a matter of minutes what used to take our amazing loan officer decades of experience to figure out.

And just like our awesome loan officer, we can ask the machine to tell us what it learned.


Interpreting Results
--------------------

