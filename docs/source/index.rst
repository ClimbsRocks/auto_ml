.. auto_ml documentation master file, created by
   sphinx-quickstart on Sun Aug  7 20:25:48 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to auto_ml's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 3

   self
   getting_started.rst
   ensembling.rst
   analytics.rst
   formatting_data.rst
   api_docs_for_geeks.rst

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

Advice
------

Before you go any further, try running the code. Load up some dictionaries in Python, where each dictionary is a row of data. Make a ``column_descriptions`` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into ``auto_ml``, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.

Core Functionality
===================

.. py:class:: Predictor(type_of_algo, column_descriptions)

  :param type_of_algo: Whether you want a classifier or regressor
  :type type_of_algo: 'regressor' or 'classifier'
  :param column_descriptions: A key/value map noting which column is ``'output'``, along with any columns that are ``'nlp'`` or ``'categorical'``. See below for more details.
  :type column_descriptions: dictionary

.. py:method:: ml_predictor.train(raw_training_data, user_input_func=None)

  :rtype: None. This is purely to fit the entire pipeline to the data. It doesn't return anything- it saves the fitted pipeline as a property of the ``Predictor`` instance.
  :param raw_training_data: The data to train on. See below for more information on formatting of this data.
  :type raw_training_data: List of dictionaries, where each dictionary has both the input data as well as the target data the ml algo is trying to predict.
  :param user_input_func: A function that you can define that will be called as the first step in the pipeline. The function will be passed the entire X dataset, must not alter the order or length of the X dataset, and must return the entire X dataset. You can perform any feature engineering you would like in this function. See below for more details.
  :type user_input_func: function



Formatting the training data
=============================

The only tricky part you have to do is get your training data into the format specified.



Training data format
---------------------
#. Must be a list (or other iterable) filled with python dictionaries.
#. The non-header-row objects can be "sparse". That is, they don't have to have all the properties. So if you are missing data for a certain row, or have a property that only applies to certain rows, you can include it or not at your discretion.

Header row information
-----------------------
The ``column_descriptions`` dictionary passed into ``Predictor()`` is essentially the header row. Here you've gotta specify some basic information about each "column" of data in the other dictionaries. This object should essentially have the same attributes as the following objects, except the values stored in each attribute will tell us information about that "column" of data.

#. ``attribute_name: 'output'`` The first object in your training data must specify one of your attributes as the output column that we're interested in training on. This is what the ``auto_ml`` predictor will try to predict.
#. ``attribute_name: 'categorical'`` All attribute names that hold a string in any of the rows after the header row will be encoded as categorical data. If, however, you have any numerical columns that you want encoded as categorical data, you can specify that here.
#. ``attribute_name: 'nlp'`` If any of your data is a text field that you'd like to run some Natural Language Processing on, specify that in the header row. Data stored in this attribute will be encoded using TF-IDF, along with some other feature engineering (count of some aggregations like total capital letters, puncutation characters, smiley faces, etc., as well as a sentiment prediction of that text).



Passing in your own feature engineering function
=================================================

You can pass in your own function to perform feature engineering on the data. This will be called as the first step in the pipeline that ``auto_ml`` builds out.

You will be passed the entire X dataset (not the y dataset), and are expected to return the entire X dataset.

The advantage of including it in the pipeline is that it will then be applied to any data you want predictions on later. You will also eventually be able to run GridSearchCV over any parameters you include here.

Limitations:
You cannot alter the length or ordering of the X dataset, since you will not have a chance to modify the y dataset. If you want to perform filtering, perform it before you pass in the data to train on.



What this project does
=======================
Automates the whole machine learning process!




Future API features that I definitely haven't built out yet
------------------------------------------------------------
#. ``grid_search`` aka, ``optimize_the_foobar_out_of_this_predictor``. Sit down for a long coffee break. Better yet, go make some cold brew. Come back when the cold brew's ready. As amped as you are on all that caffeine is as amped as this highly optimized algo will be. They'll also both take about the same amount of time to prepare. Both are best done overnight.
#. Support for multiple nlp columns.
#. ``print_analytics_output`` For the curious out there, sometimes we want to know what features are important. This option will let you figure that out.


Future internal features that you'll never see but will make this much better
------------------------------------------------------------------------------
#. Mostly, all kinds of stats-y feature engineering

  * RobustScaler
  * Handling correlated features
  * etc.
  * These will be mostly used by GridSearchCV, and are probably not things that you'll get to specify unless you dive into the internals of the project.

#. Feature selection
#. The ability to pass in a param_grid of your own to run during GridSearchCV that will override any of the properties we would use ourselves. Properties that are not valid will be logged to the console and summarily ignored. Yeah, it'll be ugly. That's what an MVP is for. Besides, you can handle it if you're diving this deep into the project.
#. Ensembling of results. Honestly, probably not all that practical, as it will likely increase the computation time for making each prediction rather dramatically. Worth mentioning in case some other contributor wants to add it in, as it's likely highly useful for competitions. But, not super great for production environments, so I'll probably ignore it until a future where I get very bored.

Just for kicks, here's how we'd implement ensembling:
Create our own custom transformer class.
This class will have a bunch of weak classifiers (non-tuned perceptrion, LinearRegression, etc.).
This custom transformer class will then use each of these weak predictors in a FeatureUnion to get predictions on each row, and append it to that row's features.
Then, we'll just continue on our merry way to the standard big predictor, using each of these weak predictions as features. It probably wouldn't increase the complexity too much, since we're using FeatureUnion to compute predictions in parallel...
Heavily caveat all this with how ensembling tends to overfit, so we'd probably have to build in significantly more complexity to evaluate all this on a holdout set of data.
Just thoughts for a future future scenario in which I've already conquered all my other ML ambitions and found myself with bored time on my hands again...

