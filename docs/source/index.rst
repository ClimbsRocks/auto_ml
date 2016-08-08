.. auto_ml documentation master file, created by
   sphinx-quickstart on Sun Aug  7 20:25:48 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to auto_ml's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2

Installation
=============

``pip install auto_ml``


Getting Started
================

``
from auto_ml import Predictor
col_desc_dictionary = {col_to_predict: 'output'}
ml_predictor = Predictor(type_of_algo='classifier', column_descriptions=col_desc_dictionary) # can pass in type_of_algo='regressor' as well
ml_predictor.train(my_formatted_but_otherwise_raw_training_data)
ml_predictor.predict(new_data)
``


Core Functionality
===================



Formatting the training data
=============================

The only tricky part you have to do is get your training data into the format specified. 

Training data format
---------------------
#. Must be a list (or other iterable) filled with python dictionaries.
#. The first dictionary in the list is essentially the header row. Here you've gotta specify some basic information about each "column" of data in the other dictionaries. This object should essentially have the same attributes as the following objects, except the values stored in each attribute will tell us information about that "column" of data. 
#. The non-header-row objects can be "sparse". That is, they don't have to have all the properties. So if you are missing data for a certain row, or have a property that only applies to certain rows, you can include it or not at your discretion. 

Header row information
-----------------------

#. ``attribute_name: 'output'`` The first object in your training data must specify one of your attributes as the output column that we're interested in training on. This is what the ``auto_ml`` predictor will try to predict. 
#. ``attribute_name: 'categorical'`` All attribute names that hold a string in any of the rows after the header row will be encoded as categorical data. If, however, you have any numerical columns that you want encoded as categorical data, you can specify that here. 
#. ``attribute_name: 'nlp'`` If any of your data is a text field that you'd like to run some Natural Language Processing on, specify that in the header row. Data stored in this attribute will be encoded using TF-IDF, along with some other feature engineering (count of some aggregations like total capital letters, puncutation characters, smiley faces, etc., as well as a sentiment prediction of that text). 



What this project does
=======================
Automates the whole machine learning process!

