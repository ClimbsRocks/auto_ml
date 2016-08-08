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


What this project does
=======================
