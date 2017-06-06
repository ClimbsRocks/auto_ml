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
   formatting_data.rst
   analytics.rst
   feature_responses.rst
   categorical_ensembling.rst
   deep_learning.rst
   api_docs_for_geeks.rst

Installation
------------

``pip install auto_ml``


Core Functionality Example
--------------------------

auto_ml is designed for production. Here's an example that includes serializing and loading the trained model, then getting predictions on single dictionaries, roughly the process you'd likely follow to deploy the trained model.

.. code-block:: python

  from auto_ml import Predictor
  from auto_ml.utils import get_boston_dataset
  from auto_ml.utils_models import load_ml_model

  # Load data
  df_train, df_test = get_boston_dataset()

  # Tell auto_ml which column is 'output'
  # Also note columns that aren't purely numerical
  # Examples include ['nlp', 'date', 'categorical', 'ignore']
  column_descriptions = {
    'MEDV': 'output'
    , 'CHAS': 'categorical'
  }

  ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

  ml_predictor.train(df_train)

  # Score the model on test data
  test_score = ml_predictor.score(df_test, df_test.MEDV)

  # auto_ml is specifically tuned for running in production
  # It can get predictions on an individual row (passed in as a dictionary)
  # A single prediction like this takes ~1 millisecond
  # Here we will demonstrate saving the trained model, and loading it again
  file_name = ml_predictor.save()

  trained_model = load_ml_model(file_name)

  # .predict and .predict_proba take in either:
  # A pandas DataFrame
  # A list of dictionaries
  # A single dictionary (optimized for speed in production evironments)
  predictions = trained_model.predict(df_test)
  print(predictions)

XGBoost, Deep Leaarning with TensorFlow & Keras, and LightGBM
-------------------------------------------------------------

auto_ml has all three of these awesome libraries integrated!
Generally, just pass one of them in for model_names.
`ml_predictor.train(data, model_names=['DeepLearningClassifier'])`

Available options are
- `DeepLearningClassifier` and `DeepLearningRegressor`
- `XGBClassifier` and `XGBRegressor`
- `LGBMClassifer` and `LGBMRegressor`

All of these projects are ready for production. These projects all have prediction time in the 1 millisecond range for a single prediction, and are able to be serialized to disk and loaded into a new environment after training.

Depending on your machine, they can occasionally be difficult to install, so they are not included in auto_ml's default installation. You are responsible for installing them yourself. auto_ml will run fine without them installed (we check what's isntalled before choosing which algorithm to use). If you want to try the easy install, just `pip install -r advanced_requirements.txt`, which will install TensorFlow, Keras, and XGBoost. LightGBM is not available as a pip install currently.


Classification
--------------

Binary and multiclass classification are both supported. Note that for now, labels must be integers (0 and 1 for binary classification). auto_ml will automatically detect if it is a binary or multiclass classification problem- you just have to pass in `ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)`


Advice
------

Before you go any further, try running the code. Load up some dictionaries in Python, where each dictionary is a row of data. Make a ``column_descriptions`` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into ``auto_ml``, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.



What this project does
=======================
Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.

A quick overview of buzzwords, this project automates:

#. Analytics (pass in data, and auto_ml will tell you the relationship of each variable to what it is you're trying to predict).
#. Feature Engineering (particularly around dates, and soon, NLP).
#. Robust Scaling (turning all values into their scaled versions between the range of 0 and 1, in a way that is robust to outliers, and works with sparse matrices).
#. Feature Selection (picking only the features that actually prove useful).
#. Data formatting (turning a list of dictionaries into a sparse matrix, one-hot encoding categorical variables, taking the natural log of y for regression problems).
#. Model Selection (which model works best for your problem).
#. Hyperparameter Optimization (what hyperparameters work best for that model).
#. Ensembling Subpredictors (automatically training up models to predict smaller problems within the meta problem).
#. Ensembling Weak Estimators (automatically training up weak models on the larger problem itself, to inform the meta-estimator's decision).
#. Big Data (feed it lots of data).
#. Unicorns (you could conceivably train it to predict what is a unicorn and what is not).
#. Ice Cream (mmm, tasty...).
#. Hugs (this makes it much easier to do your job, hopefully leaving you more time to hug those those you care about).
