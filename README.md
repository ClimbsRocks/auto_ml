# auto_ml
> Get a trained and optimized machine learning predictor at the push of a button (and, admittedly, an extended coffee break while your computer does the heavy lifting and you get to claim "compiling" https://xkcd.com/303/).

[![Build Status](https://travis-ci.org/ClimbsRocks/auto_ml.svg?branch=master)](https://travis-ci.org/ClimbsRocks/auto_ml)
[![Documentation Status](http://readthedocs.org/projects/auto-ml/badge/?version=latest)](http://auto-ml.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/auto_ml.svg)](https://badge.fury.io/py/auto_ml)
[![Coverage Status](https://coveralls.io/repos/github/ClimbsRocks/auto_ml/badge.svg?branch=master&cacheBuster=1)](https://coveralls.io/github/ClimbsRocks/auto_ml?branch=master)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)]()
<!-- Stars badge?! -->

## Installation

- `pip install auto_ml`

OR

- `git clone https://github.com/ClimbsRocks/auto_ml`
- `pip install -r requirements.txt`


## Getting Started

```
import dill
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from auto_ml import Predictor

# Load data
boston = load_boston()
df_boston = pd.DataFrame(boston.data)
df_boston.columns = boston.feature_names
df_boston['MEDV'] = boston['target']
df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.2, random_state=42)

# Tell auto_ml which column is 'output'
# Also note columns that aren't purely numerical
# Examples include ['nlp', 'date', 'categorical', 'ignore']
column_descriptions = {
  'MEDV': 'output'
  , 'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_boston_train)

# Score the model on test data
test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

# auto_ml is specifically tuned for running in production
# It can get predictions on an individual row (passed in as a dictionary)
# A single prediction like this takes ~1 millisecond
# Here we will demonstrate saving the trained model, and loading it again
file_name = ml_predictor.save()

# dill is a drop-in replacement for pickle that handles functions better
with open (file_name, 'rb') as read_file:
    trained_model = dill.load(read_file)

# .predict and .predict_proba take in either:
# A pandas DataFrame
# A list of dictionaries
# A single dictionary (optimized for speed in production evironments)
predictions = trained_model.predict(df_boston_test)
print(predictions)
```


### Advice

Before you go any further, try running the code. Load up some data (either a DataFrame, or a list of dictionaries, where each dictionary is a row of data). Make a `column_descriptions` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into `auto_ml`, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.


## Docs

The full docs are available at https://auto_ml.readthedocs.io
Again though, I'd strongly recommend running this on an actual dataset before referencing the docs any futher.


## What this project does

Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.

A quick overview of buzzwords, this project automates:

- Analytics (pass in data, and auto_ml will tell you the relationship of each variable to what it is you're trying to predict).
- Feature Engineering (particularly around dates, and NLP).
- Robust Scaling (turning all values into their scaled versions between the range of 0 and 1, in a way that is robust to outliers, and works with sparse data).
- Feature Selection (picking only the features that actually prove useful).
- Data formatting (turning a DataFrame or a list of dictionaries into a sparse matrix, one-hot encoding categorical variables, taking the natural log of y for regression problems, etc).
- Model Selection (which model works best for your problem- we try roughly a dozen apiece for classification and regression problems, including favorites like XGBoost if it's installed on your machine).
- Hyperparameter Optimization (what hyperparameters work best for that model).
- Ensembling (Train up a bunch of different estimators, then train a final estimator to intelligently aggregate them together. Also useful if you're just trying to compare many different models and see what works best.)
- Big Data (feed it lots of data- it's fairly efficient with resources).
- Unicorns (you could conceivably train it to predict what is a unicorn and what is not).
- Ice Cream (mmm, tasty...).
- Hugs (this makes it much easier to do your job, hopefully leaving you more time to hug those those you care about).


<!--

#### Passing in your own feature engineering function

You can pass in your own function to perform feature engineering on the data. This will be called as the first step in the pipeline that `auto_ml` builds out.

You will be passed the entire X dataset (not the y dataset), and are expected to return the entire X dataset in the same order.

The advantage of including it in the pipeline is that it will then be applied to any data you want predictions on later. You will also eventually be able to run GridSearchCV over any parameters you include here.

Limitations:
You cannot alter the length or ordering of the X dataset, since you will not have a chance to modify the y dataset. If you want to perform filtering, perform it before you pass in the data to train on.

 -->


### Running the tests

If you've cloned the source code and are making any changes (highly encouraged!), or just want to make sure everything works in your environment, run
`nosetests -v tests`.

The tests are pretty comprehensive, though as with everything with auto_ml, I happily welcome your contributions here!
