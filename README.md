# auto_ml
> Get a trained and optimized machine learning predictor at the push of a button (and, admittedly, an extended coffee break while your computer does the heavy lifting and you get to claim "compiling" https://xkcd.com/303/).


## Installation

- `pip install auto_ml`

OR

- `git clone https://github.com/ClimbsRocks/auto_ml`
- `pip install -r requirements.txt`


## Getting Started

```
from auto_ml import Predictor

col_desc_dictionary = {col_to_predict: 'output'}

ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)
# Can pass in type_of_estimator='regressor' as well

ml_predictor.train(list_of_dictionaries)
# Wait for the machine to learn all the complex and beautiful patterns in your data...

ml_predictor.predict(new_data)
# Where new_data is also a list of dictionaries
```

### Advice

Before you go any further, try running the code. Load up some dictionaries in Python, where each dictionary is a row of data. Make a `column_descriptions` dictionary that tells us which attribute name in each row represents the value we're trying to predict. Pass all that into `auto_ml`, and see what happens!

Everything else in these docs assumes you have done at least the above. Start there and everything else will build on top. But this part gets you the output you're probably interested in, without unnecessary complexity.


## Docs

The full docs are available at https://auto_ml.readthedocs.io
Again though, I'd strongly recommend running this on an actual dataset before referencing the docs any futher.


## What this project does

Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.

A quick overview of buzzwords, this project automates:

- Analytics (pass in data, and auto_ml will tell you the relationship of each variable to what it is you're trying to predict).
- Feature Engineering (particularly around dates, and soon, NLP).
- Robust Scaling (turning all values into their scaled versions between the range of 0 and 1, in a way that is robust to outliers, and works with sparse matrices).
- Feature Selection (picking only the features that actually prove useful).
- Data formatting (turning a list of dictionaries into a sparse matrix, one-hot encoding categorical variables, taking the natural log of y for regression problems).
- Model Selection (which model works best for your problem).
- Hyperparameter Optimization (what hyperparameters work best for that model).
- Ensembling Subpredictors (automatically training up models to predict smaller problems within the meta problem).
- Ensembling Weak Estimators (automatically training up weak models on the larger problem itself, to inform the meta-estimator's decision).
- Big Data (feed it lots of data).
- Unicorns (you could conceivably train it to predict what is a unicorn and what is not).
- Ice Cream (mmm, tasty...).
- Hugs (this makes it much easier to do your job, hopefully leaving you more time to hug those those you care about).


#### Passing in your own feature engineering function

You can pass in your own function to perform feature engineering on the data. This will be called as the first step in the pipeline that `auto_ml` builds out.

You will be passed the entire X dataset (not the y dataset), and are expected to return the entire X dataset in the same order.

The advantage of including it in the pipeline is that it will then be applied to any data you want predictions on later. You will also eventually be able to run GridSearchCV over any parameters you include here.

Limitations:
You cannot alter the length or ordering of the X dataset, since you will not have a chance to modify the y dataset. If you want to perform filtering, perform it before you pass in the data to train on.
