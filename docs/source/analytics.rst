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

  # Can pass in type_of_estimator='regressor' as well
  ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)

  # Wait for the machine to learn all the complex and beautiful patterns in your data...
  ml_predictor.train(df, ml_for_analytics=True)

  # And this time, in your shell, it will print out the results for what it found was useful in making predictions!

  # Where new_data is a single dictionary, or a DataFrame
  ml_predictor.predict(new_data)

Tangent time - what do you mean analytics from machine learning?
---------------------------------------------------------------
One of my favorite analogies for this (and really, for machine learning in general), is to think of a loan officer at a regional bank in, say, the 1940's or some other pre-computer era. She's been there for 30 years. She's seen thousands of loans cross her desk, and over time she's figured out what makes a loan likely to default, or likely to be healthy.

As a bright-eyed and bushy-tailed (mabye not bushy-tailed, you probably kept your furry tendencies in the closet back then) new loan officer, you probably wanted to learn her secrets! What was it that mattered? How could you read a loan application and figure out whether to give them a huge chunk of money or not?

It's the exact same process with machine learning. You feed the machine a ton of loan applications (in digital form now, not analog). You tell it which ones were good and which ones were bad. It learns the patterns in the data. And based on those patterns, it's able to learn in a matter of minutes what used to take our amazing loan officer decades of experience to figure out.

And just like our awesome loan officer, we can ask the machine to tell us what it learned.


Note - no free lunch
-------------------
As with nearly everything in life (other than that occasional post-climb burrito where they get the crispiness just right, or that parallel parking job in the tight space you get on the first go), this isn't perfect. Once you dive deeper into the weeds, there are, of course, all kinds of caveats. However, this approach to analytics is typically much more robust than simply building a chart to compare two variables without considering anything else. The approach of just taking two variables and charting them against each other opens us up to huge classes of errors. The approaches used here will usually knock out most of those huge classes of errors, and instead open us up to much smaller classes of errors.

Again, this isn't perfect (and really, as an analyst, you should rarely if ever be claiming that something is perfect), but it is typically a vast improvement over most analytics workflows, and very straightforward to use.


Interpreting Results
--------------------

The information coming from regression based models will be the coefficient for each feature. Note that by default, features are scaled to roughly the range of [0,1].

Roughly, you can read the results as "all else equal, we'd expect a change from being the smallest to the largest value on this particular variable will lead to [coefficient] impact on our output variable".

The information coming from random-forest-esque models will be roughly the amount of variance that is explained by this feature. These values will, in total, sum up to 1.


Interpreting Predicted Probability Buckets for Classifiers
----------------------------------------------------------

Sometimes, it's useful to know how a classifier is doing very granularly, beyond just accuracy. For instance, pretend an expensive event (say, burning a whole batch of pastries) has a 5% chance of occurring.

If you train a model, obtaining 95% accuracy looks pretty bad on the surface - it's no better than average! And in fact, you'll probably find that most (or all) of the predictions are 0 - predicting that the pastries will not burn.

It's easy to disregard the model at this point.

However, we might still find some use for it, if we dive deeper into the predicted probabilities. Maybe, for 80% of deliveries, the model predicts 0 probability of fire, while for 20% of deliveries, the model predicts 25% chance of fire. That would be quite useful if it's accurate at each of those probabilities! We're able to correctly identify all the batches that have very low risk of fire, and a subset of the batches that are 5x as risky as our average batch. That sounds pretty promising!

That's what we report out in advanced scoring for classifiers.

We take the model's predicted probabilities on every item in the scoring dataset. Then we order the predicted probabilities from lowest to highest. We bucket those sorted predictions into 10 buckets, with the lowest bucket holding the 10% of the dataset that the model predicted the lowest probability for, and the highest bucket holding the 10% of the dataset that the model predicted the highest probability for.

Then, for each bucket, we simply report what the average predicted probability was, and what the actual event occurrence was, along with what the max and min predicted probabilities were for that bucket.



In the weeds
------------

A collection of random thoughts that are much deeper into the weeds, and should really only be read once you've run the code above a few times, if at all.

#. The only types of models that support introspection like this currently are tree-based models (DecisionTree, RandomForest, XGBoost/GradientBoosted trees, etc.), and regression models (LinearRegression, LogisticRegression, Ridge, etc.).
#. Note that we are not doing any kind of PCA or other method of handling features that might be correlated with each other. So if you feed in two features that are highly related (say, number of items in order, and total order cost), you can oftentimes find some weird results, particularly from regression-based models. For instance, it might have an extra large positive coefficient for one of the features (number of items), and a negative coefficient on the other (total cost) to balance that out. If you find this happening, try running the model again with one of the two correlated features removed.
#. For forests handling two correlated variables, it will typically pick one of the variables as being highly important, and the other as relatively unimportant.
#. We scale all features to the range of roughly [0,1], so when you're interpreting the coefficients, they're fairly directly comparable in scale. For example, say we have two variables: number of items in order, and order total, in cents. Your order total variable might reasonably range from 50 to 10,000, while your number of items might only range from 1 - 10. Thus, the coefficient on the raw order total is going to be much smaller than the coefficient on number of items. But this might not accurately reflect the relative impact of these two features, because the order total feature can multiply that coefficient by a mugh larger range. When we scale both features to fall in the range of [0,1], we can now directly compare the coefficients. The way to read this then changes slightly. It's now "if we go from being the smallest to largest on this measure, what impact would we expect this to have on our output variable?".
#. Features with more granularity are typically more useful for models to differentiate on. Going back to our order total vs. number of items example, order total can potentially take on one of 10,000 values, while number of items can only take on 10 values. All else equal, the model will find order total more useful, simply because it has more options to perform the differentiation on.
#. The random forest will report results on features that are most broadly applicable, since it reports results on what reduces global variance/error. The regression models will report results on which features have the strongest impact WHEN THEY ARE PRESENT. So being in the state of Hawaii might come up very highly for our regression, because we find that when a row holds data form the state of Hawaii, we need to make a large adjustment. However, the tree-based model likely won't report that variable to be too useful, since very little of your data likely comes from the state of Hawaii. It will likely find a more global variable like income (which is likely present in every row) to be more useful to reduce overall error/variance.
