Deep Learning & Feature Learning
=================================


Deep Learning in auto_ml
-------------------------

Deep Learning is available in auto_ml if you've got Keras and TensorFlow installed on your machine. It's simple: just choose `model_names='DeepLearningRegressor'` or model_names='DeepLearningClassifier'` when invoking `ml_predictor.train()`.

That's right, we've got automated Deep Learning that runs at production-ready speeds (roughly 1ms per prediction when getting predictions one-at-a-time).





Feature Learning
-----------------

Deep Learning is great for learning features for you. It's not so amazing at turning those features into predictions (No Free Hunch all you will, I don't see too many people using Perceptrons as standalone models to turn features into predictions - frequently Gradient Boosting wins here).

So, why not use both models for what they're best at: Deep Learning to learn features for us, and Gradient Boosting to turn those features into accurate predictions?

That's exactly what the `feature_learning=True` param is for in auto_ml.

First, we'll train up a deep learning model on the `fl_data`, which is a dataset you have to pass into .train: `ml_predictor.train(df_train, feature_learning=True, fl_data=df_fl_data)`. This dataset should be a different dataset than your training data to avoid overfitting.

Once we've trained the feature_learning model, we'll split off it's final layer, and instead use it's penultimate layer, which outputs it's 10 most useful features. We'll hstack these features along with the rest of the features you have in your training data. So if you have 100 features in your training data, we'll add the 10 predicted features from the feature_learning model, and get to 110 features total.

Then we'll train a gradient boosted model (or any model of your choice - the full set of auto_ml models are available to you here) on this combined set of 110 features.

The result from this hybrid approach is up to 5% more accurate than either approach on their own.

All the typical best practices from deep learning apply here: be careful with overfitting, try to have a large enough dataset that this is actually useful, etc.

This isn't a super-novel approach (you could reasonably argue it's just a form of stacking which is already used in pretty much every Kaggle competition ever). However, conceptually, it's fun to think of taking a neural network, and swapping out it's simple output layer for a more complex gradient-boosted model as the output layer.

And, of course, this being auto_ml, it all runs automatically and at production-speeds (predictions take between 1 and 4 ms when getting predictions one-at-a-time).

Let me know if you have any feedback or improvements! It's pretty exciting to find an application for Deep Learning that improves predictions for standard classification and regression problems by up to 5%.
