Categorical Ensembling
=======================

Let's say you operate in 100 different markets across the world. At some point in time, your local leader in Johannesburg is likely going to ask your for her own model, because her market is different than everyone else's market.

You'll probably respond with some version of "the giant monolith model we've got running right now should take everything in the dataset into account, so if your market behaves differently than everyone else's, the model will have already picked that up."

Except, if you're curious, you might try to go train a model just for Johannesburg, and find that, uh, well, that model's a bit more accurate for Johannesburg than your monolith model. Not dramatically, but maybe by 2%.

So then you modify your answer to this awesome local leader lady whose ask ended up being surprisingly good: "Uh, well, turns out that you had a good idea, but, um, I really, really don't want to support 100 different models in production. There are so many ways that can go wrong. So, I'd love to give you your own model, but I can't do that for everyone, so, until you can hire me 100x as many ML engineers, I can't do that yet."

Well buckle up, 'cause you're about to become a 100x ML engineer :)

categorical_ensembling is exactly this use case. auto_ml will automatically train one model for each category of thing (in this case, 100 different models, one for each local market). You still just call `trained_pipeline.predict(data)`, and it will return the prediction from the correct model. Heck, unless you use deep learning for each of the 100 models, the trained pipeline is saved into just one file, just like a normal auto_ml trained pipeline. You've basically abstracted away all the complexity of training 100 models for each of your 100 markets, while getting around a 2% performance boost.

How does this work in practice?

Slightly different UI:
instead of `ml_predictor.train()`, you'll invoke `ml_predictor.train_categorical_ensemble()`.

You'll need to pass in an identifier for which column of data is our `categorical_column`. In our example above, it would be something like `ml_predictor.train_categorical_ensemble(df_train, categorical_column='market_name')`.

The only other thing you have to keep in mind is to pass in a value for `'market_name'` for each new item you want to get a prediction for. The API for this doesn't change from our normal predict API, I'm just calling out that this column should have a value in it.

``test_data = {'blah': 1, 'market_name': 'Mumbai'}``
``trained_ml_predictor.predict(test_data)``

A couple user-friendly features I've built on top:

1. Default category: Your company just launched 25 new markets - awesome! Except, uh, we obviously don't have any data for them yet, so we can't train a model for them. You can specify a default category, so that when we're asked to make a prediction for a category that wasn't in the training data, we use the predictor trained on this default category to generate the prediction. If you don't specify a default_category, we will choose the most-commonly-occurring (largest) category as the default. Or, you can specify `ml_predictor.train_categorical_ensemble(df_train, categorical_column='market_name', default_category='_RAISE_ERROR')` to raise an error if we're getting a prediction for a category that wasn't in our training data.

2. min_category_size: For a number of reasons, training an ML predictor on too few data points is messy. You can certainly clean the data yourself to have an effective min_category_size, but we built this in as a convenience for the user. If there's a category that has less than min_category_size observations in our training data, we won't train a model for that category, and we'll default to using the default_category to get predictions. The default value here right now is 5 (which seems really low, if you've got the data for it, I'd recommend a min size of at least a few thousand). To finish the example above: `ml_predictor.train_categorical_ensemble(df_train, categorical_column='market_name', default_category='Beijing', min_category_size=5000)`

3. You can still pass in any of the normal arguments for `.train()` that you know and love!

Performance Notes:

A. We train up only one global transformation_pipeline to minimize disk space when serializing the models
B. If `feature_learning=True` is passed in, we will train up one global feature_learning model - we will NOT train up one feature_learning model per category. The model we train for each category can then decide whether and how to use the features from our feature_learning model. Since each feature_learning model has to be serialized to disk separately right now, this design decision was made to reduce complexity, and the risk of things going wrong when transferring trained models to a production environment.
