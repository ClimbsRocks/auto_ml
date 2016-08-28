Ensembling & Subpredictors
===================================

I would say this is the tough stuff, but removing all the annoying manual work typically associated with ensembling is one of `auto_ml`'s best feaures.

Code Example
-------------------------------------

.. code-block:: python

  from auto_ml import Predictor

  # Whichever column holds the true y-values for the subpredictor you want to train should hold the value 'regressor' or 'classifier'
  col_desc_dictionary = {col_to_predict: 'output', state_code: 'categorical', 'subpredictor_name': 'regressor'}

  ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=col_desc_dictionary)

  ml_predictor.train(list_of_dictionaries, ml_for_analytics=True)
  # Wait for the machine to learn all the complex and beautiful patterns in your data...
  # If you ask for ml_for_analytics, the meta estimator will tell you how useful the subpredictor was in making the meta estimations.

  ml_predictor.predict(new_data)


What is ensembling?
--------------------------------------

Ensembling simply means using multiple machine learning algorithms together.

In our case, we focus on two primary use cases for ensembling:
1. Creating subpredictors
2. Ensembling weak classifiers trained on the main prediction problem

In both cases, we are using machine learning to make a prediction on some portion of our overall meta problem. These predictions are then fed in as features to our overall meta-estimator, along with all the other features in our dataset.


Subpredictor description
-------------------------------------

Let's say you have a meta problem you're trying to make a prediction on, like whether your home team was going to win a sports game. If I told you before the game how many points your team was going to score, you'd probably find that really useful in predicting whether your team was going to win or not.

Alas, I don't yet have a crystal ball that will tell me the future. But I've got something that's rapidly becoming nearly as good: a computer, with open-sourced software that makes it trivially easy to perform machine learning (totally unnecessary shameless plug given that you're already using said software). So what if we trained an estimator to predict how many runs your team would score? What features would we want for that? We'd probably want a ton of the features that you've already pulled to try to predict the meta problem (who's going to win the game)!

You might wonder "If I'm feeding all the same features into this sub-estimator that I'm feeding into my meta-estimator, what's the point?". As you'll soon find out in the section below, getting a bunch of experts to make a prediction on the problem, and then using a super-expert to choose from amongst those predictions, is actually a super useful approach in and of itself (both in real life, and machine learning). But we get an additional positive benefit here: This subpredictor is making predictions on a smaller and more precise problem, allowing it to be more accurate on just that one section. If we are able to build an accurate enough subpredictor, you can think of it as being very similar to just feeding in information on how many points your team was going to score.


Ensembling weak classifiers description
-----------------------------------------

Let's imagine for a minute that you're trying to predict whether your hometown ice cream place is going to win the uber-prestigious National Ice Cream Bowl.

Mmmmmm, ice cream...





