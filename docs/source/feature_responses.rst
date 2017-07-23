Feature Responses
=================

People love linear models for their explainability. Let's take a look at the classic Titanic example, and say that the "Fare" feature has a coefficient of 0.1. The way it's classically interpreted, that means that, holding all else equal, for every unit increase in fare, the person's likelihood of surviving increased by 10%.

There are two important parts here: holding all else equal, and the response of the output variable to this feature.

These are things we can get from tree-based models as well.


Methodology
-----------

Getting feature responses from a tree-based model is pretty easy. First, we take a portion of our training dataset (10k rows, by default, though user-configurable). Then, for each column, we find that column's standard deviation. Holding all else constant, we then increment all values in that column by one standard deviation. We get predictions for all rows, and compare them to our baseline predictions. The feature_response is how much the output predictions responded to this change in the feature. We repeat the process twice for each column, once incrementing by one std, and once decrementing by one std.



Output
------

The output is how much the output variable responded to these one std increments and decrements to each feature. The value we report is the average across all predictions. It is very literally, "holding all else constant, on average, how does the output respond to this change in a feature?"

CAVEATS!!
---------

Tree-based models do not have linear relationships between features and the output variable. The relationships are more complex, which is why tree-based models can be more predictive. So even if we find that the *average* feature_response for an increase in fare is positive, that relationship may not hold for all passengers, and certainly not equally for all passengers. For instance, the model might have found a group of 1st class passengers who paid a lot of money to be in cabins that were as isolated from the main deck as possible, and thus, were less likely to survive. So even though for 3rd class passengers, an increase in fare meant they were more likely to survive, the predictions for some 1st class passengers may actually decrease as fare increases.


Because of heteroskedasticity in the dataset, standard deviation might not be representative. Adding one std to every 3rd class passenger fare probably triples the cost of that ticket. But adding it to 1st Class passengers is a much more trivial percentage of their total fare.

Might lead to some unrealistic scenarios. Again, adding a std to 3rd class passengers probably makes most of them 2nd class passengers, but we're still holding class constant for all of them.

Finally, this entire approach is relatively experimental, and is not based on any journal articles or previous proofs that I know of.
