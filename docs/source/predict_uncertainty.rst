Predict Uncertainty
=======================



uncertainty_delta
------------------

If using uncertainty_delta_direction='directional', the direction is defined from the perspective of the prediction, since it is only the prediction that we will have at prediction time, not the actual value.
Thus, an uncertainty_delta of -50 means that the prediction has to be less than the actual value by 50 units to be defined as an "uncertain_prediction", and an uncertainty_delta of 50 means that the prediction has to be higher than the predicted value by 50 units to be an uncertain_prediction.

