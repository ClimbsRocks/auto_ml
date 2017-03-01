import random

from auto_ml.utils_scoring import advanced_scoring_regressors, advanced_scoring_classifiers
import numpy as np

def make_altered_row(row, permute_or_drop, permute_max, drop_pct):
    row_copy = {}

    for k, v in row.items();

        if permute_or_drop == 'permute':
            up_or_down = 1
            if random.random() > 0.5:
                up_or_down = -1

            altered_val = v + random.random() * permute_max * up_or_down * v
            row_copy[k] = altered_val

        elif permute_or_drop == 'drop':
            altered_val = v
            if random.random() < drop_pct:
                altered_val = None
            row_copy[k] = altered_val

    return row_copy

# TODO:
    # ignore categorical fields
    # make sure this works for classifiers and regressors
def get_ranged_prediction(model, row, column_descriptions, permute_or_drop='permute', permute_max=0.05, drop_pct=0.1, num_samples=100, type_of_estimator='classifier'):
    baseline = model.predict(row)
    all_predictions = []
    for _ in range(num_samples):
        row_copy = make_altered_row(row, permute_or_drop=permute_or_drop, permute_max=permute_max, drop_pct=drop_pct)
        if type_of_estimator == 'classifier':
            all_predictions.append(model.predict_proba(row_copy))
        elif type_of_estimator == 'regressor':
            all_predictions.append(model.predict(row_copy))
        else:
            print('Plase pass in "classifier" or "regressor"')


    if type_of_estimator == 'classifier':

        # If this is a classifier, then make these summaries for each class
        # What is the max, min, avg, etc. for this class?
        summarized_predictions = []
        for idx in range(len(all_predictions[0])):

            class_predictions = [x[idx] for x in all_predictions]
            summarized_predictions.append({
                'max': np.max(class_predictions)
                , 'min': np.min(class_predictions)
                , 'iqr_min': np.percentile(class_predictions, 25)
                , 'iqr_max': np.percentile(class_predictions, 75)
                , 'median': np.median(class_predictions)
                , 'avg': np.mean(class_predictions)
                , 'baseline': baseline
            })

        return summarized_predictions


    else:
        return {
            'max': np.max(all_predictions)
            , 'min': np.min(all_predictions)
            , 'iqr_min': np.percentile(all_predictions, 25)
            , 'iqr_max': np.percentile(all_predictions, 75)
            , 'median': np.median(all_predictions)
            , 'avg': np.mean(all_predictions)
            , 'baseline': baseline
        }


def score_ranged_predictions(ranged_predictions_list, actuals, type_of_estimator='classifier'):
    if regressor_or_classifier == 'regressor':
        scoring_func = advanced_scoring_regressors
    elif regressor_or_classifier == 'classifier':
        scoring_func = advanced_scoring_classifiers
    else:
        print('Plase pass in "classifier" or "regressor"')

    # baseline
    print('Now analyzing our baseline predictions')
    if type_of_estimator == 'classifier':
        baselines = []
        for class_idx in range(len(ranged_predictions_list[0])):
            baselines.append([x[class_idx]['baseline'] for x in ranged_predictions_list])
    else:
        baselines = [x['baseline'] for x in ranged_predictions_list]
    scoring_func(baselines, actuals)

    # median
    print('Now analyzing our median predictions')
    if type_of_estimator == 'classifier':
        medians = []
        for class_idx in range(len(ranged_predictions_list[0])):
            medians.append([x[class_idx]['median'] for x in ranged_predictions_list])
    else:
        medians = [x['median'] for x in ranged_predictions_list]
    scoring_func(medians, actuals)

    # avg
    print('Now analyzing our average predictions')
    if type_of_estimator == 'classifier':
        avgs = []
        for class_idx in range(len(ranged_predictions_list[0])):
            avgs.append([x[class_idx]['avg'] for x in ranged_predictions_list])
    else:
        avgs = [x['avg'] for x in ranged_predictions_list]
    scoring_func(avgs, actuals)




    # segment into quartiles for how tight the iqr is, and score each quartile
    # same for min/max
