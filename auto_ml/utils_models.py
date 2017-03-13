import dill
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier

from sklearn.cluster import MiniBatchKMeans

keras_installed = False
try:
    # Suppress some level of logs
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = 0
    from keras.constraints import maxnorm
    from keras.layers import Dense, Dropout
    from keras.models import Sequential
    from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
    # import keras
    keras_installed = True
except:
    pass

try:
    from auto_ml import utils
except ImportError:
    from ..auto_ml import utils

xgb_installed = False
try:
    import xgboost as xgb
    xgb_installed = True
except NameError:
    pass
except ImportError:
    pass

if xgb_installed:
    import xgboost as xgb


def get_model_from_name(model_name):
    model_map = {
        # Classifiers
        'LogisticRegression': LogisticRegression(n_jobs=-2),
        'RandomForestClassifier': RandomForestClassifier(n_jobs=-2),
        'RidgeClassifier': RidgeClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_jobs=-1),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=10),


        'SGDClassifier': SGDClassifier(n_jobs=-1),
        'Perceptron': Perceptron(n_jobs=-1),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),

        # Regressors
        'LinearRegression': LinearRegression(n_jobs=-2),
        'RandomForestRegressor': RandomForestRegressor(n_jobs=-2),
        'Ridge': Ridge(),
        'ExtraTreesRegressor': ExtraTreesRegressor(n_jobs=-1),
        'AdaBoostRegressor': AdaBoostRegressor(n_estimators=10),
        'RANSACRegressor': RANSACRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(presort=False),

        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),
        'SGDRegressor': SGDRegressor(shuffle=False),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(shuffle=False),

        # Clustering
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=8)
    }

    if xgb_installed:
        model_map['XGBClassifier'] = xgb.XGBClassifier(colsample_bytree=0.8, min_child_weight=5, subsample=1.0, learning_rate=0.1, n_estimators=200, nthread=-1)
        model_map['XGBRegressor'] = xgb.XGBRegressor(nthread=-1, n_estimators=200)

    if keras_installed:
        model_map['DeepLearningClassifier'] = KerasClassifier(build_fn=make_deep_learning_classifier, nb_epoch=100, batch_size=50, verbose=2)
        model_map['DeepLearningRegressor'] = KerasRegressor(build_fn=make_deep_learning_model, nb_epoch=100, batch_size=50, verbose=2)


    return model_map[model_name]

def get_name_from_model(model):
    if isinstance(model, LogisticRegression):
        return 'LogisticRegression'
    if isinstance(model, RandomForestClassifier):
        return 'RandomForestClassifier'
    if isinstance(model, RidgeClassifier):
        return 'RidgeClassifier'
    if isinstance(model, GradientBoostingClassifier):
        return 'GradientBoostingClassifier'
    if isinstance(model, ExtraTreesClassifier):
        return 'ExtraTreesClassifier'
    if isinstance(model, AdaBoostClassifier):
        return 'AdaBoostClassifier'
    if isinstance(model, SGDClassifier):
        return 'SGDClassifier'
    if isinstance(model, Perceptron):
        return 'Perceptron'
    if isinstance(model, PassiveAggressiveClassifier):
        return 'PassiveAggressiveClassifier'
    if isinstance(model, LinearRegression):
        return 'LinearRegression'
    if isinstance(model, RandomForestRegressor):
        return 'RandomForestRegressor'
    if isinstance(model, Ridge):
        return 'Ridge'
    if isinstance(model, ExtraTreesRegressor):
        return 'ExtraTreesRegressor'
    if isinstance(model, AdaBoostRegressor):
        return 'AdaBoostRegressor'
    if isinstance(model, RANSACRegressor):
        return 'RANSACRegressor'
    if isinstance(model, GradientBoostingRegressor):
        return 'GradientBoostingRegressor'
    if isinstance(model, Lasso):
        return 'Lasso'
    if isinstance(model, ElasticNet):
        return 'ElasticNet'
    if isinstance(model, LassoLars):
        return 'LassoLars'
    if isinstance(model, OrthogonalMatchingPursuit):
        return 'OrthogonalMatchingPursuit'
    if isinstance(model, BayesianRidge):
        return 'BayesianRidge'
    if isinstance(model, ARDRegression):
        return 'ARDRegression'
    if isinstance(model, SGDRegressor):
        return 'SGDRegressor'
    if isinstance(model, PassiveAggressiveRegressor):
        return 'PassiveAggressiveRegressor'
    if isinstance(model, MiniBatchKMeans):
        return 'MiniBatchKMeans'
    # Putting these at the end. By this point, we've already determined it is not any of our other models
    if xgb_installed:
        if isinstance(model, xgb.XGBClassifier):
            return 'XGBClassifier'
        if isinstance(model, xgb.XGBRegressor):
            return 'XGBRegressor'

    if keras_installed:
        if isinstance(model, KerasRegressor):
            return 'DeepLearningRegressor'
        if isinstance(model, KerasClassifier):
            return 'DeepLearningClassifier'

# Hyperparameter search spaces for each model
def get_search_params(model_name):
    grid_search_params = {
        'DeepLearningRegressor': {
            'hidden_layers': [
                [1],
                [0.5],
                [2],
                [1, 1],
                [0.5, 0.5],
                [2, 2],
                [1, 1, 1],
                [1, 0.5, 0.5],
                [0.5, 1, 1],
                [1, 0.5, 0.25],
                [1, 2, 1],
                [1, 1, 1, 1],
                [1, 0.66, 0.33, 0.1],
                [1, 2, 2, 1]
            ]
            , 'dropout_rate': [0.0, 0.2, 0.4, 0.6, 0.8]
            # , 'weight_constraint': [0, 1, 3, 5]
            , 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        },
        'DeepLearningClassifier': {
            'hidden_layers': [
                [1],
                [0.5],
                [2],
                [1, 1],
                [0.5, 0.5],
                [2, 2],
                [1, 1, 1],
                [1, 0.5, 0.5],
                [0.5, 1, 1],
                [1, 0.5, 0.25],
                [1, 2, 1],
                [1, 1, 1, 1],
                [1, 0.66, 0.33, 0.1],
                [1, 2, 2, 1]
            ]
            , 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            # , 'nb_epoch': [2, 4, 6, 10, 20]
            # , 'batch_size': [10, 25, 50, 100, 200, 1000]
            # , 'lr': [0.001, 0.01, 0.1, 0.3]
            # , 'momentum': [0.0, 0.3, 0.6, 0.8, 0.9]
            # , 'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            # , 'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            # , 'weight_constraint': [1, 3, 5]
            , 'dropout_rate': [0.0, 0.3, 0.6, 0.8, 0.9]
        },
        'XGBClassifier': {
            'max_depth': [1, 5, 10, 15],
            'learning_rate': [0.1],
            'min_child_weight': [1, 5, 10, 50],
            'subsample': [0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0]
            # 'subsample': [0.5, 1.0]
            # 'lambda': [0.9, 1.0]
        },
        'XGBRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 3, 8, 25],
            # 'lossl': ['ls', 'lad', 'huber', 'quantile'],
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            # 'objective': ['reg:linear', 'reg:gamma'],
            # 'learning_rate': [0.01, 0.1],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76],

        },
        'GradientBoostingRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2', None],
            # 'loss': ['ls', 'lad', 'huber', 'quantile']
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            # 'loss': ['ls', 'lad', 'huber'],
            'loss': ['ls', 'huber'],
            # 'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
            'subsample': [0.5, 0.8, 1.0]
        },
        'GradientBoostingClassifier': {
            'loss': ['deviance', 'exponential'],
            'max_depth': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2', None],
            # 'learning_rate': [0.01, 0.1, 0.25, 0.4, 0.7],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76]

        },

        'LogisticRegression': {
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['newton-cg', 'lbfgs', 'sag']
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'RandomForestClassifier': {
            'criterion': ['entropy', 'gini'],
            'class_weight': [None, 'balanced'],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RandomForestRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RidgeClassifier': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'Ridge': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'ExtraTreesRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [1, 2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'AdaBoostRegressor': {
            'base_estimator': [None, LinearRegression(n_jobs=-1)],
            'loss': ['linear','square','exponential']
        },
        'RANSACRegressor': {
            'min_samples': [None, .1, 100, 1000, 10000],
            'stop_probability': [0.99, 0.98, 0.95, 0.90]
        },
        'Lasso': {
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'ElasticNet': {
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'LassoLars': {
            'positive': [True, False],
            'max_iter': [50, 100, 250, 500, 1000]
        },

        'OrthogonalMatchingPursuit': {
            'n_nonzero_coefs': [None, 3, 5, 10, 25, 50, 75, 100, 200, 500]
        },

        'BayesianRidge': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001]
        },

        'ARDRegression': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'alpha_2': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001],
            'threshold_lambda': [100, 1000, 10000, 100000, 1000000]
        },

        'SGDRegressor': {
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'alpha': [.0000001, .000001, .00001, .0001, .001]
        },

        'PassiveAggressiveRegressor': {
            'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        },

        'SGDClassifier': {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'class_weight': ['balanced', None]
        },

        'Perceptron': {
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'class_weight': ['balanced', None]
        },

        'PassiveAggressiveClassifier': {
            'loss': ['hinge', 'squared_hinge'],
            'class_weight': ['balanced', None],
            'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        }

    }

    return grid_search_params[model_name]


def load_keras_model(file_name):
    from keras.models import load_model

    with open(file_name, 'rb') as read_file:
        base_pipeline = dill.load(read_file)

    keras_file_name = file_name[:-5] + '_keras_deep_learning_model.h5'

    keras_model = load_model(keras_file_name)

    base_pipeline.named_steps['final_model'].model = keras_model

    return base_pipeline


# TODO: figure out later on how to wrap this inside another wrapper or something to make num_cols more dynamic
def make_deep_learning_model(hidden_layers=None, num_cols=None, optimizer='adam', dropout_rate=0.2, weight_constraint=0, shape='standard'):
    if hidden_layers is None:
        hidden_layers = [250]

    model = Sequential()
    # Add a dense hidden layer, with num_nodes = num_cols, and telling it that the incoming input dimensions also = num_cols
    model.add(Dense(num_cols, input_dim=num_cols, activation='relu', init='normal', W_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_cols, activation='relu', init='normal', W_constraint=maxnorm(weight_constraint)))
    model.add(Dense(num_cols, activation='relu', init='normal', W_constraint=maxnorm(weight_constraint)))
    # For regressors, we want an output layer with a single node
    # For classifiers, we'll want to add in some other processing here (like a softmax activation function)
    model.add(Dense(1, init='normal'))

    # The final step is to compile the model
    # TODO: see if we can pass in our own custom loss function here
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


# TODO: eventually take in other parameters to tune
# TODO: eventually take in final_activation and hidden_layer_activations
def make_deep_learning_classifier(hidden_layers=None, num_cols=None, optimizer='adam', dropout_rate=0.2, weight_constraint=0, final_activation='sigmoid'):

    if hidden_layers is None:
        hidden_layers = [1]

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with, it is simply values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in hidden_layers:
        scaled_layers.append(int(num_cols * layer))




    # print('\n\nCreating a deep learning model with the following shape')
    # print('Each item in this list represents a layer, with your data\'s input layer coming before this graph starts')
    # print('And each number represents the number of nodes in that layer')

    # print(hidden_layers)

    model = Sequential()

    model.add(Dense(hidden_layers[0], input_dim=num_cols, init='normal', activation='relu'))

    for layer_size in scaled_layers[1:]:
        model.add(Dense(layer_size, init='normal', activation='relu'))

    model.add(Dense(1, init='normal', activation=final_activation))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'binary_crossentropy', 'poisson'])
    return model
