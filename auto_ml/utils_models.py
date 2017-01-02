from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier

from sklearn.cluster import MiniBatchKMeans

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
        # 'DeepLearningRegressor': KerasRegressor(build_fn=make_deep_learning_model, nb_epoch=10, batch_size=10, verbose=1),
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
    if isinstance(model, xgb.XGBClassifier):
        return 'XGBClassifier'
    if isinstance(model, xgb.XGBRegressor):
        return 'XGBRegressor'

    # if isinstance(model, KerasRegressor):
    #     return 'KerasRegressor'

# Hyperparameter search spaces for each model
def get_search_params(model_name):
    grid_search_params = {
        # 'DeepLearningRegressor': {
        #     # 'shape': ['triangle_left', 'triangle_right', 'triangle_cuddles', 'long', 'long_and_wide', 'standard']
        #     'dropout_rate': [0.0, 0.2, 0.4, 0.6, 0.8]
        #     , 'weight_constraint': [0, 1, 3, 5]
        #     , 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        # },
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


