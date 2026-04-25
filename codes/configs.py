# =============================================================================
# configs.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Central configuration module that defines all experimental
#            settings, including dataset names, oversampling variant lists,
#            and hyperparameter search spaces for every classifier used in the study.
# =============================================================================

import smote_variants
import math

from sklearn import tree, svm, neural_network, neighbors
from sys import platform

from patch.cascadeForestWrapper import CascadeForestWrapper

class Configurations:
    @staticmethod
    def get_dataset_names():
        dataset_names = ['ant-1.3.csv', 'ant-1.4.csv', 'ant-1.5.csv', 'ant-1.6.csv',
                         'arc.csv',
                         'camel-1.4.csv', 'camel-1.6.csv',
                         'ivy-1.4.csv', 'ivy-2.0.csv',
                         'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv',
                         'log4j-1.0.csv',
                         'pbeans2.csv',
                         'redaktor.csv',
                         'synapse-1.0.csv',
                         'systemdata.csv',
                         'tomcat.csv',
                         'xerces-1.2.csv', 'xerces-1.3.csv']

        return dataset_names

    @staticmethod
    def get_all_smote_variants():
        all_smote_variants = [s.__name__ for s in smote_variants.get_all_oversamplers()]
        all_smote_variants = ['None', 'MAHAKIL'] + all_smote_variants
        return all_smote_variants

    @staticmethod
    def get_all_classifier_params(target_size):

        max_power = int(math.log2(target_size))
        n_estimators_values = [int(2 ** n) for n in range(1, max_power + 1)]
        n_estimators_values_dnn = [int(2**n) for n in range(1, max_power-1)]
        max_depth_values = list(range(2, math.ceil(math.sqrt(target_size)) + 1))

        n_estimators_values = [2, 4 , 8] if len(n_estimators_values) == 0 else n_estimators_values
        n_estimators_values_dnn = [2, 4, 8] if len(n_estimators_values_dnn) == 0 else n_estimators_values_dnn

        model_configs = {
            'RF': {
                'model': None,
                'params': {
                    'num_boost_round': n_estimators_values,
                    'max_depth': max_depth_values,
                    'boosting_type': ['rf'],
                    'num_leaves': [2, 4, 8, 16, 32, 64],
                    'bagging_fraction': [0.5, 0.75, 1.0],
                    'colsample_bytree': [0.5, 0.75, 1.0],
                    'reg_lambda': [0, 1, 10],
                    'reg_alpha': [0, 1, 10],
                    'bagging_freq': [1, 5, 10],
                    'verbose': [-1],
                    'num_threads': [1]
                }
            },

            'GBM': {
                'model': None,
                'params': {
                    'num_boost_round': n_estimators_values,
                    'max_depth': max_depth_values,
                    'boosting_type': ['gbdt', 'dart'],
                    'learning_rate': [1e-3, 1e-2, 0.1, 0.2, 1.0],
                    'num_leaves': [2, 4, 8, 16, 32, 64],
                    'bagging_fraction': [0.5, 0.75, 1.0],
                    'colsample_bytree': [0.5, 0.75, 1.0],
                    'reg_lambda': [0, 1, 10],
                    'reg_alpha': [0, 1, 10],
                    'feature_pre_filter': [False],
                    'verbose': [-1],
                    'num_threads': [1]
                }
            },

            'DF': {
                'model': CascadeForestWrapper,
                'params': {
                    'max_layers': [2, 4, 8, 16],
                    'n_estimators': n_estimators_values,
                    'max_depth': max_depth_values,
                    'learning_rate': [1e-3, 1e-2, 0.1, 0.2, 1.0],
                    'num_leaves': [2, 4, 8, 16, 32, 64],
                    'bagging_fraction': [0.5, 0.75, 1.0],
                    'colsample_bytree': [0.5, 0.75, 1.0],
                    'reg_lambda': [0, 1, 10],
                    'reg_alpha': [0, 1, 10],
                    'feature_pre_filter': [False],
                    'n_jobs': [1],
                    'verbose': [-1],
                }
            },

            'KNN': {
                'model': neighbors.KNeighborsClassifier,
                'params': {
                    'n_neighbors': [int(2 ** n)+1 for n in range(1, 7)],
                    'metric': ['euclidean', 'minkowski'],
                }
            },

            'CART': {
                'model': tree.DecisionTreeClassifier,
                'params': {
                    'max_depth': max_depth_values,
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 5, 10, 20],
                }
            },

            'SVM': {
                'model': svm.SVC,
                'params': {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4, 5],
                    'C': [1e-3, 1e-2, 0.1, 1.0, 10.0, 100],
                    'cache_size': [1000],
                    'tol': [0.1, 1],
                    'max_iter': [100000],
                }
            },

            'ANN': {
                'model': neural_network.MLPClassifier,
                'params': {
                    'hidden_layer_sizes': n_estimators_values_dnn,
                    'activation': ['identity', 'tanh', 'relu'],
                    'alpha': [1e-3, 1e-2, 0.1, 0.5, 1.0, 10],
                    'solver': ['adam'],
                    'early_stopping': [True],
                    'tol': [0.1, 1],
                    'n_iter_no_change': [10],
                }
            },

            'DNN': {
                'model': neural_network.MLPClassifier,
                'params': {
                    'hidden_layer_sizes': n_estimators_values_dnn,
                    'dnn_depth': [2, 4, 8, 16],
                    'activation': ['identity', 'tanh', 'relu'],
                    'alpha': [1e-3, 1e-2, 0.1, 0.5, 1.0, 10],
                    'solver': ['adam'],
                    'early_stopping': [True],
                    'tol': [0.1, 1],
                    'n_iter_no_change': [10],
                }
            },
        }

        return model_configs