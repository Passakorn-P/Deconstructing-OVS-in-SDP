# =============================================================================
# experiments.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Core experiment driver.  Iterates over all combinations of
#            datasets, oversampling techniques, and learning algorithms;
#            performs Optuna-based HPO; and stores per-fold performance
#            metrics (AUC, MCC, PD, etc.) to Parquet files for subsequent
#            statistical analysis.
# =============================================================================

import logging
import optuna
import os
import random
import smote_variants
import time
import threading

import lightgbm as lgb
import numpy as np
import pandas as pd

from configs import Configurations
from data_handler import DataHandler
from optuna_db_helpers import OptunaDBHelpers
from threadpoolctl import threadpool_limits

from concurrent.futures import ProcessPoolExecutor
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from patch.mahakil import Mahakil

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
optuna.logging.set_verbosity(optuna.logging.WARN)
logging.getLogger(smote_variants.__name__).setLevel(logging.CRITICAL)

class Experiments:

    @staticmethod
    def init_experiment():
        all_dataset_names = Configurations.get_dataset_names()
        all_smote_variant_names = Configurations.get_all_smote_variants()
        all_model_names = list(Configurations.get_all_classifier_params(1).keys())
        return all_dataset_names, all_smote_variant_names, all_model_names

    @staticmethod
    def main_loop():

        n_trials = 2000
        total_reps = 20

        all_dataset_names, all_smote_variant_names, all_model_names = Experiments.init_experiment()
        start_time = time.time()

        for n_model, cur_model_name in enumerate(all_model_names):
            for n_dataset, dataset_name in enumerate(all_dataset_names):
                cur_dataset = DataHandler.dataset_reader(dataset_name)
                results_df = pd.DataFrame()

                try:
                    initial_results_df = pd.read_parquet(f'../results/exp/{dataset_name.replace(".csv","")}__{cur_model_name}.parquet')
                except:
                    initial_results_df = pd.DataFrame()

                for n_ovs, cur_ovs in enumerate(all_smote_variant_names):
                    end_time = time.time()
                    print(f'{cur_model_name}({n_model+1}/{len(all_model_names)}) \t {dataset_name}({n_dataset+1}/{len(all_dataset_names)}) \t {cur_ovs}({n_ovs+1}/{len(all_smote_variant_names)}) \t elapsed time(s):{end_time - start_time:.2f}')
                    start_time = time.time()
                    if cur_model_name == 'KNN':
                        study_result = Experiments.run_optimization_sk(n_trials, cur_ovs, cur_model_name, dataset_name, 0, cur_dataset)
                        study_results = [study_result for _ in range(total_reps)]
                    elif (cur_model_name == 'GBM') or (cur_model_name == 'RF'):
                        with ProcessPoolExecutor(max_workers=total_reps) as executor:
                            futures = [executor.submit(Experiments.run_optimization_lgb, n_trials, cur_ovs, cur_model_name, dataset_name, rep, cur_dataset) for rep in range(total_reps)]
                            study_results = [future.result() for future in futures]
                    elif cur_model_name == 'DF':
                        OptunaDBHelpers.fast_recreate()
                        Experiments.create_optuna_study(True, False)
                        with ProcessPoolExecutor(max_workers=total_reps) as executor:
                            futures = [executor.submit(Experiments.run_optimization_df, n_trials, cur_ovs, cur_model_name, cur_dataset) for _ in range(total_reps)]
                            study_results = [future.result() for future in futures]
                        completed_trials = np.concatenate(study_results)
                        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=False)[:3]
                        with ProcessPoolExecutor(max_workers=total_reps) as executor:
                            futures = [executor.submit(Experiments.finalize_df, top_trials, dataset_name, cur_dataset, cur_model_name, cur_ovs, rep) for rep in range(total_reps)]
                            study_results = [future.result() for future in futures]
                    else:
                        with ProcessPoolExecutor(max_workers=total_reps) as executor:
                            futures = [executor.submit(Experiments.run_optimization_sk, n_trials, cur_ovs, cur_model_name, dataset_name, rep, cur_dataset) for rep in range(total_reps)]
                            study_results = [future.result() for future in futures]

                    sub_result_df = pd.concat(study_results, ignore_index=True)
                    results_df = pd.concat([results_df, sub_result_df], ignore_index=True)

                    if ('ovs' in initial_results_df) and (cur_ovs in initial_results_df['ovs'].values):
                        initial_results_df = initial_results_df[initial_results_df['ovs'] != cur_ovs]

                results_df = pd.concat([initial_results_df, results_df], ignore_index=True)
                results_df.to_parquet(f'../results/exp/{dataset_name.replace(".csv","")}__{cur_model_name}.parquet', index=False)

    @staticmethod
    def create_optuna_study(is_full_parallel=False, is_load_if_exists=False):
        if is_full_parallel:
            return optuna.create_study(
                    direction='minimize',
                    study_name="ese",
                    pruner=optuna.pruners.HyperbandPruner(),
                    storage = "postgresql://optuna:optuna@localhost/optuna_db",
                    load_if_exists = is_load_if_exists,
            ),  Experiments.no_improvement_callback_factory()

        else:
            return optuna.create_study(
                direction='minimize',
                study_name="ese",
                pruner=optuna.pruners.HyperbandPruner(),
                storage=None,
                load_if_exists=is_load_if_exists,
            ),  Experiments.no_improvement_callback_factory()

    @staticmethod
    def run_optimization_lgb(n_trials, cur_ovs, cur_model_name, dataset_name, rep, cur_dataset):

        X_train_resampled, X_test_scaled, y_train_resampled, y_test, model_config = Experiments.feature_process(cur_dataset, cur_ovs, cur_model_name)

        study, patience_callback = Experiments.create_optuna_study()
        study.optimize(lambda trial: Experiments.optuna_inner_lgb(trial, model_config, X_train_resampled, y_train_resampled), n_trials=n_trials, n_jobs=1, timeout=1000, callbacks=[patience_callback])
        best_trial = study.best_trial
        train_data = lgb.Dataset(X_train_resampled, y_train_resampled)
        model = lgb.train({k: best_trial.params[k] for k in best_trial.params if (k != 'num_boost_round')}, train_data, num_boost_round=best_trial.params['num_boost_round'])
        actual, pred = y_test.values, np.array([0 if x < 0.5 else 1 for x in model.predict(X_test_scaled)])
        metrics = Experiments.compute_metrics(actual, pred)
        result = {'dataset': dataset_name, 'model': cur_model_name, 'ovs': cur_ovs, 'rep': rep, 'actual': actual, 'predict': pred, **metrics}
        results_df = pd.DataFrame([result])

        return results_df

    @staticmethod
    def run_optimization_sk(n_trials, cur_ovs, cur_model_name, dataset_name, rep, cur_dataset):

        X_train_resampled, X_test_scaled, y_train_resampled, y_test, model_config = Experiments.feature_process(cur_dataset, cur_ovs, cur_model_name)

        study, patience_callback = Experiments.create_optuna_study()
        study.optimize(lambda trial: Experiments.optuna_inner_sk(trial, model_config, X_train_resampled, y_train_resampled), n_trials=n_trials, n_jobs=1, timeout=1000, callbacks=[patience_callback])
        best_trial = study.best_trial
        if cur_model_name == 'ANN':
            best_trial.params['hidden_layer_sizes'] = (best_trial.params['hidden_layer_sizes'],)
            model = model_config['model'](**{k: best_trial.params[k] for k in best_trial.params if (k != 'norm') & (k != 'feat') & (k != 'dnn_depth')})
        elif cur_model_name == 'DNN':
            best_trial.params['hidden_layer_sizes'] = (best_trial.params['hidden_layer_sizes'],) * best_trial.params['dnn_depth']
            model = model_config['model'](**{k: best_trial.params[k] for k in best_trial.params if (k != 'norm') & (k != 'feat') & (k != 'dnn_depth')})
        else:
            if 'n_neighbors' in best_trial.params:
                n_samples = len(X_train_resampled)
                best_trial.params['n_neighbors'] = min(best_trial.params['n_neighbors'], n_samples - 1)
            model = model_config['model'](**{k: best_trial.params[k] for k in best_trial.params})

        with threadpool_limits(limits=1, user_api='blas'):
            model.fit(X_train_resampled, y_train_resampled)

        actual, pred = y_test.values, np.array([0 if x < 0.5 else 1 for x in model.predict(X_test_scaled)])

        try:
            metrics = Experiments.compute_metrics(actual, pred)
        except:
            metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'roc': 0, 'mcc': 0, 'gmean': 0, 'false_alarm': 0}

        result = {'dataset': dataset_name, 'model': cur_model_name, 'ovs': cur_ovs, 'rep': rep, 'actual': actual, 'predict': pred, **metrics}
        results_df = pd.DataFrame([result])

        return results_df

    @staticmethod
    def run_optimization_df(n_trials, cur_ovs, cur_model_name, cur_dataset):
        X_train_resampled, X_test_scaled, y_train_resampled, y_test, model_config = Experiments.feature_process(cur_dataset, cur_ovs, cur_model_name)
        study, patience_callback = Experiments.create_optuna_study(is_full_parallel=True, is_load_if_exists=True)
        study.optimize(lambda trial: Experiments.optuna_inner_df(trial, model_config, X_train_resampled, y_train_resampled), n_trials=n_trials, n_jobs=1, timeout=1000, callbacks=[patience_callback])
        completed_trials = [t for t in study.trials if t.value is not None]
        return completed_trials

    @staticmethod
    def finalize_df(top_trials, dataset_name, cur_dataset, cur_model_name, cur_ovs, rep):
        X_train_resampled, X_test_scaled, y_train_resampled, y_test, model_config = Experiments.feature_process(cur_dataset, cur_ovs, cur_model_name)
        best_trial = random.choice(top_trials)
        model = model_config['model'](max_layers=best_trial.params['max_layers'], n_tolerant_rounds=1, verbose=-1, n_jobs=1)
        custom_estimators = [
            LGBMClassifier(**{k: best_trial.params[k] for k in best_trial.params if k != 'max_layers'}),
        ]
        model.set_estimator(custom_estimators)
        model.fit(X_train_resampled, y_train_resampled)
        actual, pred = y_test.values, np.array([0 if x < 0.5 else 1 for x in model.predict(X_test_scaled)])
        metrics = Experiments.compute_metrics(actual, pred)
        result = {'dataset': dataset_name, 'model': cur_model_name, 'ovs': cur_ovs, 'rep': rep, 'actual': actual, 'predict': pred, **metrics}
        results_df = pd.DataFrame([result])
        return results_df

    @staticmethod
    def feature_process(cur_dataset, cur_ovs, cur_model_name):
        X_train, X_test, y_train, y_test = Experiments.create_train_test_split(cur_dataset, test_size=0.3)

        corr_matrix = X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X_train_reduced = X_train.drop(columns=to_drop)
        X_test_reduced = X_test.drop(columns=to_drop)

        v_selector = VarianceThreshold(threshold=0)
        v_selector.fit(X_train_reduced)
        X_train_reduced_2 = v_selector.transform(X_train_reduced)
        X_test_reduced_2 = v_selector.transform(X_test_reduced)

        scaler = MinMaxScaler()
        scaler.fit(X_train_reduced_2)
        X_train_scaled, X_test_scaled = scaler.transform(X_train_reduced_2), scaler.transform(X_test_reduced_2)

        with threadpool_limits(limits=1, user_api='blas'):
            X_train_resampled, y_train_resampled, X_test_scaled = Experiments.static_resampling(X_train_scaled, y_train.values, X_test_scaled, cur_ovs)

        all_model_configs = Configurations.get_all_classifier_params(target_size=len(y_train_resampled))
        model_config = all_model_configs[cur_model_name]



        return X_train_resampled, X_test_scaled, y_train_resampled, y_test, model_config

    @staticmethod
    def optuna_inner_sk(trial, model_config, X_train_resampled, y_train_resampled):

        params = model_config['params']
        search_space = {}

        for param_name, param_info in params.items():
            if type(param_info[0]) == int:
                search_space[param_name] = trial.suggest_int(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == float:
                search_space[param_name] = trial.suggest_float(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == str:
                search_space[param_name] = trial.suggest_categorical(param_name, param_info)

        if ('dnn_depth' in search_space) & ('hidden_layer_sizes' in search_space):
            search_space['hidden_layer_sizes'] = tuple([search_space['hidden_layer_sizes']] * search_space['dnn_depth'])
            search_space = {k: search_space[k] for k in search_space if k != 'dnn_depth'}
        elif ('hidden_layer_sizes' in search_space) & ('dnn_depth' not in search_space):
            search_space['hidden_layer_sizes'] = (search_space['hidden_layer_sizes'],)

        min_class_count = np.min(np.bincount(y_train_resampled))
        n_splits = np.minimum(min_class_count, 10)

        if 'n_neighbors' in search_space:
            n_samples = len(X_train_resampled) * ((n_splits-1) // n_splits)
            search_space['n_neighbors'] = min(search_space['n_neighbors'], n_samples - 1)

        model = model_config['model'](**search_space)

        try:
            with threadpool_limits(limits=1, user_api='blas'):
                cv_results = cross_val_score(model, X_train_resampled, y_train_resampled, cv=StratifiedKFold(n_splits=n_splits), n_jobs=1, scoring='roc_auc')
            error = 1 - cv_results.mean()
            return error
        except:
            return 2.0

    @staticmethod
    def optuna_inner_df(trial, model_config, X_train_resampled, y_train_resampled):
        params = model_config['params']
        search_space = {}

        for param_name, param_info in params.items():
            if param_name == 'max_layers':
                max_layers = trial.suggest_int(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == int:
                search_space[param_name] = trial.suggest_int(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == float:
                search_space[param_name] = trial.suggest_float(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == str:
                search_space[param_name] = trial.suggest_categorical(param_name, param_info)

        model = model_config['model'](max_layers=max_layers, n_tolerant_rounds=1, verbose=-1, n_jobs=1)
        custom_estimators = [
            LGBMClassifier(**search_space),
        ]
        model.set_estimator(custom_estimators)
        min_class_count = np.min(np.bincount(y_train_resampled))
        n_splits = np.minimum(min_class_count, 10)
        cv_results = cross_val_score(model, X_train_resampled, y_train_resampled, cv=StratifiedKFold(n_splits=n_splits), n_jobs=1, scoring='roc_auc')
        error = 1 - cv_results.mean()

        return error

    @staticmethod
    def optuna_inner_lgb(trial, model_config,  X_train_resampled, y_train_resampled):
        params = model_config['params']
        search_space = {}
        for param_name, param_info in params.items():
            if param_name == 'num_boost_round':
                boost = trial.suggest_int(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == int:
                search_space[param_name] = trial.suggest_int(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == float:
                search_space[param_name] = trial.suggest_float(param_name, param_info[0], param_info[-1])
            elif type(param_info[0]) == str:
                search_space[param_name] = trial.suggest_categorical(param_name, param_info)

        train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
        min_class_count = np.min(np.bincount(y_train_resampled))
        n_splits = np.minimum(min_class_count, 10)
        cv_results = lgb.cv(search_space, train_data, num_boost_round=boost, nfold=n_splits, stratified=True, metrics='auc')
        error = 1 - cv_results['valid auc-mean'][-1]

        return error

    @staticmethod
    def no_improvement_callback_factory(patience=20):
        lock = threading.Lock()

        def callback(study: optuna.Study, trial: optuna.Trial):
            with lock:

                if "best_value" not in study.user_attrs:
                    study.set_user_attr("best_value", study.best_value)
                    study.set_user_attr("best_trial_number", trial.number)
                    return

                best_value = study.user_attrs["best_value"]
                best_trial_number = study.user_attrs.get("best_trial_number", trial.number)

                if study.best_value < best_value - 1e-6:
                    study.set_user_attr("best_value", study.best_value)
                    study.set_user_attr("best_trial_number", trial.number)
                else:
                    if trial.number - best_trial_number >= patience:
                        study.stop()

        return callback


    @staticmethod
    def create_train_test_split(cur_dataset, test_size=0.3):
        x = cur_dataset.drop(columns=['bug'])
        y = cur_dataset['bug']
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def static_resampling(X_train, y_train, X_test, cur_ovs):

        if cur_ovs == 'MAHAKIL':
            mahakil = Mahakil()
            X_train_resampled, y_train_resampled = mahakil.fit_resample(X_train, y_train)
        elif cur_ovs == 'None':
            X_train_resampled, y_train_resampled = X_train, y_train
        else:
            cur_ovs_instance = getattr(smote_variants, cur_ovs)(proportion=1.0)
            X_train_resampled, y_train_resampled = cur_ovs_instance.sample(X_train, y_train)
            if cur_ovs == 'E_SMOTE':
                X_test = X_test[:, cur_ovs_instance.mask]
            elif cur_ovs == 'ISOMAP_Hybrid':
                X_test = cur_ovs_instance.isomap.transform(X_test)

        return X_train_resampled, y_train_resampled, X_test

    @staticmethod
    def compute_metrics(actual, predict):

        actual = np.array(actual)
        predict = np.array(predict)

        tn, fp, fn, tp = confusion_matrix(actual, predict).ravel()

        recall = recall_score(actual, predict, zero_division=0)
        mcc = matthews_corrcoef(actual, predict)
        roc = roc_auc_score(actual, predict)
        false_alarm = fp / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            'auc': roc,
            'mcc': mcc,
            'pd': recall,
            'pf': false_alarm,
        }

        return metrics
