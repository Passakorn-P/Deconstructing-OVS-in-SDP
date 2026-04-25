# =============================================================================
# stats.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Statistical analysis module. Wraps R-based non-parametric
#            tests (Brunner-Munzel via WRS::bprm) and effect-size measures
#            (Cliff's delta via WRS::cid) through rpy2, and aggregates
#            experimental results from Parquet files for hypothesis testing
#            across all research questions.
# =============================================================================

import concurrent
import itertools
import os
import pickle

import pandas as pd
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

class Stats:

    @staticmethod
    def brunner_munzel_r(dist_a, dist_b):
        """Run R WRS::bprm (Brunner-Munzel) on two distributions, return p-value."""
        numpy2ri.activate()
        wrs = importr('WRS')
        r_a = ro.FloatVector(dist_a.tolist())
        r_b = ro.FloatVector(dist_b.tolist())
        try:
            result = wrs.bprm(r_a, r_b)
            p_value = float(result.rx2('p.value')[0])
        except Exception:
            p_value = 1.0
        return p_value

    @staticmethod
    def cliff_delta_r(dist_a, dist_b):
        """Run R WRS::cid (Cliff's delta) on two distributions, return delta value."""
        numpy2ri.activate()
        wrs = importr('WRS')
        r_a = ro.FloatVector(dist_a.tolist())
        r_b = ro.FloatVector(dist_b.tolist())
        try:
            result = wrs.cid(r_a, r_b)
            d = float(result.rx2('CI')[0])  # or result.rx2('dhat')[0] depending on WRS version
        except Exception:
            d = 0.0
        return d

    @staticmethod
    def get_initial_data(is_reset=True):
        if is_reset:
            all_perf = pd.concat([
                pd.read_parquet(f'../results/exp/{file}')
                for file in os.listdir('../results/exp/')
                if file.endswith('.parquet')
            ])

            all_perf['model'] = all_perf['model'].astype(str).str.replace(r'(?<!_)_(?!_)', '-', regex=True)
            all_perf['ovs'] = all_perf['ovs'].astype(str).str.replace(r'(?<!_)_(?!_)', '-', regex=True).str.replace('polynom-fit-SMOTE-bus', 'polynom-fit-SMOTE')

            in_paper_ovs = ['None', 'MAHAKIL'] + [
                "SMOTE", "SMOTE-TomekLinks", "SMOTE-ENN", "Borderline-SMOTE1", "Borderline-SMOTE2", "AHC", "LLE-SMOTE", "cluster-SMOTE",
                "distance-SMOTE", "ADASYN", "SMMO", "polynom-fit-SMOTE", "Stefanowski", "ADOMS", "Safe-Level-SMOTE", "MSMOTE",
                "ISOMAP-Hybrid", "DE-oversampling", "CE-SMOTE", "Edge-Det-SMOTE", "SMOBD", "SUNDO", "MSYN", "LN-SMOTE", "CBSO", "E-SMOTE",
                "Random-SMOTE", "NDO-sampling", "DSRBF", "SVM-balance", "TRIM-SMOTE", "SMOTE-RSB", "DBSMOTE", "ASMOBD", "SN-SMOTE",
                "ProWSyn", "SL-graph-SMOTE", "NRSBoundary-SMOTE", "LVQ-SMOTE", "SOI-CJ", "Assembled-SMOTE", "ISMOTE", "ROSE", "SMOTE-OUT",
                "SMOTE-Cosine", "Selected-SMOTE", "MWMOTE", "PDFOS", "IPADE-ID", "RWO-sampling", "NEATER", "SDSMOTE", "DSMOTE", "G-SMOTE",
                "NT-SMOTE", "SSO", "Supervised-SMOTE", "DEAGO", "Gazzah", "MCT", "ADG", "SMOTE-IPF", "KernelADASYN", "MOT2LD", "V-SYNTH",
                "Lee", "SPY", "SMOTE-PSOBAT", "OUPS", "SMOTE-D", "MDO", "VIS-RST", "GASMOTE", "A-SUWO", "SMOTE-FRST-2T", "AND-SMOTE",
                "SMOTE-PSO", "CURE-SMOTE", "SOMO", "NRAS", "Gaussian-SMOTE", "CCR", "ANS", "AMSCO", "kmeans-SMOTE"
            ]

            mask = all_perf['ovs'].isin(in_paper_ovs)
            all_perf = all_perf.loc[mask].copy()

            all_perf['predictor'] = all_perf['model'] + "__" + all_perf['ovs']

            result_df_auc = Stats.compute_wtl_parallel_chunked(all_perf, 'auc', higher_is_better=True)
            result_df_mcc = Stats.compute_wtl_parallel_chunked(all_perf, 'mcc', higher_is_better=True)
            result_df_pd  = Stats.compute_wtl_parallel_chunked(all_perf, 'pd',  higher_is_better=True)
            result_df_pf  = Stats.compute_wtl_parallel_chunked(all_perf, 'pf', higher_is_better=False)

            for df in [result_df_auc, result_df_mcc, result_df_pd, result_df_pf]:
                df.sort_values(by='Net', ascending=False, inplace=True)

            result_df_auc_renamed = result_df_auc.rename(columns={'Win': 'AUC__Win', 'Tie': 'AUC__Tie', 'Loss': 'AUC__Loss', 'Net': 'AUC__Net'})
            result_df_mcc_renamed = result_df_mcc.rename(columns={'Win': 'MCC__Win', 'Tie': 'MCC__Tie', 'Loss': 'MCC__Loss', 'Net': 'MCC__Net'})
            result_df_pd_renamed  = result_df_pd.rename( columns={'Win': 'PD__Win',  'Tie': 'PD__Tie',  'Loss': 'PD__Loss',  'Net': 'PD__Net'})
            result_df_pf_renamed  = result_df_pf.rename( columns={'Win': 'PF__Win',  'Tie': 'PF__Tie',  'Loss': 'PF__Loss',  'Net': 'PF__Net'})

            auc_pf_result_df = (result_df_auc + result_df_pf).rename(columns={'Win': 'AUC_PF__Win', 'Tie': 'AUC_PF__Tie', 'Loss': 'AUC_PF__Loss', 'Net': 'AUC_PF__Net'})
            mcc_pf_result_df = (result_df_mcc + result_df_pf).rename(columns={'Win': 'MCC_PF__Win', 'Tie': 'MCC_PF__Tie', 'Loss': 'MCC_PF__Loss', 'Net': 'MCC_PF__Net'})
            pd_pf_result_df  = (result_df_pd  + result_df_pf).rename(columns={'Win': 'PD_PF__Win',  'Tie': 'PD_PF__Tie',  'Loss': 'PD_PF__Loss',  'Net': 'PD_PF__Net'})

            rq1_result_df = pd.concat([
                mcc_pf_result_df['MCC_PF__Net'], result_df_mcc_renamed['MCC__Net'],
                pd_pf_result_df['PD_PF__Net'],   result_df_pd_renamed['PD__Net'],
                auc_pf_result_df['AUC_PF__Net'], result_df_auc_renamed['AUC__Net'],
                result_df_pf_renamed['PF__Net'],
            ], axis=1)

            rq2_result_df = pd.concat([
                mcc_pf_result_df['MCC_PF__Net'], result_df_mcc_renamed['MCC__Win'], result_df_mcc_renamed['MCC__Loss'], result_df_mcc_renamed['MCC__Net'],
                pd_pf_result_df['PD_PF__Net'],   result_df_pd_renamed['PD__Win'],   result_df_pd_renamed['PD__Loss'],   result_df_pd_renamed['PD__Net'],
                auc_pf_result_df['AUC_PF__Net'], result_df_auc_renamed['AUC__Win'], result_df_auc_renamed['AUC__Loss'], result_df_auc_renamed['AUC__Net'],
                result_df_pf_renamed['PF__Net'],  result_df_pf_renamed['PF__Win'],   result_df_pf_renamed['PF__Loss'],
            ], axis=1)

            rq3_result_raw = (
                all_perf.groupby(['predictor'])[['mcc', 'auc', 'pd', 'pf']]
                        .mean().reset_index().set_index('predictor')
            )

            with open('../results/stats/rq1_result_df.pkl', 'wb') as f: pickle.dump(rq1_result_df, f)
            with open('../results/stats/rq2_result_df.pkl', 'wb') as f: pickle.dump(rq2_result_df, f)
            with open('../results/stats/rq3_result_df.pkl', 'wb') as f: pickle.dump(rq3_result_raw, f)

        else:
            with open('../results/stats/rq1_result_df.pkl', 'rb') as f: rq1_result_df  = pickle.load(f)
            with open('../results/stats/rq2_result_df.pkl', 'rb') as f: rq2_result_df  = pickle.load(f)
            with open('../results/stats/rq3_result_df.pkl', 'rb') as f: rq3_result_raw = pickle.load(f)

        rq3_result_raw = rq3_result_raw.join(rq1_result_df[['MCC__Net', 'PD__Net', 'AUC__Net', 'PF__Net']])

        return rq1_result_df, rq2_result_df, rq3_result_raw

    @staticmethod
    def compute_wtl_parallel_chunked(df, metric, higher_is_better=True):
        avg_df = df.groupby(['dataset', 'predictor'])[metric].mean().reset_index()
        predictors = avg_df['predictor'].unique()
        pairs = list(itertools.combinations(predictors, 2))

        num_cores = os.cpu_count() or 4
        chunk_size = max(1, len(pairs) // num_cores)
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

        final_stats = {p: {'Win': 0, 'Tie': 0, 'Loss': 0} for p in predictors}

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [
                executor.submit(Stats.process_chunk, chunk, avg_df, metric, higher_is_better)
                for chunk in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                chunk_result = future.result()
                for predictor, scores in chunk_result.items():
                    final_stats[predictor]['Win']  += scores['Win']
                    final_stats[predictor]['Tie']  += scores['Tie']
                    final_stats[predictor]['Loss'] += scores['Loss']

        result_df = pd.DataFrame.from_dict(final_stats, orient='index')
        result_df['Net'] = result_df['Win'] - result_df['Loss']
        return result_df.sort_values(by='Net', ascending=False)

    @staticmethod
    def process_chunk(pairs_chunk, avg_df, metric, higher_is_better):
        numpy2ri.activate()
        wrs = importr('WRS')

        local_stats = {}

        for p_a, p_b in pairs_chunk:
            for p in (p_a, p_b):
                if p not in local_stats:
                    local_stats[p] = {'Win': 0, 'Tie': 0, 'Loss': 0}

            dist_a = avg_df[avg_df['predictor'] == p_a][metric].values
            dist_b = avg_df[avg_df['predictor'] == p_b][metric].values

            if len(dist_a) == 0 or len(dist_b) == 0:
                continue

            if np.array_equal(dist_a, dist_b):
                local_stats[p_a]['Tie'] += 1
                local_stats[p_b]['Tie'] += 1
                continue

            r_a = numpy2ri.py2rpy(dist_a)
            r_b = numpy2ri.py2rpy(dist_b)

            try:
                if np.abs(dist_a - dist_b).sum() > 1e-6:
                    bprm_result = Stats.convert_to_python_dict(wrs.bprm(r_a, r_b))
                    p_value = bprm_result['p.value']
                else:
                    p_value = 1.0
            except Exception:
                p_value = 1.0

            try:
                if np.abs(dist_a - dist_b).sum() > 1e-6:
                    cid_result = Stats.convert_to_python_dict(wrs.cid(r_a, r_b))
                    d = cid_result['d']
                else:
                    d = 0.0
            except Exception:
                d = 0.0

            if (not np.isnan(p_value)) and (p_value < 0.05) and (abs(d) >= 0.147):
                mean_a, mean_b = np.mean(dist_a), np.mean(dist_b)
                if mean_a == mean_b:
                    local_stats[p_a]['Tie'] += 1
                    local_stats[p_b]['Tie'] += 1
                elif (mean_a > mean_b) == higher_is_better:
                    local_stats[p_a]['Win'] += 1
                    local_stats[p_b]['Loss'] += 1
                else:
                    local_stats[p_b]['Win'] += 1
                    local_stats[p_a]['Loss'] += 1
            else:
                local_stats[p_a]['Tie'] += 1
                local_stats[p_b]['Tie'] += 1

        return local_stats

    @staticmethod
    def convert_to_python_dict(test_result):
        python_dict = {}
        for key, r_obj in test_result.items():
            try:
                python_value = list(r_obj) if len(r_obj) > 1 else float(r_obj[0])
            except:
                try:
                    python_value = list(r_obj) if len(r_obj) > 1 else str(r_obj[0])
                except:
                    python_value = None
            python_dict[key] = python_value
        return python_dict