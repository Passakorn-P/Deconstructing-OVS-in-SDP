# =============================================================================
# runtime_experiment.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Collect the oversampling time of the evaluated OVS techniques.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.stats import gaussian_kde

class Runtime:

    RESULTS_DIR = '../results/exp2/'

    IN_PAPER_DATASETS = [
        'ant-1.3', 'ant-1.4', 'ant-1.5', 'ant-1.6', 'ant-1.7', 'arc', 'camel-1.4',
        'camel-1.6', 'ivy-1.4', 'ivy-2.0', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2',
        'jedit-4.3', 'log4j-1.0', 'log4j-1.1', 'pbeans2', 'redaktor',
        'synapse-1.0', 'synapse-1.1', 'synapse-1.2', 'systemdata', 'tomcat',
        'xerces-1.2', 'xerces-1.3'
    ]

    IN_PAPER_OVS = ['None', 'MAHAKIL', 'ROS', 'COSTE'] + [
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

    ANNOTATE = ['None', 'DSMOTE', 'ROSE', 'Lee', 'MSYM', 'Random-SMOTE', 'SMOTE', 'DSRBF', 'CE-SMOTE', 'NRSBoundary-SMOTE', 'ISOMAP-Hybrid', 'ROS', 'MAHAKIL', 'COSTE']
    DISPLAY_NAME_MAP = {'None': 'No-sampling'}

    @staticmethod
    def _display_name(ovs):
        return Runtime.DISPLAY_NAME_MAP.get(ovs, ovs)

    RQ4_ZONES = {
        'None': 'A', 'KernelADASYN': 'B', 'SOMO': 'B', 'SMOTE-ENN': 'B', 'SMOTE-RSB': 'B', 'PDFOS': 'B', 'DSMOTE': 'B',
        'LLE-SMOTE': 'B', 'A-SUWO': 'B', 'ROSE': 'B', 'Gaussian-SMOTE': 'B', 'Lee': 'B', 'Supervised-SMOTE': 'B',
        'MSYN': 'Balanced-high', 'AND-SMOTE': 'Balanced-high', 'DSRBF': 'B', 'ProWSyn': 'B', 'SMOTE-Cosine': 'B',
        'Random-SMOTE': 'B', 'LVQ-SMOTE': 'B', 'CE-SMOTE': 'B',
        'Selected-SMOTE': 'B', 'SMOTE': 'B', 'SMOTE-TomekLinks': 'B',
        'SN-SMOTE': 'B', 'Assembled-SMOTE': 'B', 'SDSMOTE': 'B',
        'OUPS': 'B', 'MCT': 'B', 'NRSBoundary-SMOTE': 'B',
        'Safe-Level-SMOTE': 'B', 'MSMOTE': 'B', 'ROS': 'A',
        'COSTE': 'A', 'SMOTE-PSO': 'B',
        'ISOMAP-Hybrid': 'B',
    }

    @staticmethod
    def load_runtime_data():
        frames = []
        for file in os.listdir(Runtime.RESULTS_DIR):
            if not file.endswith('.parquet'):
                continue
            dataset_name = file.replace('.parquet', '')
            if dataset_name not in Runtime.IN_PAPER_DATASETS:
                continue
            df = pd.read_parquet(os.path.join(Runtime.RESULTS_DIR, file))
            frames.append(df)

        all_df = pd.concat(frames, ignore_index=True)
        all_df['dataset_name'] = all_df['dataset_name'].str.replace('.csv', '', regex=False)

        all_df['cur_oversampler'] = (
            all_df['cur_oversampler'].astype(str)
            .str.replace(r'(?<!_)_(?!_)', '-', regex=True)
            .str.replace('polynom-fit-SMOTE-bus', 'polynom-fit-SMOTE', regex=False)
        )

        all_df = all_df[all_df['cur_oversampler'].isin(Runtime.IN_PAPER_OVS)].copy()

        return all_df

    @staticmethod
    def compute_stats(all_df):
        all_df = all_df.copy()
        all_df['mean_time'] = all_df['time_records'].apply(lambda x: float(np.mean(x)))
        all_df['std_time'] = all_df['time_records'].apply(lambda x: float(np.std(x)))
        all_df['n_reps'] = all_df['time_records'].apply(len)

        summary = (
            all_df.groupby('cur_oversampler')
            .agg(
                n_datasets=('dataset_name', 'nunique'),
                mean_time_s=('mean_time', 'mean'),
                median_time_s=('mean_time', 'median'),
                std_time_s=('mean_time', 'std'),
                min_time_s=('mean_time', 'min'),
                max_time_s=('mean_time', 'max'),
            )
            .reset_index()
            .rename(columns={'cur_oversampler': 'OVS'})
            .sort_values('mean_time_s')
            .reset_index(drop=True)
        )

        for col in ['mean_time_s', 'median_time_s', 'std_time_s', 'min_time_s', 'max_time_s']:
            summary[col] = summary[col].round(4)

        return summary

    @staticmethod
    def _flatten_alpha(color, alpha, background='white'):

        fg = np.array(to_rgb(color))
        bg = np.array(to_rgb(background))
        return tuple(alpha * fg + (1 - alpha) * bg)

    @staticmethod
    def plot_runtime_kde(summary, save_path='../results/figures/RQ4.eps'):

        summary = summary.copy()
        summary['log_time'] = np.log10(summary['mean_time_s'].to_numpy(dtype=float))
        log_values = summary['log_time'].to_numpy()
        n = len(log_values)

        kde = gaussian_kde(log_values)
        log_grid = np.linspace(log_values.min(), log_values.max(), 1000)
        y_grid = kde(log_grid)
        x_grid = 10 ** log_grid

        fig, ax = plt.subplots(figsize=(8, 4.5))

        fill_color = Runtime._flatten_alpha('steelblue', 0.25, background='white')
        ax.fill_between(x_grid, y_grid, color=fill_color, alpha=1.0,
                         edgecolor='none', zorder=1)
        ax.plot(x_grid, y_grid, color='steelblue', linewidth=1.8, zorder=2,
                label=f'All 89 OVS techniques')

        rug_y = -0.03 * y_grid.max()
        ax.plot(10 ** log_values, np.full_like(log_values, rug_y), '|', color='gray',
                alpha=0.3, markersize=9, markeredgewidth=1.0, zorder=2, clip_on=False)

        for thresh in [0.01,0.1, 1, 10, 100]:
            ax.axvline(thresh, color='gray', linestyle=':', linewidth=1.2, alpha=0.1, zorder=1)

        ax.set_xlim(left=0.0015, right=10 ** log_values.max() + 100)

        is_annot = summary['OVS'].isin(Runtime.ANNOTATE).values
        annot_df = summary.loc[is_annot].sort_values('log_time').reset_index(drop=True)

        force_up = np.zeros(len(annot_df), dtype=bool)
        force_right = np.zeros(len(annot_df), dtype=bool)
        force_left = np.zeros(len(annot_df), dtype=bool)
        force_downleft = np.zeros(len(annot_df), dtype=bool)

        for i in range(len(annot_df)):
            if (annot_df.loc[i, 'log_time'] == annot_df['log_time'].max()) or annot_df.loc[i, 'OVS'] in ['None']:
                force_up[i] = True
            elif annot_df.loc[i, 'OVS'] in ['MAHAKIL']:
                force_right[i] = True
            elif annot_df.loc[i, 'OVS'] in ['ROS']:
                force_left[i] = True
            elif annot_df.loc[i, 'OVS'] in ['ROSE']:
                force_downleft[i] = True

        peak_log_x = log_grid[np.argmax(y_grid)]
        peak_i = int((annot_df['log_time'] - peak_log_x).abs().idxmin())


        r = 20
        for i, row in annot_df.iterrows():
            x = 10 ** row['log_time']
            y = float(kde(row['log_time'])[0])

            if force_up[i]:
                angle_deg = 90
            elif force_right[i]:
                angle_deg = 0
            elif force_left[i]:
                angle_deg = 180
            elif force_downleft[i]:
                angle_deg = 225
            elif i > peak_i:
                angle_deg = 45
            else:
                angle_deg = 135

            rad = np.radians(angle_deg)
            dx, dy = r * np.cos(rad) * 0.5, r * np.sin(rad) * 0.5
            if angle_deg == 90:
                dy = dy * 1.1
            elif (angle_deg == 0):
                dx = dx * 1.75
            elif (angle_deg == 180):
                dx = dx * 1.15
            elif (angle_deg == 225):
                dx = dx * 1.05
                dy = dy * 1.75

            if np.isclose(dx, 0):
                ha = 'center'
            elif dx > 0:
                ha = 'left'
            else:
                ha = 'right'

            if np.isclose(dy, 0):
                va = 'center'
            elif dy > 0:
                va = 'bottom'
            else:
                va = 'top'

            zone = Runtime.RQ4_ZONES.get(row['OVS'])
            dot_color = 'green' if (zone == 'A' or zone is None) else 'crimson'

            ax.scatter([x], [y], color=dot_color, s=100, edgecolors='black',
                       linewidths=1.1, zorder=100)
            ax.vlines(x, rug_y, y, color=dot_color, linestyle='--', linewidth=1.0,
                      alpha=0.7, zorder=3)

            ax.annotate(
                Runtime._display_name(row['OVS']), xy=(x, y), xycoords='data',
                xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va,
                fontsize=9.5, fontweight='bold', color=dot_color,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.8, alpha=0.8, shrinkA=1, shrinkB=4),
                zorder=5,
            )

        n_green = sum(1 for ovs in annot_df['OVS'] if Runtime.RQ4_ZONES.get(ovs) in (None, 'A'))
        n_crimson = len(annot_df) - n_green
        if n_crimson:
            ax.scatter([], [], color='crimson', s=100, edgecolors='black', linewidths=1.1,
                       label=f'OVS techniques highlighted in Table 4')
        if n_green:
            ax.scatter([], [], color='green', s=100, edgecolors='black', linewidths=1.1,
                       label=r'OVS techniques outside $\mathtt{smote}$-$\mathtt{variants}$')


        zone_map = summary['OVS'].map(Runtime.RQ4_ZONES)
        in_zone = zone_map.notna()
        for _, row in summary.loc[in_zone].iterrows():
            x = 10 ** row['log_time']
            y = float(kde(row['log_time'])[0])
            ax.scatter([x], [y], color='gray', s=55, linewidths=1, zorder=4.5, edgecolors='black')

        if in_zone.any():
            ax.scatter([], [], color='gray', s=55, linewidths=1, edgecolors='black',
                       label=f'Other techniques in Table 4')

        ax.set_xscale('log')

        def _plain_num(v, _pos=None):
            v = float(v)
            if v == int(v):
                return f'{int(v)}'
            return f'{v:g}'

        ax.xaxis.set_major_formatter(FuncFormatter(_plain_num))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_xlabel('Oversampling time (s)',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax.set_title("")
        ax.set_ylim(bottom=rug_y * 1.6, top=0.65)
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig, ax

