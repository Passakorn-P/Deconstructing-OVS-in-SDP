# =============================================================================
# plots.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Generates all publication-quality figures for the journal paper.
#            Contains plotting routines for each research question,
#            including scatter plots, convex-hull visualisations, and
#            statistical summary charts using Matplotlib and Seaborn.
# =============================================================================

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

class Plots:

    @staticmethod
    def RQ1(rq1_result_df):

        METRICS = ['AUC', 'MCC', 'PD']
        GLOBAL_LIM = (-1001, 1001)
        MARKER_BASE = 180
        FONT_BASE = 16

        GROUP_1 = ['CART', 'DF', 'GBM', 'RF']
        GROUP_2 = ['ANN', 'DNN', 'KNN', 'SVM']

        _color = plt.cm.tab10.colors
        C_BASE = _color[1]
        C_MEDIAN = _color[2]
        C_BEST = _color[0]

        XLABEL_Y = 0.115
        LEGEND_Y = 0.055
        PANEL_TOP = 0.920
        PANEL_BOT = 0.145

        df = rq1_result_df.reset_index().rename(columns={'index': 'predictor'})
        df[['model', 'ovs']] = df['predictor'].str.rsplit('__', n=1, expand=True)

        def get_sorted_models(algo_group):
            return sorted(algo_group, key=lambda m: m.lower())

        def plot_group(algo_group, group_label, group_id):
            sorted_models = get_sorted_models(algo_group)
            n_rows = len(METRICS)
            n_cols = len(sorted_models)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex='col', sharey='row')

            if n_rows == 1: axes = axes[np.newaxis, :]
            if n_cols == 1: axes = axes[:, np.newaxis]

            for row_i, metric in enumerate(METRICS):
                for col_i, model in enumerate(sorted_models):
                    ax = axes[row_i, col_i]
                    x_col = 'PF__Net'
                    y_col = f'{metric}__Net'

                    model_df = df[df['model'] == model]
                    none_row = model_df[model_df['ovs'].str.upper() == 'NONE']
                    ovs_df = model_df[model_df['ovs'].str.upper() != 'NONE'].copy()

                    bx = int(none_row[x_col].values[0]) if not none_row.empty else 0
                    by = int(none_row[y_col].values[0]) if not none_row.empty else 0

                    best_row = ovs_df.loc[ovs_df[f'{metric}_PF__Net'].idxmax()]
                    sx, sy = int(best_row[x_col]), int(best_row[y_col])

                    mx = int(ovs_df[x_col].median())
                    my = int(ovs_df[y_col].median())


                    pts = ovs_df[[x_col, y_col]].dropna().values
                    if len(pts) > 2:
                        try:
                            hull = ConvexHull(pts)
                            poly = Polygon(pts[hull.vertices], facecolor='#ECF0F1', alpha=0.5,
                                           edgecolor='#BDC3C7', linestyle='--', zorder=1)
                            ax.add_patch(poly)
                        except Exception:
                            pass

                    ax.scatter(ovs_df[x_col], ovs_df[y_col], color=_color[7], alpha=0.25, s=20, zorder=2)
                    ax.annotate('', xy=(sx, sy), xytext=(bx, by),
                                arrowprops=dict(arrowstyle='-|>', color=_color[5], lw=2.5, alpha=0.8, mutation_scale=20))

                    ax.scatter(bx, by, color=C_BASE, marker='o', s=MARKER_BASE, edgecolors='black', zorder=8)
                    ax.scatter(mx, my, color=C_MEDIAN, marker='^', s=MARKER_BASE, edgecolors='black', zorder=9)
                    ax.scatter(sx, sy, color=C_BEST, marker='*', s=MARKER_BASE + 8, edgecolors='black', zorder=10)

                    ax.text(bx, by - 170, f"({bx},{by})", fontsize=FONT_BASE - 4, color=C_BASE,
                            ha='center', fontweight='bold', zorder=1000, clip_on=False)
                    ax.text(sx, sy + 80, f"({sx},{sy})", fontsize=FONT_BASE - 4, color=C_BEST,
                            ha='center', fontweight='bold', zorder=1000, clip_on=False)
                    ax.text(sx, sy + 190, best_row['ovs'], fontsize=FONT_BASE - 4, color=C_BEST,
                            ha='center', fontstyle='italic', fontweight='bold', zorder=1000, clip_on=False)

                    ax.plot([0, GLOBAL_LIM[1]], [0, 0], color='gray', linestyle='dotted', linewidth=2.0, alpha=0.8)
                    ax.plot([0, 0], [0, GLOBAL_LIM[1]], color='gray', linestyle='dotted', linewidth=2.0, alpha=0.8)

                    ax.set_xlim(GLOBAL_LIM)
                    ax.set_ylim(GLOBAL_LIM)
                    max_tick = rq1_result_df.shape[0] - 1
                    ticks = [-max_tick, -(max_tick // 2), 0, max_tick // 2, max_tick]
                    ax.set_xticks(ticks)
                    ax.set_yticks(ticks)
                    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

                    if row_i == n_rows - 1:
                        ax.tick_params(axis='x', bottom=True, labelbottom=True, labelsize=FONT_BASE - 4)
                    else:
                        ax.tick_params(axis='x', bottom=True, labelbottom=False, labelsize=FONT_BASE - 4)

                    if col_i == 0:
                        ylabel = '$p_d$' if metric == 'PD' else metric
                        ax.set_ylabel(ylabel, fontsize=FONT_BASE-1, fontweight='bold')
                        ax.tick_params(axis='y', left=True, labelleft=True, labelsize=FONT_BASE - 4)
                    else:
                        ax.tick_params(axis='y', left=True, labelleft=False, labelsize=FONT_BASE - 4)

                    if row_i == 0:
                        ax.set_title(model, fontsize=FONT_BASE, fontweight='bold')

            fig.text(0.5, XLABEL_Y-0.015, 'Net wins: $p_f$', ha='center', va='center',
                     fontsize=FONT_BASE-1, fontweight='bold')
            fig.text(0.01, 0.5, 'Net wins:', ha='center', va='center',
                     rotation='vertical', fontsize=FONT_BASE-1, fontweight='bold')

            if group_id == 0:
                fig.suptitle('OVS Impact Trajectory', fontsize=FONT_BASE + 5, fontweight='bold', y=1.000)
                fig.text(0.5, 0.95, f'— {group_label} —', ha='center',
                         fontsize=FONT_BASE+1, fontweight='bold')
                dummy = [Line2D([0], [0], color='w', label=' ') for _ in range(3)]
                leg = fig.legend(handles=dummy, loc='lower center',
                                 bbox_to_anchor=(0.5, LEGEND_Y), ncol=3,
                                 fontsize=FONT_BASE, frameon=True,
                                 facecolor='white', edgecolor='white')
                for text in leg.get_texts():
                    text.set_color('white')
                leg.get_frame().set_alpha(0.0)

            elif group_id == 1:
                fig.text(0.5, 0.95, f'— {group_label} —', ha='center',
                         fontsize=FONT_BASE+1, fontweight='bold')
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BASE,
                           markersize=12, markeredgecolor='black', label='No-sampling'),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor=C_MEDIAN,
                           markersize=12, markeredgecolor='black', label='Median OVS'),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor=C_BEST,
                           markersize=18, markeredgecolor='black', label='Optimal OVS'),
                ]
                fig.legend(handles=legend_elements, loc='lower center',
                           bbox_to_anchor=(0.5, LEGEND_Y-0.02), ncol=3,
                           fontsize=FONT_BASE + 1, frameon=True,
                           facecolor='white', edgecolor='#BDC3C7')

            fig.subplots_adjust(
                left=0.08, right=0.99,
                top=PANEL_TOP, bottom=PANEL_BOT,
                hspace=0.08, wspace=0.08
            )

            plt.savefig(f'../results/figures/RQ1_{group_id}.eps', dpi=300, bbox_inches='tight')
            plt.show()

        plot_group(GROUP_1, 'Group 1: Tree and Tree-based Ensemble', 0)
        plot_group(GROUP_2, 'Group 2: Distance-based, Kernel-based, and Neural Networks', 1)

    @staticmethod
    def RQ2(rq2_result_df):
        from scipy.stats import pearsonr
        from matplotlib.legend_handler import HandlerTuple

        GROUP_1 = ['CART', 'DF', 'GBM', 'RF']
        GROUP_2 = ['ANN', 'DNN', 'KNN', 'SVM']

        METRICS_LIST = ['AUC', 'MCC', 'PD']
        THRESHOLD = 0.5
        DOT_SIZE = 220
        FONT_BASE = 24
        FS_TITLE, FS_LABEL, FS_TICKS = FONT_BASE, FONT_BASE - 2, FONT_BASE - 5

        def _prep_ranked_df(df, metric):
            df = df.reset_index().rename(columns={'index': 'predictor'})
            df[['model', 'ovs']] = df['predictor'].str.rsplit('__', n=1, expand=True)
            df_ovs = df[df['ovs'].str.upper() != 'NONE'].copy()
            df_ovs['pm'] = df_ovs[f'{metric}__Net'].rank(pct=True)
            df_ovs['pf'] = df_ovs['PF__Net'].rank(pct=True)
            return df_ovs

        def _filter_group(df_ovs, model_group):
            if model_group is not None:
                return df_ovs[df_ovs['model'].isin(model_group)]
            return df_ovs

        def _prep_group_df(df, metric, model_group):
            return _filter_group(_prep_ranked_df(df, metric), model_group)

        def _above_threshold(summary, threshold):
            return set(summary[
                           (summary['pm'] > threshold) & (summary['pf'] > threshold)
                           ].index)

        def get_common_performers(df, metrics, threshold=0.5, model_group=None):
            common_set = None
            for metric in metrics:
                summary = (
                    _prep_group_df(df, metric, model_group)
                    .groupby('ovs')
                    .agg(pm=('pm', 'median'), pf=('pf', 'median'))
                )
                valid = _above_threshold(summary, threshold)
                common_set = valid if common_set is None else common_set & valid
            return list(common_set) if common_set else []

        def get_partial_performers(df, metrics, common_ovs, threshold=0.5, model_group=None):
            any_set = set()
            for metric in metrics:
                summary = (
                    _prep_group_df(df, metric, model_group)
                    .groupby('ovs')
                    .agg(pm=('pm', 'median'), pf=('pf', 'median'))
                )
                any_set |= _above_threshold(summary, threshold)
            return any_set - set(common_ovs)

        def _plot_panel(ax, df, metric, ovs_colors, partial_ovs,
                        model_group=None, row_idx=0,
                        g1_common_ovs=None, is_pd=False):
            TIER_T = 0.5
            TIER_COLOR = 'steelblue'

            agg = (
                _prep_group_df(df, metric, model_group)
                .groupby('ovs')
                .agg(pct_metric=('pm', 'median'), pct_PF=('pf', 'median'))
                .reset_index()
            )

            in_q2 = agg[(agg['pct_metric'] > TIER_T) & (agg['pct_PF'] > TIER_T)]
            outside = agg[~agg.index.isin(in_q2.index)]
            partial = outside[outside['ovs'].isin(partial_ovs)]
            below = outside[~outside['ovs'].isin(partial_ovs)]

            ax.scatter(below['pct_PF'], below['pct_metric'],
                       color='lightgray', edgecolors='white', linewidths=1.2,
                       zorder=2, s=DOT_SIZE)
            ax.scatter(partial['pct_PF'], partial['pct_metric'],
                       color=TIER_COLOR, edgecolors='white', linewidths=1.2,
                       zorder=2, s=DOT_SIZE)

            ax.plot([TIER_T, TIER_T], [TIER_T, 1.05],
                    color='gray', linestyle='--', lw=2, alpha=0.6, zorder=3)
            ax.plot([TIER_T, 1.05], [TIER_T, TIER_T],
                    color='gray', linestyle='--', lw=2, alpha=0.6, zorder=3)

            for _, row in in_q2.iterrows():
                color = ovs_colors.get(row['ovs'], TIER_COLOR)
                if color == TIER_COLOR:
                    ax.scatter(row['pct_PF'], row['pct_metric'],
                               color=color, edgecolors='white', linewidths=1.2,
                               zorder=4, s=DOT_SIZE)
                else:
                    ax.scatter(row['pct_PF'], row['pct_metric'],
                               color=color, edgecolors='black', linewidths=1.2,
                               zorder=6, s=DOT_SIZE * 3, marker='*')

            if row_idx == 1 and g1_common_ovs is not None:
                g1_rows = agg[agg['ovs'].isin(g1_common_ovs)]
                already_filled = set(in_q2['ovs'])
                g1_not_starred = g1_rows[~g1_rows['ovs'].isin(already_filled)]
                for _, row in g1_not_starred.iterrows():
                    color = ovs_colors.get(row['ovs'], 'dimgray')
                    ax.scatter(
                        row['pct_PF'], row['pct_metric'],
                        facecolors='none', edgecolors=color,
                        linewidths=1.8, zorder=5,
                        s=DOT_SIZE * 3, marker='*'
                    )

            if is_pd and len(agg) > 2:
                try:
                    r, _ = pearsonr(agg['pct_PF'], agg['pct_metric'])
                    ax.text(
                        0.97, 0.97, f'$r = {r:.2f}$',
                        transform=ax.transAxes,
                        fontsize=FS_TICKS + 1, ha='right', va='top',
                        color='steelblue', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                  alpha=0.85, ec='lightgray', lw=0.8)
                    )
                except Exception:
                    pass

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_yticks(np.arange(0, 1.1, 0.1))

            ax.set_xlabel('Median Percentile: $p_f$',
                          fontsize=FS_LABEL - 2, fontweight='bold')
            y_label = 'Median Percentile: $p_d$' if metric == 'PD' else f'Median Percentile: {metric}'
            ax.set_ylabel(y_label,
                          fontsize=FS_LABEL - 2, fontweight='bold')

            metric_label = '$p_d$' if metric == 'PD' else metric
            ax.text(0.01, 0.98, f'{metric_label} vs $p_f$',
                    transform=ax.transAxes,
                    fontsize=FS_TITLE - 2, fontweight='bold',
                    va='top', ha='left')

            ax.tick_params(axis='both', which='major', labelsize=FS_TICKS)
            ax.grid(True, linestyle=':', alpha=0.4)

        groups = [
            (GROUP_1, 'Group 1: Tree and Tree-based Ensemble'),
            (GROUP_2, 'Group 2: Distance-based, Kernel-based, and Neural Networks'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(24, 18))
        fig.subplots_adjust(top=0.88, bottom=0.20, left=0.05,
                            right=0.98, hspace=0.28, wspace=0.15)

        g1_common = get_common_performers(
            rq2_result_df, METRICS_LIST,
            threshold=THRESHOLD, model_group=GROUP_1
        )
        _cmap = plt.get_cmap('tab20')
        ovs_colors = {name: _cmap(i + 1) for i, name in enumerate(sorted(g1_common))}

        for row_idx, (algo_group, group_label) in enumerate(groups):
            common_ovs = get_common_performers(
                rq2_result_df, METRICS_LIST,
                threshold=THRESHOLD, model_group=algo_group
            )
            partial_ovs = get_partial_performers(
                rq2_result_df, METRICS_LIST, common_ovs,
                threshold=THRESHOLD, model_group=algo_group
            )

            for col_idx, metric in enumerate(METRICS_LIST):
                ax = axes[row_idx, col_idx]
                _plot_panel(
                    ax, rq2_result_df, metric, ovs_colors, partial_ovs,
                    model_group=algo_group,
                    row_idx=row_idx,
                    g1_common_ovs=g1_common,
                    is_pd=(metric == 'PD'),
                )

            fig.canvas.draw()
            pos = axes[row_idx, 1].get_position()
            fig.text(
                0.5, pos.y1 + 0.008,
                f'— {group_label} —',
                ha='center', va='bottom',
                fontsize=FS_LABEL + 3, fontweight='bold',
            )

        fig.suptitle('Consistent OVS Techniques Across Learning Algorithms and Metrics',
                     fontsize=FONT_BASE + 8, fontweight='bold', y=0.95)

        LEG_FSIZE = 18
        LEG_TITLE_FSIZE = 18
        LEG_GAP = +0.43

        fig.canvas.draw()
        center_pos = axes[0, 1].get_position()
        LEG_TOP_REF = center_pos.y0 - LEG_GAP
        loc_left, loc_right = 'upper left', 'upper right'

        filled_star = mlines.Line2D([], [], marker='*', color='w',
                                    markerfacecolor='dimgray', markeredgecolor='black',
                                    markersize=18, label='Group 1 Consistent (all metrics)')
        hollow_star = mlines.Line2D([], [], marker='*', color='w',
                                    markerfacecolor='none', markeredgecolor='dimgray',
                                    markeredgewidth=1.8, markersize=18,
                                    label='Consistent in Group 1 only')
        partial_handle = mlines.Line2D([], [], marker='o', color='w',
                                       markerfacecolor='steelblue', markeredgecolor='white',
                                       markersize=12, label='In Quadrant 1 (\u22651 metric, not all 3)')
        other_handle = mlines.Line2D([], [], marker='o', color='w',
                                     markerfacecolor='lightgray', markeredgecolor='white',
                                     markersize=12, label='Outside Quadrant 1 (all metrics)')

        leg1 = fig.legend(
            handles=[filled_star, hollow_star, partial_handle, other_handle],
            loc=loc_left,
            bbox_to_anchor=(0.02, LEG_TOP_REF),
            ncol=2,
            fontsize=LEG_FSIZE, frameon=True, framealpha=0.9, edgecolor='lightgray',
            title='Marker Types', title_fontsize=LEG_TITLE_FSIZE,
            handletextpad=0.6, columnspacing=1.0,
        )
        fig.add_artist(leg1)

        N_COLS = 4
        names_sorted = sorted(g1_common)
        L = len(names_sorted)
        nrows = math.ceil(L / N_COLS)

        order = []
        for c in range(N_COLS):
            for r in range(nrows):
                pos = r * N_COLS + c
                if pos < L:
                    order.append(names_sorted[pos])


        handles_combined = [
            (
                mlines.Line2D([], [], marker='*', color='w',
                              markerfacecolor=ovs_colors[name],
                              markeredgecolor='black', markersize=18),
                mlines.Line2D([], [], marker='*', color='w',
                              markerfacecolor='none',
                              markeredgecolor=ovs_colors[name],
                              markeredgewidth=1.8, markersize=18),
            )
            for name in order
        ]

        leg2 = fig.legend(
            handles=handles_combined,
            labels=order,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
            loc=loc_right,
            bbox_to_anchor=(0.98, LEG_TOP_REF),
            ncol=N_COLS,
            fontsize=LEG_FSIZE, frameon=True, framealpha=0.9, edgecolor='lightgray',
            title='OVS Methods  (★ filled: Group 1 Consistent (all metrics); ☆ Hollow: Consistent in Group 1 only)',
            title_fontsize=LEG_TITLE_FSIZE,
            columnspacing=0.9, handletextpad=0.45,
        )

        plt.savefig(f'../results/figures/RQ2.eps', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def RQ3(rq3_result_df):
        X_LIM = (0.0, 0.7)
        Y_LIM = (0.0, 0.9)
        CMAP = 'cividis'
        PARETO_COL = 'black'
        LABEL_BG = 'white'
        X_PUSH = 0.03
        X_PUSH_STAR = 0.12
        MIN_Y_GAP = 0.07

        FS = dict(label=26, title=46, subtitle=38, axis=34, tick=30, legend=26, cbar=32)

        GROUP_TWEAKS_2D = {
            'DF': dict(x_scale=-2.5, y_nudge=-0.015, side=None, relpos=None),
            'GBM': dict(x_scale=+4.2, y_nudge=-0.37, side=None, relpos=None),
            'KNN': dict(x_scale=-2.65, y_nudge=+0.1, side=None, relpos=None),
            'RF': dict(x_scale=-2.5, y_nudge=-0.2, side=None, relpos=(0.5, 0.0)),
        }

        GROUP_TWEAKS_3D = {
            'DF': dict(x_scale=+1.15, y_nudge=-0.155, side=None, relpos=None),
            'RF': dict(x_scale=+1.60, y_nudge=-0.035, side=None, relpos=None),
        }

        def _pareto_front(costs):
            n = costs.shape[0]
            efficient = np.ones(n, dtype=bool)
            for i, c in enumerate(costs):
                if efficient[i]:
                    efficient[efficient] = (
                            np.any(costs[efficient] < c, axis=1)
                            | (i == np.arange(n))[efficient]
                    )
                    efficient[i] = True
            return efficient

        def _base_pred(series):
            return series.str.split('__').str[0]

        def _build_colour_map(frames):
            tab10 = [mpl.colors.to_hex(c) for c in plt.cm.tab10.colors]
            all_bases = sorted(set().union(*[f['base_pred'].unique() for f in frames]))
            colours = {b: tab10[i] for i, b in enumerate(all_bases)}
            print("=== colour assignment ===")
            for i, b in enumerate(all_bases):
                print(f"  [{i}] {b!r}  →  {tab10[i]}")
            return colours

        def _build_groups(source):
            return (
                source.groupby('base_pred')
                .agg(
                    n=('mcc', 'count'),
                    mcc_mean=('mcc', 'mean'),
                    auc_mean=('auc', 'mean'),
                    pd_mean=('pd', 'mean'),
                    pf_mean=('pf', 'mean'),
                    cy=('pd', 'mean'),
                )
                .reset_index()
            )

        def _make_label(grp, n, suffix):
            return (
                f"{grp['base_pred']} $\\times$ {n} {suffix}\n"
                f"Avg. AUC: {grp['auc_mean']:.3f}\nAvg. MCC: {grp['mcc_mean']:.3f}\n"
                f"Avg. $p_d$: {grp['pd_mean']:.3f}\nAvg. $p_f$: {grp['pf_mean']:.3f}"
            )

        def _annotate_group(ax, members, grp, x_anchor, y_text, ha, va, relpos, facecolor):
            col = base_colours[grp['base_pred']]
            arrow = dict(arrowstyle='-|>', color=col, lw=2.5, relpos=relpos, mutation_scale=22)
            first = True
            for _, pt in members.iterrows():
                xy = (pt['pf'], pt['pd'])
                ax.annotate("", xy=xy, xytext=(x_anchor, y_text), zorder=4, arrowprops=arrow)
                if first:
                    ax.annotate(
                        grp['_label'], xy=xy, xytext=(x_anchor, y_text),
                        fontsize=FS['label'], fontweight='bold', color=col,
                        ha=ha, va=va, multialignment='left', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=facecolor,
                                  edgecolor=col, lw=3.0, alpha=0.95),
                    )
                    first = False

        def _resolve_annotation_geometry(tweak, side):
            if tweak['relpos']:
                rp = tweak['relpos']
                va = 'bottom' if rp[1] == 0.0 else 'center'
                ha = 'center' if rp[0] == 0.5 else ('right' if side == 'left' else 'left')
            else:
                rp = (1.0, 0.5) if side == 'left' else (0.0, 0.5)
                ha = 'right' if side == 'left' else 'left'
                va = 'center'
            return rp, ha, va

        def _style_axes(ax):
            ax.set_xlim(*X_LIM)
            ax.set_xticks(np.arange(X_LIM[0], X_LIM[1] + 0.01, 0.1))
            ax.set_ylim(*Y_LIM)
            ax.set_yticks(np.arange(Y_LIM[0], Y_LIM[1] + 0.01, 0.1))
            ax.set_aspect('equal')
            ax.set_xlabel(
                r'$p_f$: Lower is Better',
                fontsize=FS['axis'], fontweight='bold', labelpad=15,
            )
            ax.set_ylabel(
                r'$p_d$: Higher is Better',
                fontsize=FS['axis'], fontweight='bold', labelpad=15,
            )
            ax.tick_params(axis='both', labelsize=FS['tick'], length=10, width=2)

        df = rq3_result_df.copy().reset_index()
        df['Front_2D'] = _pareto_front(np.column_stack((-df['pd'], df['pf'])))

        f2d = df[df['Front_2D']].sort_values('pf').copy()
        f3d = df[~df['Front_2D']].copy()

        f2d['base_pred'] = _base_pred(f2d['predictor'])
        f3d['base_pred'] = _base_pred(f3d['predictor'])

        mask = (f2d['pf'] < 0.5) & (f2d['pd'] > 0.5)
        if mask.any():
            MCC_THRESH = float(f2d.loc[mask, 'mcc'].mean())
        else:
            MCC_THRESH = float(f2d['mcc'].mean())

        print(f"Computed MCC_THRESH = {MCC_THRESH:.4f} (mean MCC of Pareto front where pf<0.5 and pd>0.5)")

        f3d_res = f3d[f3d['mcc'] > MCC_THRESH].sort_values('pf').copy()
        base_colours = _build_colour_map([f2d, f3d_res])
        v_min, v_max = df['mcc'].min(), df['mcc'].max()
        scatter_kw = dict(cmap=CMAP, vmin=v_min, vmax=v_max)
        f2d_edge = [base_colours[bp] for bp in f2d['base_pred']]

        sns.set_style("whitegrid")
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(32, 20))
        for ax in (ax_left, ax_right):
            _style_axes(ax)

        ax_right.set_ylabel('')
        ax_right.tick_params(axis='y', labelleft=True)

        ax_left.scatter(f3d['pf'], f3d['pd'],
                        c=f3d['mcc'], s=200, alpha=0.3, edgecolors='none', zorder=2, **scatter_kw)
        ax_left.plot(f2d['pf'], f2d['pd'],
                     color=PARETO_COL, lw=4, linestyle='--', zorder=3, label='Pareto front')
        ax_left.scatter(f2d['pf'], f2d['pd'],
                        c=f2d['mcc'], s=700, marker='D', edgecolors=f2d_edge,
                        linewidths=2.5, zorder=4, label='Pareto-optimal', **scatter_kw)

        groups_2d = _build_groups(f2d).sort_values('pd_mean', ascending=False).reset_index(drop=True)
        groups_2d['_label'] = groups_2d.apply(lambda r: _make_label(r, int(r['n']), 'OVS'), axis=1)

        prev_y_L = prev_y_R = -np.inf
        for idx, grp in groups_2d.iterrows():
            bp = grp['base_pred']
            members = f2d[f2d['base_pred'] == bp]
            tweak = GROUP_TWEAKS_2D.get(bp, dict(x_scale=1.0, y_nudge=0.0, side=None, relpos=None))
            side = tweak['side'] or ('left' if idx % 2 == 0 else 'right')
            rp, ha, va = _resolve_annotation_geometry(tweak, side)

            if side == 'left':
                x_anchor = members['pf'].min() - X_PUSH * tweak['x_scale']
                y_text = max(grp['cy'], prev_y_L + MIN_Y_GAP) + tweak['y_nudge']
                prev_y_L = y_text
            else:
                x_anchor = members['pf'].max() + X_PUSH * tweak['x_scale']
                y_text = max(grp['cy'], prev_y_R + MIN_Y_GAP) + tweak['y_nudge']
                prev_y_R = y_text

            _annotate_group(ax_left, members, grp, x_anchor, y_text,
                            ha=ha, va=va, relpos=rp, facecolor=LABEL_BG)

        ax_left.set_title('Pareto-Optimal Configurations',
                          fontsize=FS['subtitle'], fontweight='bold', pad=20)

        mask_grey = f3d['mcc'] <= MCC_THRESH
        mask_hot  = ~mask_grey

        ax_right.scatter(f3d.loc[mask_grey, 'pf'], f3d.loc[mask_grey, 'pd'],
                         c=f3d.loc[mask_grey, 'mcc'], s=200, alpha=0.3,
                         edgecolors='none', zorder=2, **scatter_kw)

        for bp in sorted(f3d.loc[mask_hot, 'base_pred'].unique()):
            mask_bp = mask_hot & (f3d['base_pred'] == bp)
            ax_right.scatter(f3d.loc[mask_bp, 'pf'], f3d.loc[mask_bp, 'pd'],
                             color=base_colours.get(bp, '#aaaaaa'),
                             s=200, alpha=0.6, edgecolors='none', zorder=3)

        ax_right.plot(f2d['pf'], f2d['pd'],
                      color=PARETO_COL, lw=4, linestyle='--', zorder=4, label='Pareto front')
        ax_right.scatter(f2d['pf'], f2d['pd'],
                         c=f2d['mcc'], s=700, marker='D', edgecolors=f2d_edge,
                         linewidths=2.5, zorder=5, **scatter_kw)

        f3d_edge = [base_colours[bp] for bp in f3d_res['base_pred']]
        sc_star = ax_right.scatter(
            f3d_res['pf'], f3d_res['pd'],
            c=f3d_res['mcc'], s=1400, marker='*', edgecolors=f3d_edge, linewidths=3,
            zorder=6,
            label=f'MCC-recovered',  # [updated label]
            **scatter_kw,
        )

        groups_3d = _build_groups(f3d_res).sort_values('cy').reset_index(drop=True)
        groups_3d['_label'] = groups_3d.apply(lambda r: _make_label(r, int(r['n']), 'ovs'), axis=1)

        prev_y = -np.inf
        for _, grp in groups_3d.iterrows():
            bp = grp['base_pred']
            members = f3d_res[f3d_res['base_pred'] == bp]
            tweak = GROUP_TWEAKS_3D.get(bp, dict(x_scale=1.0, y_nudge=0.0, side='right', relpos=None))
            side = tweak['side'] or 'right'
            rp, ha, va = _resolve_annotation_geometry(tweak, side)

            if side == 'left':
                x_anchor = members['pf'].min() - X_PUSH_STAR * tweak['x_scale']
            else:
                x_anchor = members['pf'].max() + X_PUSH_STAR * tweak['x_scale']

            y_text = max(grp['cy'], prev_y + MIN_Y_GAP) + tweak['y_nudge']
            prev_y = y_text
            _annotate_group(ax_right, members, grp, x_anchor, y_text,
                            ha=ha, va=va, relpos=rp, facecolor=LABEL_BG)

        ax_right.set_title(f'MCC-Recovered Configurations (MCC $\geq$ {MCC_THRESH:.3f})',
                           fontsize=FS['subtitle'], fontweight='bold', pad=20)

        seen = {}
        for h, l in zip(*ax_left.get_legend_handles_labels()):
            seen.setdefault(l, h)
        for h, l in zip(*ax_right.get_legend_handles_labels()):
            seen.setdefault(l, h)

        ax_right.legend(
            seen.values(), seen.keys(),
            loc='lower right',
            bbox_to_anchor=(0.975, 0.015),
            fontsize=FS['legend'],
            labelspacing=0.7,
            frameon=True, shadow=True, borderpad=1.2
        )

        fig.suptitle('Pareto-Optimal and MCC-Competitive Configurations',
                     fontsize=FS['title'], fontweight='bold', y=0.99)

        plt.tight_layout(rect=[0, 0.06, 0.91, 0.95])
        plt.subplots_adjust(wspace=0.05)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sc_star, cax=cbar_ax)
        cbar.ax.set_title('MCC', fontsize=FS['cbar'], fontweight='bold', pad=12)
        cbar.ax.tick_params(labelsize=FS['tick'])

        plt.savefig(f'../results/figures/RQ3.eps', dpi=300, bbox_inches='tight')
        plt.show()

        combined = pd.concat([
            f2d.assign(source='pareto'),
            f3d_res.assign(source='rescued')
        ], ignore_index=True)

        all_preds = set(combined['predictor'])
        rest = rq3_result_df[~rq3_result_df.index.isin(all_preds)].copy()
        rest['base_pred'] = rest.index.str.split('__').str[0]
        rest['source'] = 'non_pareto'

        def group_stats(df, label_source, label_clf=None):
            if label_clf:
                sub = df[df['base_pred'] == label_clf]
                return [{'Group': label_source, 'Classifier': label_clf, 'n': len(sub),
                         'MCC Net': sub['mcc'].mean(), 'PD Net': sub['pd'].mean(),
                         'AUC Net': sub['auc'].mean(), 'PF Net': sub['pf'].mean()}]
            else:
                return [{'Group': label_source, 'Classifier': 'All', 'n': len(df),
                         'MCC Net': df['mcc'].mean(), 'PD Net': df['pd'].mean(),
                         'AUC Net': df['auc'].mean(), 'PF Net': df['pf'].mean()}]

        rows = []
        for clf in ['KNN', 'DF', 'RF', 'GBM']:
            rows += group_stats(combined[combined['source'] == 'pareto'], 'Pareto', clf)
        for clf in ['RF', 'DF']:
            rows += group_stats(combined[combined['source'] == 'rescued'], 'Rescued', clf)
        rows += group_stats(rest, 'Non-Pareto/Non-Rescued')


        tbl = pd.DataFrame(rows)
        mapping = {
            'AUC Net': 'Net wins$_{AUC}$',
            'MCC Net': 'Net wins$_{MCC}$',
            'PD Net': 'Net wins$_{p_d}$',
            'PF Net': 'Net wins$_{p_f}$',
        }

        # rename columns
        tbl = tbl.rename(columns=mapping)
        new_cols = list(mapping.values())
        tbl[new_cols] = tbl[new_cols].round(0).astype(int)
        print(tbl.to_markdown(index=False))


        cols = ['source', 'predictor', 'base_pred', 'pd', 'pf', 'mcc', 'auc']
        combined_tbl = combined[cols].sort_values(['source', 'mcc'], ascending=[True, False]).reset_index(drop=True)

        print(f"\nMAX MCC MEAN ON 2D LINE: {MCC_THRESH:.3f}")
        print(f"--- Combined Pareto + Rescued ({len(combined_tbl)} rows) ---")
        print(combined_tbl.sort_values(by='pf').to_markdown(index=False))

        counts = combined_tbl['source'].value_counts().reindex(['pareto', 'rescued']).fillna(0).astype(int)
        print(f"\nCounts: pareto={counts['pareto']}, rescued={counts['rescued']}")