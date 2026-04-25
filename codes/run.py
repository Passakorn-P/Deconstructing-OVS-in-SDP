# =============================================================================
# run.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Unified entry point for the entire experimental pipeline.
#            Provides three runnable stages via the --stage flag:
#              experiment  — run the full HPO experiment loop and save
#                            per-fold results to ../results/exp/
#              stats       — aggregate Parquet results, run statistical
#                            tests (Brunner-Munzel, Cliff's delta), and
#                            cache RQ dataframes to ../results/stats/*.pkl
#              plots       — load cached RQ dataframes and render all
#                            figures for the paper
# Usage    :
#   python run.py --stage experiment
#   python run.py --stage stats
#   python run.py --stage plots
# =============================================================================

import argparse
import pickle
import os
import sys
import time

from stats import Stats
from plots import Plots

# ---------------------------------------------------------------------------
# Ensure the codes/ directory is on sys.path so sibling modules resolve
# correctly regardless of the working directory the caller uses.
# ---------------------------------------------------------------------------
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

def run_experiment():
    print("STAGE: experiment")

    os.makedirs(os.path.join(_cur_dir, '..', 'results', 'exp'), exist_ok=True)

    from experiments import Experiments

    t0 = time.time()
    Experiments.main_loop()
    elapsed = time.time() - t0

    print(f"\nExperiment finished in {elapsed / 3600:.2f} h  ({elapsed:.0f} s)")


def run_stats(is_reset: bool = True):
    """Aggregate results and run statistical tests (Stage 2).

    Parameters
    ----------
    is_reset : bool
        True  → read raw Parquet files and recompute everything.
        False → load previously cached .pkl files.
    """
    print(f"STAGE: stats  (reset={is_reset})")

    os.makedirs(os.path.join(_cur_dir, '..', 'results', 'stats'), exist_ok=True)

    t0 = time.time()
    rq1, rq2, rq3 = Stats.get_initial_data(is_reset=is_reset)
    elapsed = time.time() - t0

    print(f"\nStatistical analysis finished in {elapsed:.1f} s")

    return rq1, rq2, rq3


def run_plots():
    """Render publication figures from cached RQ dataframes (Stage 3).

    If any of the required cached RQ dataframes are missing, the stats
    stage is automatically executed first.
    """
    print("STAGE: plots")

    os.makedirs(os.path.join(_cur_dir, '..', 'results', 'figures'), exist_ok=True)
    results_dir = os.path.join(_cur_dir, '..', 'results', 'stats')

    pkl_paths = {
        1: os.path.join(results_dir, 'rq1_result_df.pkl'),
        2: os.path.join(results_dir, 'rq2_result_df.pkl'),
        3: os.path.join(results_dir, 'rq3_result_df.pkl'),
    }

    if not all(os.path.exists(p) for p in pkl_paths.values()):
        print("[INFO] One or more cached RQ files not found — running stats stage first …")
        run_stats()

    with open(pkl_paths[1], 'rb') as f:
        rq1_df = pickle.load(f)
    print("Generating RQ1 plot …")
    Plots.RQ1(rq1_df)
    print("  RQ1 done.")

    with open(pkl_paths[2], 'rb') as f:
        rq2_df = pickle.load(f)
    print("Generating RQ2 plot …")
    Plots.RQ2(rq2_df)
    print("  RQ2 done.")

    with open(pkl_paths[3], 'rb') as f:
        rq3_df = pickle.load(f)
    print("Generating RQ3 plot …")
    Plots.RQ3(rq3_df)
    print("  RQ3 done.")

    print("\nAll figures generated.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run.py',
        description=(
            "Entry point for the 'Deconstructing Oversampling in SDP' "
            "experimental pipeline.\n\n"
            "Stages:\n"
            "  experiment  Run HPO loop and save raw results to results/exp/\n"
            "  stats       Aggregate results and run statistical tests\n"
            "  plots       Render publication figures from cached RQ dataframes\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--stage',
        choices=['experiment', 'stats', 'plots'],
        required=True,
        help='Pipeline stage to execute.',
    )


    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.stage == 'experiment':
        run_experiment()

    elif args.stage == 'stats':
        run_stats()

    elif args.stage == 'plots':
        run_plots()

if __name__ == '__main__':
    main()

