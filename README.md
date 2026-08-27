# Replication Package: Deconstructing Oversampling in Software Defect Prediction

This repository contains the complete replication package for the manuscript *"Deconstructing Oversampling in Software Defect Prediction: Algorithm Constraints, Trade-offs, and New Baselines"*. It provides all source code, benchmark datasets, and execution configurations strictly required to fully reproduce the empirical results.

---

## Repository Structure

```
Deconstructing_OVS_in_SDP/
├── codes/
│   ├── run.py                   # Entry point: run the full pipeline from here
│   ├── configs.py               # Experimental configurations (datasets, classifiers, HPO spaces)
│   ├── data_handler.py          # Dataset loading and preprocessing
│   ├── experiments.py           # Core HPO experiment loop
│   ├── optuna_db_helpers.py     # PostgreSQL helpers for Optuna (used specifically for the DF learning algorithm)
│   ├── plots.py                 # Publication figure generation (RQ1-RQ3)
│   ├── runtime_experiment.py    # Oversampling runtime aggregation and RQ4 figure generation
│   ├── stats.py                 # Statistical tests and result aggregation
│   └── patch/
│       ├── cascadeForestWrapper.py   # scikit-learn compatibility patch for deep-forest
│       ├── coste.py                 # COSTE oversampling implementation
│       ├── mahakil.py               # MAHAKIL oversampling implementation
│       └── ros.py                   # ROS (Random OverSampling) implementation
├── datasets/                    # 20 SDP benchmark datasets (CSV)
├── results/                     # Generated at runtime: Parquet + pkl + figures
└── requirements.txt
```

---

## Prerequisites

### 1) Python environment

Python 3.10+ is recommended. Install all dependencies:

```bash
pip install -r codes/requirements.txt
```

The dependencies in requirements.txt are strictly pinned to resolve structural library conflicts. The deepforest architecture requires legacy versions of NumPy and SciPy, which inherently conflict with the deep learning packages (TensorFlow and Keras) utilized by specific OVS techniques. Installing these exact pinned versions is required to ensure a stable execution environment.

---

### 2) R and the WRS package

Statistical tests (Brunner-Munzel and Cliff's delta) are executed using R via `rpy2`.  
This project uses the **`WRS`** package (Wilcox Robust Statistics), specifically:

- `WRS::bprm` — Brunner-Munzel test (p-value)
- `WRS::cid` — Cliff's delta effect size

**Important:** This is **not** `WRS2` (the CRAN package).  
The original `WRS` package from GitHub:
**<https://github.com/nicebread/wrs>** is required.
Follow the installation instructions in the GitHub repository's README for the full set of dependencies.

Verify the installation in the R console:


```R
library(WRS)
```

---

### 3) PostgreSQL (required for parallelized Optuna used in DF classifier only)

The Deep Forest (DF) learning algorithm utilizes shared PostgreSQL-backed Optuna storage, allowing all 20 repetitions to optimize cooperatively within the same study (full parallel HPO).  

All other classifiers (ANN, CART, DNN, GBM, KNN, RF, SVM) run in parallel by repetitions using in-memory Optuna storage, and PostgreSQL is not needed for them.

PostgreSQL must be preconfigured before running experiments that include DF.

#### Required configuration (hardcoded in the source)

| Parameter | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `optuna_db` |
| User | `optuna` |
| Password | `optuna` |

#### Setup steps

```bash
# Create the role and database (run as postgres superuser)
psql -U postgres -c "CREATE ROLE optuna WITH LOGIN PASSWORD 'optuna';"
psql -U postgres -c "CREATE DATABASE optuna_db WITH OWNER = optuna;"
```

The experiment runner automatically calls `OptunaDBHelpers.fast_recreate()` before each DF run to drop and recreate the database, eliminating the need for manual cleanup between runs.

---

### 4) Techniques outside `smote-variants`

Three oversampling techniques are not sourced from the `smote-variants` package and are instead included directly as local modules under `codes/patch/`. No additional installation is required for any of them.

#### 4.1) COSTE

The COSTE technique is included directly in `codes/patch/coste.py`.  
It is a pure-Python reconstruction of the algorithm described in:  
Feng, S., Keung, J., Yu, X., Xiao, Y., Bennin, K.E., Kabir, M.A., Zhang, M. (2021). *"COSTE: Complexity-based OverSampling TEchnique to alleviate the class imbalance problem in software defect prediction."* Information and Software Technology, 129, 106432.

To resolve inconsistencies between the paper's pseudocode and its textual narrative, divergences from a strict implementation of Algorithm 1 (re-ranking per pass, batch-only merging, the explicit binary-class assumption, and the nested subfold AUC fitness evaluation) are documented in the module's docstring.

#### 4.2) MAHAKIL

The MAHAKIL technique is included directly in `codes/patch/mahakil.py`.  
It is a pure-Python re-implementation adapted from the original repository:  
**<https://github.com/ai-se/MAHAKIL_imbalance>**

#### 4.3) ROS

The ROS (Random OverSampling) technique is included directly in `codes/patch/ros.py`.  
It is a thin wrapper around `imblearn.over_sampling.RandomOverSampler`, exposing the same `fit_resample` / `fit_sample` interface used by the other patched techniques.

---

## Running the Pipeline

All stages are managed via a single entry point:

```bash
cd codes

# Stage 1: Run the full HPO experiment loop
python run.py --stage experiment

# Stage 2: Aggregate results and run statistical tests
python run.py --stage stats

# Stage 3: Generate all publication figures (RQ1-RQ4)
python run.py --stage plots
```

> **Note (RQ4):** The RQ4 figure (oversampling runtime distribution) is generated as part of the `plots` stage via `runtime_experiment.py`. It reads pre-collected per-dataset timing data (`time_records`, `cur_oversampler`, `dataset_name`) from `results/exp2/<dataset>.parquet`. This timing data is produced by a separate runtime-measurement run and is not created by `--stage experiment`; ensure `results/exp2/` is populated before running `--stage plots`.

### Output paths (relative to project root)

| Stage | Output                                                                                |
|---|----------------------------------------------------------------------------------------|
| `experiment` | `results/exp/<dataset>__<model>.parquet`                                          |
| `stats` | `results/stats/rq1_result_df.pkl`, `rq2_result_df.pkl`, `rq3_result_df.pkl`             |
| `plots` | `results/figures/RQ1_*.eps`, `RQ2.eps`, `RQ3.eps`, `RQ4.eps`                            |

---

## Experimental Design Summary

| Parameter               | Configuration                                                                                                                                                         |
|:------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Datasets**            | 25 SDP benchmark datasets (located in `datasets/`)                                                                                                                    |
| **OVS Techniques**      | 89 total (87 SMOTE variants + COSTE + MAHAKIL + ROS + No-sampling)                                                                                                    |
| **Learning Algorithms** | ANN, CART, DF, DNN, GBM, KNN, RF, SVM                                                                                                                                 |
| **HPO Budget**          | Maximum 2,000 trials per run (Early stopping patience = 20). *Note: Reduce this parameter substantially to verify pipeline execution flow prior to full replication.* |
| **Repetitions**         | 20 independent repetitions per configuration. *Note: Reduce this parameter for initial pipeline verification.*                                                        |
| **Data Partitioning**   | 70% Training / 30% Hold-out testing (Stratified random sampling)                                                                                                      |
| **CV strategy (HPO)**   | Stratified K-Fold (k = min(minority class size, 10))                                                                                                                  |
| **Metrics reported**    | AUC, MCC, PD (Recall), PF (False Alarm)                                                                                                                               |
| **Statistical test**    | Brunner-Munzel (`WRS::bprm`), α = 0.05                                                                                                                                |
| **Effect size**         | Cliff's delta (`WRS::cid`), threshold = 0.147 (negligible)                                                                                                            |

---

## Datasets

The 25 benchmark datasets are located in the `datasets/` directory as CSV files. Each dataset utilizes static software metrics as independent features. The dependent `bug` variable records the absolute defect count, which is strictly binarized (defects > 0 → 1) to formulate the binary classification task.


---

## Execution Notes and System Constraints

- **Validated execution environment:** The full experimental pipeline was executed and validated using Python 3.11 on an Arch Linux distribution, utilizing an AMD Ryzen™ 7900 processor (12 physical cores, 24 logical threads) equipped with 48 GB of system memory.
- **Floating-point determinism:** The `TF_ENABLE_ONEDNN_OPTS` environment variable is explicitly set to `"0"` at runtime to suppress non-deterministic oneDNN optimizations from TensorFlow and MKL backends.
- **Parallel execution control:** BLAS thread counts are strictly capped at 1 per worker utilizing `threadpoolctl`. This prevents CPU over-subscription and execution deadlocks during parallel evaluation via `ProcessPoolExecutor`.
- **GPU deactivation:** The `CUDA_VISIBLE_DEVICES` environment variable is explicitly set to `"-1"` to disable GPU hardware acceleration. This forces deep learning-based `smote-variants` techniques to execute exclusively on the CPU, preventing CUDA memory allocation conflicts during parallel multiprocessing.
- **Test-set transformations:** The `E_SMOTE` and `ISOMAP_Hybrid` techniques require distinct spatial transformations on the hold-out test set. The `experiments.py` script automatically manages this procedure.
--- 


