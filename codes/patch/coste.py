"""
COSTE: Complexity-based OverSampling TEchnique
Publication reconstruction of Feng, S., Keung, J., Yu, X., Xiao, Y.,
Bennin, K.E., Kabir, M.A., Zhang, M. (2021). "COSTE: Complexity-based
OverSampling TEchnique to alleviate the class imbalance problem in software
defect prediction." Information and Software Technology, 129, 106432.

`_generate_synthetic` below is a direct line-by-line mapping to Algorithm 1
of the paper, preserved exactly with its line-number comments. Two points
where Algorithm 1's pseudocode is ambiguous/inconsistent with the
paper's narrative (Section 4.4) are resolved inside that function and
documented at the exact line they occur (see D1, D2 below). All other
divergences from a naive/literal reading are as given below:

  D1. Re-ranking inside the repeat loop (see Line 12 comment below).
      Section 4.4's narrative explicitly states: "we insert the newly
      generated instances into the original dataset and repeat the above
      phases from 4.3 to 4.4 until the desired number is reached" — Section
      4.3 is "Calculating complexity and rank," so re-ranking on each pass
      is textually required. Algorithm 1's Line 12 says only "repeat Line
      5," which if followed literally (no re-ranking) produces exact
      duplicate synthetic instances from pass 2 onward and destroys the
      complexity-adjacency property that is COSTE's core mechanism.
  D2. Merging only the newly-generated `batch` into N_min per pass (see
      Line 11/18 comments below), not the cumulative Array_syn. Algorithm
      1's literal text says "merge N_min and Array_syn," but Array_syn is
      cumulative across passes — literally re-merging it every pass would
      re-insert already-merged synthetics, causing compounding/exponential
      duplication after 2+ passes. Section 4.4's narrative ("insert the
      newly generated instances") supports isolating only the current
      pass's batch.
  D3. Binary-class assumption made explicit in fit_resample — the paper's
      SDP context (defective vs. non-defective) is inherently binary; no
      multi-class oversampling logic is implied anywhere in the paper.
  D4. Nested subfold AUC evaluation for DE fitness in _fitness_auc,
      following Section 5.7's description of the inner validation
      procedure, rather than a single random train/val split.
"""

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


class COSTE:
    """
    Parameters
    ----------
    N, G, F, CR, bound_min, bound_max : DE hyperparameters, paper Table 2
        defaults: N=200, G=20, F=0.3, CR=0.9, bounds=[-1, 1]
    clf : sklearn-like classifier with fit/predict_proba, default KNN
        Used as the inner-loop AUC evaluator for DE fitness (Section 4.3).
        Not the outer classifier for final evaluation — train your own
        downstream classifier separately on the output of fit_resample.
    inner_folds : int, default=5
        Number of subfolds for the nested DE weight-selection procedure
        described in Section 5.7 (D4).
    random_state : int or None
    """

    def __init__(self, N=200, G=20, F=0.3, CR=0.9,
                 bound_min=-1.0, bound_max=1.0,
                 clf=None, inner_folds=5, random_state=None):
        self.N = N
        self.G = G
        self.F = F
        self.CR = CR
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.clf = clf if clf is not None else KNeighborsClassifier()
        if hasattr(self.clf, "n_jobs"):
            self.clf.set_params(n_jobs=1)
        self.inner_folds = inner_folds
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_resample(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.reset_index(drop=True)
        y = np.array(y)

        # D3: explicit binary assumption. COSTE's SDP context is always
        # defective (minority) vs. non-defective (majority).
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"COSTE is defined for binary classification only "
                f"(defective vs. non-defective). Got {len(classes)} classes."
            )

        counts = Counter(y)
        minority_cls = min(counts, key=counts.get)
        majority_cls = max(counts, key=counts.get)

        if counts[minority_cls] == counts[majority_cls]:
            return X.values, y

        # Algorithm 1, Line 2: apply min-max normalization to the dataset
        X_norm, mins, maxs = self._minmax_normalize(X)

        idx_min = np.where(y == minority_cls)[0]
        idx_maj = np.where(y == majority_cls)[0]
        X_min_norm = X_norm.iloc[idx_min].reset_index(drop=True)
        X_maj_norm = X_norm.iloc[idx_maj].reset_index(drop=True)

        # Section 4.1/4.3: DE finds optimal complexity weights, using AUC
        # as the fitness signal (Section 4.3, explicit paper statement)
        weights = self._optimize_weights(
            X_min_norm, X_maj_norm,
            np.full(len(idx_min), minority_cls),
            np.full(len(idx_maj), majority_cls),
        )

        # Algorithm 1: generate synthetic instances (see method below)
        with threadpool_limits(limits=1):
            N_bal_norm = self._generate_synthetic(X_min_norm, X_maj_norm, weights)

        ranges = (maxs - mins).replace(0, 1.0)
        X_bal_raw = N_bal_norm * ranges.values + mins.values
        X_bal_raw = pd.DataFrame(X_bal_raw, columns=X.columns)

        n_min_final = len(N_bal_norm) - len(idx_maj)
        y_bal = np.concatenate([
            np.full(n_min_final, minority_cls),
            np.full(len(idx_maj), majority_cls),
        ])

        return X_bal_raw.values, y_bal

    def fit_sample(self, X, y):
        return self.fit_resample(X, y)

    def sample(self, X, y):
        return self.fit_resample(X, y)

    # ------------------------------------------------------------------
    # Eq. 8: min-max normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _minmax_normalize(X):
        mins = X.min()
        maxs = X.max()
        ranges = (maxs - mins).replace(0, 1.0)
        X_norm = (X - mins) / ranges
        return X_norm, mins, maxs

    # ------------------------------------------------------------------
    # Section 3.3 (DE mechanics) + Section 4.3 (AUC fitness)
    # ------------------------------------------------------------------
    def _optimize_weights(self, X_min_norm, X_maj_norm, y_min, y_maj):
        d = X_min_norm.shape[1]
        lo, hi = self.bound_min, self.bound_max
        rng = np.random.RandomState(self.random_state)

        # Eq. 1-4: initialization
        pop = lo + rng.rand(self.N, d) * (hi - lo)
        fit_vals = np.array([
            self._fitness_auc(w, X_min_norm, X_maj_norm, y_min, y_maj)
            for w in pop
        ])

        for gen in range(self.G):
            for i in range(self.N):
                idxs = [j for j in range(self.N) if j != i]
                r1, r2, r3 = rng.choice(idxs, 3, replace=False)

                # Eq. 5: mutation
                V = pop[r1] + self.F * (pop[r2] - pop[r3])
                V = np.clip(V, lo, hi)

                # Eq. 6-7: crossover
                U = pop[i].copy()
                j_rand = rng.randint(d)
                for j in range(d):
                    if rng.rand() <= self.CR or j == j_rand:
                        U[j] = V[j]

                f_U = self._fitness_auc(U, X_min_norm, X_maj_norm, y_min, y_maj)
                if f_U > fit_vals[i]:
                    pop[i] = U
                    fit_vals[i] = f_U

        return pop[np.argmax(fit_vals)]

    def _fitness_auc(self, w, X_min_norm, X_maj_norm, y_min, y_maj):
        """
        D4: nested subfold AUC evaluation (Section 5.7).
        """
        X_all = pd.concat([X_min_norm, X_maj_norm], ignore_index=True)
        y_all = np.concatenate([y_min, y_maj])

        n_splits = min(self.inner_folds,
                        min(np.bincount(y_all == y_min[0])))
        if n_splits < 2:
            return 0.0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        aucs = []

        for train_idx, test_idx in skf.split(X_all, y_all):
            y_train = y_all[train_idx]
            X_train = X_all.iloc[train_idx].reset_index(drop=True)

            sub_min_mask = y_train == y_min[0]
            X_sub_min = X_train[sub_min_mask].reset_index(drop=True)
            X_sub_maj = X_train[~sub_min_mask].reset_index(drop=True)

            if len(X_sub_min) < 2 or len(X_sub_maj) < 1:
                continue

            with threadpool_limits(limits=1):
                X_sub_bal_norm = self._generate_synthetic(X_sub_min, X_sub_maj, w)
                n_min_bal = len(X_sub_bal_norm) - len(X_sub_maj)
                y_sub_bal = np.concatenate([
                    np.full(n_min_bal, y_min[0]),
                    np.full(len(X_sub_maj), y_maj[0]),
                ])

                X_test = X_all.iloc[test_idx]
                y_test = y_all[test_idx]
                if len(np.unique(y_test)) < 2:
                    continue

                clf = self.clf
                clf.fit(X_sub_bal_norm, y_sub_bal)
                probs = clf.predict_proba(X_test)[:, 1]
            try:
                aucs.append(roc_auc_score(y_test, probs))
            except ValueError:
                continue

        return float(np.mean(aucs)) if aucs else 0.0

    # ------------------------------------------------------------------
    # Algorithm 1: COSTE algorithm data generation
    # ------------------------------------------------------------------
    def _generate_synthetic(self, X_min_norm, X_maj_norm, weights):
        feature_cols = X_min_norm.columns.tolist()
        N_min = X_min_norm.copy().reset_index(drop=True)
        N_maj = X_maj_norm

        # Line 1: Array_syn <- array for storing new synthetic instances
        Array_syn = []

        # Line 2: apply the min-max normalization method to the dataset
        # (already done in fit_resample before arriving at this LOC)

        def _rank_by_complexity(N_min_current):
            # Line 3: for each instance Xi in N_min, calculate its complexity using Equation (9)
            complexity = N_min_current[feature_cols].values.astype(float) @ weights
            # Line 4: rank Xi in the ascending order based on complexity
            order = np.argsort(complexity)
            return N_min_current.iloc[order].reset_index(drop=True)

        # Wrapper loop to allow the 'repeat Line 5' pseudo-GOTO jump
        while True:

            # Line 5: calculate the number of new synthetic instances needed T
            T = len(N_maj) - len(N_min)

            # Safety break just in case balancing is fully achieved
            if T <= 0:
                break

            # Line 6: if T > number(N_min) - 1 then
            if T > len(N_min) - 1:

                # ------------------------------------------------------------------
                # D1: Line 12 of the pseudocode says "repeat Line 5" (which skips
                # re-ranking). However, Section 4.4 explicitly states to "repeat
                # the above phases from 4.3 to 4.4" (Calculating complexity and
                # rank). It is re-ranked here to match the textual explanation,
                # not the literal pseudocode.
                # ------------------------------------------------------------------
                ranked = _rank_by_complexity(N_min)

                # ------------------------------------------------------------------
                # D2: 'batch' used here to match Section 4.4's instruction to
                # merge only "newly generated instances" (not the cumulative
                # Array_syn, which would compound duplicates across passes).
                # ------------------------------------------------------------------
                batch = []

                # Line 7: for i = 1, 2, ..., number(N_min) - 1 do
                for i in range(len(N_min) - 1):
                    # Line 8: new synthetic instance Xnew = (Xi + X(i+1)) / 2
                    X_new = (ranked.iloc[i].values + ranked.iloc[i + 1].values) / 2.0
                    # Line 9: add Xnew into Array_syn
                    batch.append(X_new)
                    Array_syn.append(X_new)
                # Line 10: end for

                # Line 11: update N_min by merging N_min and Array_syn
                N_min = pd.concat([N_min, pd.DataFrame(batch, columns=feature_cols)], ignore_index=True)

                # Line 12: repeat Line 5
                continue  # This jumps back to the top of the while loop, hitting Line 5 again

            # Line 13: else
            else:
                # ------------------------------------------------------------------
                # D1 (applies here too): re-ranking applied to match the textual
                # explanation, ensuring the final batch uses accurately updated
                # complexities.
                # ------------------------------------------------------------------
                ranked = _rank_by_complexity(N_min)

                # ------------------------------------------------------------------
                # D2 (applies here too): 'batch' isolates only this pass's
                # newly-generated instances.
                # ------------------------------------------------------------------
                batch = []

                # Line 14: for i = 1, 2, ..., T - 1 do
                for i in range(T - 1):
                    # Line 15: new synthetic instance Xnew = (Xi + X(i+1)) / 2
                    X_new = (ranked.iloc[i].values + ranked.iloc[i + 1].values) / 2.0
                    # Line 16: add Xnew into Array_syn
                    batch.append(X_new)
                    Array_syn.append(X_new)
                # Line 17: end for

                # Line 18: update N_min by merging N_min and Array_syn
                N_min = pd.concat([N_min, pd.DataFrame(batch, columns=feature_cols)], ignore_index=True)
                # Line 19: end if
                break

        # Line 20: return balanced dataset N_bal by merging N_min and N_maj
        N_bal = pd.concat([N_min, N_maj], ignore_index=True)
        return N_bal