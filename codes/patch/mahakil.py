"""
MAHAKIL: Diversity Based Oversampling Approach
Based on: https://github.com/ai-se/MAHAKIL_imbalance
"""
from __future__ import division

from collections import Counter
import warnings
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import mahalanobis
import scipy as sp


class MAHAKIL:
    """MAHAKIL oversampling technique.

    Generates synthetic samples based on Mahalanobis distance and
    genetic inheritance principles.

    Parameters
    ----------
    sampling_strategy : str or dict, default='auto'
        Sampling strategy to use:
        - 'auto': Balance all minority classes to match majority class
        - dict: Specify target number of samples for each class

    random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm.
    """

    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matrix containing the data to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : array-like of shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        # Convert to DataFrame for easier manipulation
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # Reset index
        X = X.reset_index(drop=True)
        y = np.array(y)

        # Determine sampling strategy
        target_stats = Counter(y)
        n_samples_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)

        if self.sampling_strategy == 'auto':
            # Balance all minority classes to majority class size
            sampling_strategy_ = {
                key: n_samples_majority - value
                for key, value in target_stats.items()
                if key != class_majority and value < n_samples_majority
            }
        elif isinstance(self.sampling_strategy, dict):
            # Use provided dictionary
            sampling_strategy_ = {}
            for class_label, n_target_samples in self.sampling_strategy.items():
                current_samples = target_stats[class_label]
                if n_target_samples < current_samples:
                    raise ValueError(
                        f"With oversampling, target samples ({n_target_samples}) "
                        f"must be >= current samples ({current_samples}) for class {class_label}"
                    )
                samples_to_generate = n_target_samples - current_samples
                if samples_to_generate > 0:
                    sampling_strategy_[class_label] = samples_to_generate
        else:
            raise ValueError(
                "sampling_strategy must be 'auto' or a dictionary"
            )

        # Initialize resampled data with original data
        X_resampled = X.copy()
        y_resampled = y.copy()

        # Generate samples for each minority class
        for class_label, n_samples in sampling_strategy_.items():
            if n_samples == 0:
                continue

            # Get minority class samples
            class_indices = np.where(y == class_label)[0]
            X_class = X.iloc[class_indices].copy()

            # Generate synthetic samples
            X_new = self._generate_samples(X_class, n_samples)
            y_new = np.full(len(X_new), class_label)

            # Append to resampled data
            X_resampled = pd.concat([X_resampled, X_new], ignore_index=True)
            y_resampled = np.concatenate([y_resampled, y_new])

        return X_resampled.values, y_resampled

    def _generate_samples(self, X_class, n_samples):
        """Generate synthetic samples for a minority class.

        Parameters
        ----------
        X_class : DataFrame
            Samples from the minority class

        n_samples : int
            Number of synthetic samples to generate

        Returns
        -------
        X_new : DataFrame
            Generated synthetic samples
        """
        # Store original column names
        feature_cols = X_class.columns.tolist()

        # Compute Mahalanobis distances
        try:
            # Calculate covariance matrix
            cov_matrix = X_class.cov().values

            # Add regularization for numerical stability
            epsilon = 1e-6
            cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

            # Try to invert
            cov_inv = sp.linalg.inv(cov_matrix)
        except:
            # If inversion fails, try removing correlated features
            X_class = self._remove_correlated_features(X_class)
            feature_cols = X_class.columns.tolist()

            cov_matrix = X_class.cov().values
            epsilon = 1e-6
            cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

            try:
                cov_inv = sp.linalg.inv(cov_matrix)
            except:
                # Use pseudo-inverse as last resort
                cov_inv = sp.linalg.pinv(cov_matrix)

        # Calculate Mahalanobis distances
        mean_vec = X_class.mean().values
        maha_distances = []

        for idx in range(len(X_class)):
            try:
                dist = mahalanobis(X_class.iloc[idx].values, mean_vec, cov_inv) ** 2
                if np.isnan(dist) or np.isinf(dist):
                    # Fallback to Euclidean distance
                    dist = np.sum((X_class.iloc[idx].values - mean_vec) ** 2)
            except:
                # Fallback to Euclidean distance
                dist = np.sum((X_class.iloc[idx].values - mean_vec) ** 2)
            maha_distances.append(dist)

        # Add distance information
        X_class = X_class.copy()
        X_class['maha_dist'] = maha_distances
        X_class['maha_rank'] = X_class['maha_dist'].rank(ascending=False)

        # Sort by Mahalanobis distance (descending)
        X_class = X_class.sort_values('maha_dist', ascending=False).reset_index(drop=True)

        # Partition into two groups based on median rank
        median_rank = X_class['maha_rank'].median()

        # Assign group labels (simulating chromosomal inheritance)
        group_labels = []
        for idx, row in X_class.iterrows():
            if row['maha_rank'] <= median_rank:
                group_labels.append(1)
            else:
                group_labels.append(2)

        X_class['group'] = group_labels
        X_class['sample_id'] = range(len(X_class))

        # Generate synthetic samples through inheritance
        X_new = self._create_offspring(X_class, feature_cols, n_samples)

        return X_new

    def _create_offspring(self, X_class, feature_cols, n_samples):
        """Create offspring samples through genetic-like recombination.

        Parameters
        ----------
        X_class : DataFrame
            Parent samples with group labels

        feature_cols : list
            Original feature column names

        n_samples : int
            Number of offspring to generate

        Returns
        -------
        offspring : DataFrame
            Generated offspring samples
        """
        offspring_list = []
        samples_generated = 0
        generation = 0

        # Keep track of all samples (parents + offspring)
        all_samples = X_class.copy()

        while samples_generated < n_samples:
            generation += 1

            # Group samples by their group label
            groups = all_samples.groupby('group')

            # Generate offspring by averaging within each group
            new_offspring = []

            for group_label, group_data in groups:
                if len(group_data) > 0:
                    # Create offspring as mean of group members (genetic recombination)
                    offspring = group_data[feature_cols].mean()
                    offspring_df = pd.DataFrame([offspring], columns=feature_cols)
                    offspring_df['group'] = group_label
                    offspring_df['generation'] = generation

                    new_offspring.append(offspring_df)
                    offspring_list.append(offspring_df[feature_cols])
                    samples_generated += 1

                    if samples_generated >= n_samples:
                        break

            if len(new_offspring) == 0:
                break

            # Add new offspring to the pool for next generation
            new_offspring_df = pd.concat(new_offspring, ignore_index=True)
            all_samples = pd.concat([all_samples, new_offspring_df], ignore_index=True)

            # If we've generated enough, stop
            if samples_generated >= n_samples:
                break

        # Combine all offspring
        if offspring_list:
            result = pd.concat(offspring_list, ignore_index=True)
            # Return exactly n_samples
            return result.iloc[:n_samples]
        else:
            # Fallback: return copies of random samples
            indices = np.random.choice(len(X_class), size=n_samples, replace=True)
            return X_class.iloc[indices][feature_cols].reset_index(drop=True)

    def _remove_correlated_features(self, X):
        """Remove highly correlated features.

        Parameters
        ----------
        X : DataFrame
            Feature matrix

        Returns
        -------
        X_filtered : DataFrame
            Feature matrix with correlated features removed
        """
        # Convert to numeric if needed
        X = X.apply(pd.to_numeric, errors='coerce')

        # Remove columns with NaN or constant values
        X = X.loc[:, X.var() > 1e-10]

        if X.shape[1] == 0:
            raise ValueError("All features have zero variance")

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation > 0.95
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

        # Drop correlated features
        X_filtered = X.drop(columns=to_drop)

        return X_filtered


# Alias for compatibility
class Mahakil(MAHAKIL):
    """Alias for MAHAKIL for backward compatibility."""

    def fit_sample(self, X, y):
        """Legacy method name for compatibility.

        Use fit_resample instead.
        """
        warnings.warn(
            "fit_sample is deprecated. Use fit_resample instead.",
            DeprecationWarning
        )
        return self.fit_resample(X, y)