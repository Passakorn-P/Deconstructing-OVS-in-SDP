"""
ROS: Random OverSampling
"""
from collections import Counter
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

class ROS:

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.sampler = RandomOverSampler(random_state=self.random_state)

    def fit_resample(self, X, y):

        X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        if isinstance(X_resampled, pd.DataFrame):
            X_resampled = X_resampled.values
        else:
            X_resampled = np.asarray(X_resampled)

        y_resampled = np.asarray(y_resampled)

        return X_resampled, y_resampled

    def fit_sample(self, X, y):
        return self.fit_resample(X, y)