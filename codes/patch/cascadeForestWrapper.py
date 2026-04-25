# =============================================================================
# patch/cascadeForestWrapper.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Thin compatibility wrapper around deepforest's
#            CascadeForestClassifier to ensure the `classes_` attribute is
#            always populated after fitting, making the model compatible with
#            scikit-learn pipeline conventions used in the experiments.
# =============================================================================

import numpy as np
from deepforest import CascadeForestClassifier
class CascadeForestWrapper(CascadeForestClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        if not hasattr(self, 'classes_'):
            self.classes_ = np.unique(y)
        return self