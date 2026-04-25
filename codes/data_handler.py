# =============================================================================
# data_handler.py
# -----------------------------------------------------------------------------
# Project  : Deconstructing Oversampling in Software Defect Prediction:
#            Algorithm Constraints, Trade-offs, and New Baselines
# Purpose  : Handles loading and preprocessing of SDP datasets.  Supports
#            both CSV and ARFF formats, drops constant/non-numeric columns,
#            and binarises the defect label ('bug') for classification.
# =============================================================================

import pandas as pd
from scipy.io import arff
import os

class DataHandler:

    @staticmethod
    def dataset_reader(dataset_name):

        datasets_path = '../datasets/' + dataset_name
        if not os.path.exists(datasets_path):
            raise FileNotFoundError(f"Dataset not found at: {datasets_path}")

        if dataset_name.endswith('.arff'):
            df = DataHandler.read_arff(datasets_path)
        else:
            df = pd.read_csv(datasets_path)

        columns_to_drop = [col for col in df.columns if df[col].nunique() == 1 or df[col].dtype == 'object']
        df = df.drop(columns=columns_to_drop)

        df.loc[df['bug'] > 0, 'bug'] = 1

        return df

    @staticmethod
    def read_arff(datasets_path):
        data, meta = arff.loadarff(datasets_path)
        df = pd.DataFrame(data)
        lbl = 'label' if 'label' in df.columns else 'Defective'
        df['bug'] = df[lbl].apply(lambda s: 1 if 'Y' in str(s) else 0)
        df = df.drop(columns=[lbl])

        return df