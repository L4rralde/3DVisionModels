import numpy as np
import pandas as pd

from .types import FeaturesDf


def filter_features_df(feats_df: FeaturesDf, lower: float=0.2, upper: float=0.8) -> FeaturesDf:
    sims_ut = np.triu(feats_df.sims_mat, 1)
    rows_duplicated, cols_duplicated = np.where(sims_ut > upper)
    idcs_to_drop = np.unique(cols_duplicated)

    high_mask = np.array([True] * len(feats_df))
    high_mask[idcs_to_drop] = False

    most_similar_value = sims_ut.max(0)
    to_drop = np.where(most_similar_value < lower)[0]
    low_mask = np.array([True] * len(feats_df))
    low_mask[to_drop] = False

    mask = high_mask & low_mask
    df_filtered = pd.DataFrame(feats_df.df[mask])

    return FeaturesDf(df_filtered, feats_df.dir_path)
