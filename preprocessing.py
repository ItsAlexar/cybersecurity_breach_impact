import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 10

# =============================================================================
# Missing rows/cols filter 
# =============================================================================

def compute_missing_stats(df):
    missing_cols = pd.DataFrame({'missing_count': df.isna().sum()})
    missing_cols['missing_%'] = (missing_cols['missing_count'] / len(df) * 100).round(2)
    missing_cols = missing_cols[missing_cols['missing_count'] > 0].sort_values('missing_%', ascending=False)

    missing_rows = pd.DataFrame({'missing_count': df.isna().sum(axis=1)})
    missing_rows['missing_%'] = (missing_rows['missing_count'] / df.shape[1] * 100).round(2)
    missing_rows = missing_rows[missing_rows['missing_count'] > 0].sort_values('missing_%', ascending=False)

    return missing_cols, missing_rows

def get_cols_rows_to_drop(missing_cols, missing_rows, threshold_cols, threshold_rows):
    cols_to_drop = missing_cols[missing_cols['missing_%'] > threshold_cols]
    rows_to_drop = missing_rows[missing_rows['missing_%'] > threshold_rows]
    return cols_to_drop, rows_to_drop

def drop_missing(df, numerical_cols, threshold_cols=60, threshold_rows=30):
    missing_cols, missing_rows = compute_missing_stats(df)
    cols_to_drop, rows_to_drop = get_cols_rows_to_drop(missing_cols, missing_rows, threshold_cols, threshold_rows)
    df = df.drop(columns=cols_to_drop.index).drop(index=rows_to_drop.index).copy()
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    return df, numerical_cols

# =============================================================================
# Imputation 
# =============================================================================

def impute(df, numerical_cols):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

# =============================================================================
# Skewness Correction 
# =============================================================================

def correct_skewness(df, numerical_cols):
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    df[numerical_cols] = pt.fit_transform(df[numerical_cols])
    return df

# =============================================================================
# Outlier Handling 
# =============================================================================

def get_outlier_bounds(df, cols):
    bounds = df[cols].quantile([0.25, 0.75])
    bounds.index = ["Q1", "Q3"]
    bounds.loc["IQR"]         = bounds.loc["Q3"] -       bounds.loc["Q1"]
    bounds.loc["inner_lower"] = bounds.loc["Q1"] - 1.5 * bounds.loc["IQR"]
    bounds.loc["inner_upper"] = bounds.loc["Q3"] + 1.5 * bounds.loc["IQR"]
    bounds.loc["outer_lower"] = bounds.loc["Q1"] - 3.0 * bounds.loc["IQR"]
    bounds.loc["outer_upper"] = bounds.loc["Q3"] + 3.0 * bounds.loc["IQR"]
    return bounds

def get_outlier_masks(df, cols):
    bounds = get_outlier_bounds(df, cols)
    # Potential outliers (inner fence, includes extreme outliers)
    outlier_mask_inner = (df[cols] < bounds.loc["inner_lower"]) | (df[cols] > bounds.loc["inner_upper"])
    # Extreme outliers (outer fence, subset of the previous)
    outlier_mask_outer = (df[cols] < bounds.loc["outer_lower"]) | (df[cols] > bounds.loc["outer_upper"])
    return outlier_mask_inner, outlier_mask_outer

def get_outlier_stats(df, cols):
    outlier_mask_inner, outlier_mask_outer = get_outlier_masks(df, cols)
    outlier_stats = pd.DataFrame({
    "inner_count":   (outlier_mask_inner & ~outlier_mask_outer).sum(),
    "inner_%":     (outlier_mask_inner & ~outlier_mask_outer).mean() * 100,
    "outer_count": outlier_mask_outer.sum(),
    "outer_%":   outlier_mask_outer.mean() * 100,
    "total_count":   outlier_mask_inner.sum(),
    "total_%":     outlier_mask_inner.mean() * 100,}).round(2)
    return outlier_stats

def get_outlier_stats(df, cols):
    outlier_mask_inner, outlier_mask_outer = get_outlier_masks(df, cols)
    outlier_stats = pd.DataFrame({
    "inner_count":   (outlier_mask_inner & ~outlier_mask_outer).sum(),
    "inner_%":     (outlier_mask_inner & ~outlier_mask_outer).mean() * 100,
    "outer_count": outlier_mask_outer.sum(),
    "outer_%":   outlier_mask_outer.mean() * 100,
    "total_count":   outlier_mask_inner.sum(),
    "total_%":     outlier_mask_inner.mean() * 100,}).round(2)
    return outlier_stats

def cap_outliers(df, cols):
    bounds = get_outlier_bounds(df, cols)
    df_capped = df.copy()
    for col in cols:
        df_capped[col] = df_capped[col].clip(lower=bounds.loc["outer_lower", col], upper=bounds.loc["outer_upper", col])
    return df_capped

# =============================================================================
# Scaling 
# =============================================================================

def scale(df, numerical_cols):
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
