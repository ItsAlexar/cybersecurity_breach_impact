import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


# distinguish between numerical and categorical features
def define_feature_types(df):
    numerical_features = df.select_dtypes(include=["number"]).columns
    categorical_features = df.columns.difference(numerical_features, sort=False)
    return numerical_features, categorical_features

"""potential_targets = [category for category in cat_cols if category not in ["incident_id", "industry_primary", "industry_secondary"]] # Exclude incident_id as it's an identifier, not a target variable
mappings = {} # Save in case of need of later interpretation

for target_name in potential_targets:
    target = df_merged[target_name]
    target_pct = target.value_counts(normalize=True) * 100
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    df_merged[f"{target_name}_encoded"] = target_encoded # TODO not correct, should return df_encoded only

    # Keep the mapping for interpretability later
    mapping = {label: index for index, label in enumerate(le.classes_)}
    mappings['target_name'] = mapping
    print(f"Mapping for {target_name}:")
    print(mapping)
    print()"""