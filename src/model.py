"""Phase 5: Data Preparation & Stratified Train/Test Split module for Stay-Tuned.

This module prepares the clean feature matrix (X) and target vector (y) from the engineered
features and target definitions, and executes an 80/20 stratified train/test split.

Important Design Rule:
----------------------
X_train and y_train are designated for model training, cross-validation, and hyperparameter tuning.
X_test and y_test are strictly held out untouched until final model evaluation.
No scaling, encoding, imputation, or preprocessing is fitted across the full dataset.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_oulad_tables
from src.problem_definition import scope_to_module, compute_target
from src.feature_engineering import engineer_features


def prepare_data(features_df, targets_df):
    """Align features and target, separate X and y, and identify feature types.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Engineered feature matrix indexed by 'id_student' (or containing 'id_student').
    targets_df : pandas.DataFrame
        Disengagement target DataFrame containing 'id_student' and 'label'.

    Returns
    -------
    tuple
        (X, y, numerical_feature_names, categorical_feature_names)
        - X : pandas.DataFrame, shape (N, D), feature matrix indexed by id_student
        - y : pandas.Series, shape (N,), binary target vector (0 = retained, 1 = disengaged)
        - numerical_feature_names : list of str
        - categorical_feature_names : list of str
    """
    # 1. Reset index if id_student is in index to ensure clean merging
    feat_df = features_df.copy()
    if "id_student" not in feat_df.columns:
        feat_df = feat_df.reset_index()

    targ_df = targets_df[["id_student", "label"]].copy()

    # 2. Merge features and target on id_student
    merged_df = feat_df.merge(targ_df, on="id_student", how="inner")

    # Check for duplicate students
    if merged_df["id_student"].duplicated().any():
        raise ValueError("Duplicate student IDs found after merging features and targets!")

    # Set id_student as index
    merged_df = merged_df.set_index("id_student")

    # 3. Separate feature matrix X and target series y
    feature_cols = [col for col in merged_df.columns if col != "label"]
    X = merged_df[feature_cols].copy()
    y = merged_df["label"].copy().astype(int)

    # Ensure id_student is not a feature column
    assert "id_student" not in X.columns, "id_student must not be included in feature matrix X!"

    # 4. Dynamically identify numerical and categorical features
    numerical_feature_names = list(X.select_dtypes(include=[np.number]).columns)
    categorical_feature_names = list(X.select_dtypes(include=["object", "category"]).columns)

    # 5. Compute class balance and missing value statistics
    total_students = len(X)
    n_features = X.shape[1]
    n_num = len(numerical_feature_names)
    n_cat = len(categorical_feature_names)
    total_nulls = int(X.isnull().sum().sum())

    class_counts = y.value_counts().sort_index()
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    pct_0 = (count_0 / total_students * 100) if total_students > 0 else 0.0
    pct_1 = (count_1 / total_students * 100) if total_students > 0 else 0.0

    print("=" * 80)
    print("DATA PREPARATION SUMMARY (prepare_data)")
    print("=" * 80)
    print(f"Total Eligible Students:             {total_students:,}")
    print(f"Total Model Features:                {n_features}")
    print(f"  - Numerical Features ({n_num}):        {', '.join(numerical_feature_names)}")
    print(f"  - Categorical Features ({n_cat}):      {', '.join(categorical_feature_names) if n_cat > 0 else 'None'}")
    print(f"Missing Values in Feature Matrix X:  {total_nulls}")
    print("-" * 80)
    print("Target Class Balance (y):")
    print(f"  - Class 0 (Non-disengaged / Active): {count_0:>4} ({pct_0:.2f}%)")
    print(f"  - Class 1 (Disengaged / At-Risk):    {count_1:>4} ({pct_1:.2f}%)")
    print(f"  - Imbalance Ratio:                   {count_0 / max(count_1, 1):.2f} : 1")
    print("=" * 80 + "\n")

    return X, y, numerical_feature_names, categorical_feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Execute an 80/20 stratified train/test split on feature matrix and target.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    test_size : float, default=0.2
        Fraction of data to allocate to the holdout test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Split strategy: We use a stratified random split rather than a time-based split
    # because our features are already aggregated into a flat feature matrix
    # (one row per student, with no raw time series remaining). Each student's data
    # is independent. The temporal reasoning is embedded in the FEATURES
    # (Days 0–14 aggregation, trends, deltas, inactivity gaps), not in row ordering.
    # Stratification ensures the disengagement class balance is preserved in both
    # the training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    n_train = len(X_train)
    n_test = len(X_test)
    total = len(X)

    # Train class counts & percentages
    train_counts = y_train.value_counts().sort_index()
    train_0, train_1 = train_counts.get(0, 0), train_counts.get(1, 0)
    train_pct_0 = (train_0 / n_train * 100)
    train_pct_1 = (train_1 / n_train * 100)

    # Test class counts & percentages
    test_counts = y_test.value_counts().sort_index()
    test_0, test_1 = test_counts.get(0, 0), test_counts.get(1, 0)
    test_pct_0 = (test_0 / n_test * 100)
    test_pct_1 = (test_1 / n_test * 100)

    print("=" * 80)
    print("STRATIFIED TRAIN / TEST SPLIT SUMMARY (split_data)")
    print("=" * 80)
    print(f"Full Dataset Size:       {total:,} students (100.0%)")
    print(f"Training Set Size:       {n_train:,} students ({n_train / total * 100:.1f}%)")
    print(f"Holdout Test Set Size:   {n_test:,} students ({n_test / total * 100:.1f}%)")
    print("-" * 80)
    print("Class Balance Preservation:")
    print(f"{'Dataset Split':<18} | {'Total':<8} | {'Class 0 (Active)':<22} | {'Class 1 (Disengaged)'}")
    print("-" * 80)
    full_0, full_1 = (y == 0).sum(), (y == 1).sum()
    print(f"{'Full Cohort':<18} | {total:<8} | {full_0:>4} ({full_0/total*100:5.2f}%)          | {full_1:>4} ({full_1/total*100:5.2f}%)")
    print(f"{'Training Set':<18} | {n_train:<8} | {train_0:>4} ({train_pct_0:5.2f}%)          | {train_1:>4} ({train_pct_1:5.2f}%)")
    print(f"{'Holdout Test Set':<18} | {n_test:<8} | {test_0:>4} ({test_pct_0:5.2f}%)          | {test_1:>4} ({test_pct_1:5.2f}%)")
    print("=" * 80 + "\n")

    return X_train, X_test, y_train, y_test


def print_phase5_validation_report(X, y, X_train, X_test, y_train, y_test, num_features, cat_features):
    """Print the official Phase 5 validation report verifying all constraints."""
    print("=" * 80)
    print("PHASE 5: DATA PREPARATION & STRATIFIED SPLIT VALIDATION REPORT")
    print("=" * 80)
    print(f"1. Total Students in X:                  {len(X)} (Expected: 372)")
    print(f"2. Total Target Labels in y:             {len(y)} (Expected: 372)")
    print(f"3. Total Model Features:                 {X.shape[1]} (Expected: 8)")
    print(f"4. 'id_student' in X columns:            {'id_student' in X.columns} (Expected: False)")
    print(f"5. Training Set Size:                    {len(X_train)} (80% -> 297)")
    print(f"6. Holdout Test Set Size:                {len(X_test)} (20% -> 75)")
    print(f"7. Missing Values in X_train:            {X_train.isnull().sum().sum()}")
    print(f"8. Missing Values in X_test:             {X_test.isnull().sum().sum()}")
    print(f"9. Disengaged Rate in Training Set:      {(y_train == 1).mean() * 100:.2f}% (Full: {(y == 1).mean() * 100:.2f}%)")
    print(f"10. Disengaged Rate in Test Set:         {(y_test == 1).mean() * 100:.2f}% (Full: {(y == 1).mean() * 100:.2f}%)")
    print("-" * 80)

    # Assertions
    assert len(X) == 372, f"Expected 372 rows, got {len(X)}"
    assert len(y) == 372, f"Expected 372 rows, got {len(y)}"
    assert X.shape[1] == 8, f"Expected 8 features, got {X.shape[1]}"
    assert "id_student" not in X.columns, "'id_student' must not be in X columns"
    assert len(X_train) == 297, f"Expected 297 train rows, got {len(X_train)}"
    assert len(X_test) == 75, f"Expected 75 test rows, got {len(X_test)}"
    assert X_train.isnull().sum().sum() == 0, "Missing values in X_train"
    assert X_test.isnull().sum().sum() == 0, "Missing values in X_test"

    print("ALL PHASE 5 CHECKS PASSED: Data prepared and stratified split executed successfully.")
    print("Holdout test set is safely isolated for final evaluation.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 1. Resolve raw data directory
    data_dir = PROJECT_ROOT / "data" / "raw"

    # 2. Load raw tables
    raw_tables = load_oulad_tables(data_dir)

    # 3. Scope to AAA-2013J
    scoped = scope_to_module(raw_tables, code_module="AAA", code_presentation="2013J")

    # 4. Compute target for eligible students
    targets_df = compute_target(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
        student_vle_df=scoped["student_vle"],
        assessments_df=raw_tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
        code_module="AAA",
        code_presentation="2013J",
    )

    # Filter scoped student_info to eligible students
    eligible_student_info = scoped["student_info"][
        scoped["student_info"]["id_student"].isin(targets_df["id_student"])
    ].copy()

    # 5. Extract the 8 temporal model features (Days 0-14)
    features_df = engineer_features(
        student_vle_df=scoped["student_vle"],
        student_info_df=eligible_student_info,
        obs_end_day=14,
    )

    # 6. Prepare X, y, and feature type lists
    X, y, num_features, cat_features = prepare_data(features_df, targets_df)

    # 7. Execute 80/20 Stratified Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 8. Print official Phase 5 validation report
    print_phase5_validation_report(
        X, y, X_train, X_test, y_train, y_test, num_features, cat_features
    )
