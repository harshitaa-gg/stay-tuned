"""Phase 5, 6, 7 & 8: Machine Learning Modeling Pipeline for Stay-Tuned.

This module encompasses:
1. Data preparation (aligning features and targets into X, y).
2. Stratified 80/20 train/test split (isolating the holdout test set).
3. Leakage-safe Scikit-learn preprocessing ColumnTransformer and model Pipeline builder.
4. Stratified 5-Fold Cross-Validation model comparison across Baseline, Logistic Regression,
   Random Forest, and XGBoost on training data only.
5. Hyperparameter tuning of Random Forest and XGBoost using RandomizedSearchCV (Phase 8).
6. Multi-metric cross-validation comparison of tuned candidate models.

Anti-Leakage Architecture:
--------------------------
- X_test and y_test are NEVER accessed during model selection, cross-validation, or tuning.
- All preprocessing (imputation, scaling) is encapsulated inside Pipelines so it is fitted
  strictly on each training fold during cross-validation.
- Best model selection and operational threshold analysis are deferred until after tuned CV review.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
import joblib

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


def build_preprocessing_pipeline(numerical_features, categorical_features):
    """Construct an unfitted Scikit-learn ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    numerical_features : list of str
        List of numerical feature column names.
    categorical_features : list of str
        List of categorical feature column names.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted ColumnTransformer defining median imputation + standard scaling for numerical
        features, and most-frequent imputation + one-hot encoding for categorical features.
    """
    transformers = []

    # 1. Numerical preprocessing pipeline: Median Imputation -> StandardScaler
    if len(numerical_features) > 0:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numerical_pipeline, numerical_features))

    # 2. Categorical preprocessing pipeline: Most Frequent Imputation -> OneHotEncoder
    if len(categorical_features) > 0:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    # 3. Combine transformations
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


def build_model_pipeline(preprocessor, classifier):
    """Combine an unfitted preprocessor and an unfitted classifier into a complete ML Pipeline.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Unfitted ColumnTransformer preprocessing definition.
    classifier : estimator object
        Unfitted Scikit-learn or XGBoost classifier estimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted complete model pipeline.
    """
    # We use a Pipeline instead of manually preprocessing the data to prevent
    # data leakage. During training and cross-validation, preprocessing steps
    # such as imputation and scaling are fitted only on the training portion
    # of the data. The learned statistics are then applied to validation or
    # test data without allowing those datasets to influence the preprocessing.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def compare_models(X_train, y_train, preprocessor):
    """Evaluate and compare multiple ML models using Stratified 5-Fold Cross-Validation on training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix (N_train = 297).
    y_train : pandas.Series
        Training target vector.
    preprocessor : sklearn.compose.ColumnTransformer
        Unfitted preprocessing pipeline definition.

    Returns
    -------
    tuple
        (fitted_models, results_df)
        - fitted_models : dict of {str: Pipeline}, each fitted on full X_train, y_train
        - results_df : pandas.DataFrame, sorted by pr_auc_mean desc, f1_mean desc
    """
    # 1. Define four models
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    logistic_regression = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )

    xgboost = XGBClassifier(
        n_estimators=100,
        eval_metric="logloss",
        random_state=42,
    )

    # 2. Define model dictionary
    models = {
        "baseline": baseline,
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }

    # 3. Create stratified 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # 4. Evaluate models using ONLY training data
    scoring = [
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "average_precision",
    ]

    results_list = []

    for name, model in models.items():
        # Build complete pipeline
        pipeline = build_model_pipeline(clone(preprocessor), model)

        # Run cross-validation strictly on training data
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results_list.append({
            "model": name,
            "pr_auc_mean": float(np.mean(cv_results["test_average_precision"])),
            "pr_auc_std": float(np.std(cv_results["test_average_precision"])),
            "f1_mean": float(np.mean(cv_results["test_f1"])),
            "f1_std": float(np.std(cv_results["test_f1"])),
            "recall_mean": float(np.mean(cv_results["test_recall"])),
            "recall_std": float(np.std(cv_results["test_recall"])),
            "precision_mean": float(np.mean(cv_results["test_precision"])),
            "precision_std": float(np.std(cv_results["test_precision"])),
            "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        })

    # 5. Create comparison DataFrame and sort by PR-AUC and F1 descending
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(
        by=["pr_auc_mean", "f1_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Select the best model based on cross-validation results above,
    # NOT holdout test set performance.

    # 6. Fit each pipeline on full training set
    fitted_models = {}
    for name, model in models.items():
        full_pipeline = build_model_pipeline(clone(preprocessor), model)
        full_pipeline.fit(X_train, y_train)
        fitted_models[name] = full_pipeline

    return fitted_models, results_df


def tune_model(X_train, y_train, preprocessor, model_name):
    """Tune hyperparameters for a candidate model using RandomizedSearchCV on X_train only.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix (N=297).
    y_train : pandas.Series
        Training target vector.
    preprocessor : sklearn.compose.ColumnTransformer
        Unfitted preprocessing pipeline definition.
    model_name : str
        Candidate model identifier ('random_forest' or 'xgboost').

    Returns
    -------
    tuple
        (best_pipeline, best_params, best_cv_score)
        - best_pipeline : Pipeline, refitted on the full X_train, y_train
        - best_params : dict, optimal hyperparameters found
        - best_cv_score : float, mean PR-AUC across the 5 CV folds for best_params
    """
    if model_name == "random_forest":
        base_model = RandomForestClassifier(random_state=42)
        param_distributions = {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__max_features": ["sqrt", "log2", 0.5],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
        }
    elif model_name == "xgboost":
        # Calculate class imbalance ratio dynamically from y_train only
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
        param_distributions = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        }
    else:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. Must be 'random_forest' or 'xgboost'."
        )

    # Build leakage-safe pipeline:
    # The pipeline ensures preprocessing is fitted separately inside each cross-validation training fold.
    pipeline = build_model_pipeline(clone(preprocessor), base_model)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    randomized_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    # Fit hyperparameter search strictly using X_train and y_train only
    randomized_search.fit(X_train, y_train)

    best_pipeline = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    best_cv_score = float(randomized_search.best_score_)

    print("=" * 60)
    print("PHASE 8: HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Model:                        {model_name}")
    print(f"Best Parameters:              {best_params}")
    print(f"Best Cross-Validation PR-AUC: {best_cv_score:.4f}")
    print("=" * 60 + "\n")

    return best_pipeline, best_params, best_cv_score


def tune_candidate_models(X_train, y_train, preprocessor):
    """Tune both Random Forest and XGBoost candidate models on X_train.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training target vector.
    preprocessor : sklearn.compose.ColumnTransformer
        Unfitted ColumnTransformer.

    Returns
    -------
    dict
        Dictionary containing tuned pipelines, parameters, and CV PR-AUC scores.
    """
    # 1. Tune Random Forest
    rf_pipe, rf_params, rf_score = tune_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        model_name="random_forest",
    )

    # 2. Tune XGBoost
    xgb_pipe, xgb_params, xgb_score = tune_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        model_name="xgboost",
    )

    tuned_models = {
        "random_forest": {
            "pipeline": rf_pipe,
            "best_params": rf_params,
            "best_cv_pr_auc": rf_score,
        },
        "xgboost": {
            "pipeline": xgb_pipe,
            "best_params": xgb_params,
            "best_cv_pr_auc": xgb_score,
        },
    }

    return tuned_models


def compare_tuned_models(X_train, y_train, tuned_models):
    """Compare the best tuned models across multiple metrics using 5-Fold Stratified CV on X_train.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training target vector.
    tuned_models : dict
        Dictionary of tuned models returned by tune_candidate_models.

    Returns
    -------
    pandas.DataFrame
        Comparison DataFrame with rows 'random_forest_tuned' and 'xgboost_tuned'.
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = [
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "average_precision",
    ]

    results_list = []

    for name in ["random_forest", "xgboost"]:
        model_info = tuned_models[name]
        pipeline = model_info["pipeline"]

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        results_list.append({
            "model": f"{name}_tuned",
            "pr_auc_mean": float(np.mean(cv_results["test_average_precision"])),
            "pr_auc_std": float(np.std(cv_results["test_average_precision"])),
            "f1_mean": float(np.mean(cv_results["test_f1"])),
            "f1_std": float(np.std(cv_results["test_f1"])),
            "recall_mean": float(np.mean(cv_results["test_recall"])),
            "recall_std": float(np.std(cv_results["test_recall"])),
            "precision_mean": float(np.mean(cv_results["test_precision"])),
            "precision_std": float(np.std(cv_results["test_precision"])),
            "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        })

    # Final model selection must be based on cross-validation results from
    # X_train and y_train only. The holdout test set must not influence model
    # selection, hyperparameter tuning, or threshold selection.

    comparison_df = pd.DataFrame(results_list)
    return comparison_df


def print_phase8_validation_report(tuned_models, comparison_df):
    """Print the official Phase 8 validation report verifying all tuning constraints."""
    print("=" * 80)
    print("PHASE 8: HYPERPARAMETER TUNING VALIDATION REPORT")
    print("=" * 80)
    print("1. Tuned Models:")
    print("   Random Forest")
    print("   XGBoost")
    print("\n2. Training Data Used:")
    print("   X_train, y_train ONLY")
    print("\n3. Holdout Test Set Used:")
    print("   False")
    print("\n4. Cross-Validation:")
    print("   5-Fold StratifiedKFold")
    print("\n5. Hyperparameter Search:")
    print("   RandomizedSearchCV")
    print("\n6. Search Iterations:")
    print("   30 per model")
    print("\n7. Primary Optimization Metric:")
    print("   PR-AUC (Average Precision)")
    print("\n8. Random Forest:")
    print(f"   Best Parameters: {tuned_models['random_forest']['best_params']}")
    print(f"   Best CV PR-AUC:  {tuned_models['random_forest']['best_cv_pr_auc']:.4f}")
    print("\n9. XGBoost:")
    print(f"   Best Parameters: {tuned_models['xgboost']['best_params']}")
    print(f"   Best CV PR-AUC:  {tuned_models['xgboost']['best_cv_pr_auc']:.4f}")
    print("\n10. Tuned Model Comparison:")
    print("-" * 80)

    formatted_df = comparison_df.copy()
    formatted_df["PR-AUC"] = formatted_df.apply(
        lambda r: f"{r['pr_auc_mean']:.4f} +/- {r['pr_auc_std']:.4f}", axis=1
    )
    formatted_df["F1"] = formatted_df.apply(
        lambda r: f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}", axis=1
    )
    formatted_df["Recall"] = formatted_df.apply(
        lambda r: f"{r['recall_mean']:.4f} +/- {r['recall_std']:.4f}", axis=1
    )
    formatted_df["Precision"] = formatted_df.apply(
        lambda r: f"{r['precision_mean']:.4f} +/- {r['precision_std']:.4f}", axis=1
    )
    formatted_df["ROC-AUC"] = formatted_df.apply(
        lambda r: f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}", axis=1
    )

    display_cols = ["model", "PR-AUC", "F1", "Recall", "Precision", "ROC-AUC"]
    print(formatted_df[display_cols].to_string(index=False))
    print("-" * 80)

    # Strict Anti-Leakage and Completion Assertions
    assert "random_forest" in tuned_models and "xgboost" in tuned_models, "Both RF and XGBoost must be tuned"
    assert isinstance(tuned_models["random_forest"]["pipeline"], Pipeline), "RF best estimator must be Pipeline"
    assert isinstance(tuned_models["xgboost"]["pipeline"], Pipeline), "XGBoost best estimator must be Pipeline"
    assert len(comparison_df) == 2, f"Expected 2 rows in tuned comparison, got {len(comparison_df)}"

    # Confirm models/best_model.joblib is NOT saved prematurely
    saved_model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    assert not saved_model_path.exists(), "best_model.joblib must NOT be saved in Phase 8!"

    print("============================================================")
    print("PHASE 8 CHECKS PASSED: Hyperparameter tuning completed.")
    print("Holdout test set remains completely sealed.")
    print("============================================================\n")


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


def print_phase6_validation_report(preprocessor, sample_pipeline, num_features, cat_features):
    """Print the official Phase 6 validation report verifying the unfitted pipeline."""
    print("=" * 80)
    print("PHASE 6: PREPROCESSING PIPELINE VALIDATION REPORT")
    print("=" * 80)
    print(f"1. Number of Numerical Features:         {len(num_features)} (Expected: 8)")
    print(f"2. Number of Categorical Features:       {len(cat_features)} (Expected: 0)")
    print("3. Numerical Preprocessing Steps:        Median Imputation -> StandardScaler")
    print("4. Categorical Preprocessing Steps:      Most Frequent Imputation -> OneHotEncoder")
    print(f"5. Preprocessor Type:                    {type(preprocessor).__name__}")
    print(f"6. Preprocessor Status:                  Defined but NOT fitted (hasattr 'transformers_': {hasattr(preprocessor, 'transformers_')})")
    print(f"7. Sample Model Pipeline Type:           {type(sample_pipeline).__name__}")
    print(f"8. Sample Pipeline Steps:                {list(sample_pipeline.named_steps.keys())}")
    print(f"9. Model Pipeline Status:                Defined but NOT fitted (Holdout test set untouched)")
    print("-" * 80)

    # Assertions
    assert isinstance(preprocessor, ColumnTransformer), "preprocessor must be a ColumnTransformer"
    assert isinstance(sample_pipeline, Pipeline), "sample_pipeline must be a Pipeline"
    assert hasattr(preprocessor, "transformers_") is False, "Preprocessor must NOT be fitted in Phase 6!"
    assert len(num_features) == 8, f"Expected 8 numerical features, got {len(num_features)}"

    print("ALL PHASE 6 CHECKS PASSED: Preprocessing and model pipeline definitions are leakage-safe.")
    print("=" * 80 + "\n")


def print_phase7_validation_report(results_df, fitted_models, n_train):
    """Print the official Phase 7 model comparison and cross-validation report."""
    print("=" * 80)
    print("PHASE 7: MODEL TRAINING & CROSS-VALIDATION REPORT")
    print("=" * 80)
    print(f"Training Cohort Size:            {n_train} students")
    print("Cross-Validation Strategy:       5-Fold StratifiedKFold (shuffle=True, random_state=42)")
    print("Evaluated Models:                1. baseline (DummyClassifier)")
    print("                                 2. logistic_regression")
    print("                                 3. random_forest")
    print("                                 4. xgboost")
    print("Evaluation Metrics:              PR-AUC (avg_precision), F1, Recall, Precision, ROC-AUC")
    print("Data Source for Selection:       X_train, y_train ONLY (Holdout test set untouched)")
    print("-" * 80)
    print("Model Cross-Validation Comparison Table (sorted by PR-AUC desc):")
    print("-" * 80)

    # Format table nicely
    formatted_df = results_df.copy()
    formatted_df["PR-AUC"] = formatted_df.apply(lambda r: f"{r['pr_auc_mean']:.4f} +/- {r['pr_auc_std']:.4f}", axis=1)
    formatted_df["F1"] = formatted_df.apply(lambda r: f"{r['f1_mean']:.4f} +/- {r['f1_std']:.4f}", axis=1)
    formatted_df["Recall"] = formatted_df.apply(lambda r: f"{r['recall_mean']:.4f} +/- {r['recall_std']:.4f}", axis=1)
    formatted_df["Precision"] = formatted_df.apply(lambda r: f"{r['precision_mean']:.4f} +/- {r['precision_std']:.4f}", axis=1)
    formatted_df["ROC-AUC"] = formatted_df.apply(lambda r: f"{r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f}", axis=1)

    display_cols = ["model", "PR-AUC", "F1", "Recall", "Precision", "ROC-AUC"]
    print(formatted_df[display_cols].to_string(index=False))
    print("-" * 80)

    # Assertions
    assert len(results_df) == 4, f"Expected 4 models in comparison table, got {len(results_df)}"
    assert len(fitted_models) == 4, f"Expected 4 fitted pipelines, got {len(fitted_models)}"
    for name, pipe in fitted_models.items():
        assert hasattr(pipe.named_steps["classifier"], "classes_"), f"Model {name} was not fitted on X_train"
        assert hasattr(pipe.named_steps["preprocessor"], "transformers_"), f"Preprocessor for {name} was not fitted on X_train"

    print("ALL PHASE 7 CHECKS PASSED: 4 models compared via 5-Fold Stratified CV on X_train.")
    print("All pipelines fitted on complete X_train. Holdout test set remains completely untouched.")
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

    # 6. Prepare X, y, and feature type lists (Phase 5)
    X, y, num_features, cat_features = prepare_data(features_df, targets_df)

    # 7. Execute 80/20 Stratified Split (Phase 5)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 8. Print official Phase 5 validation report
    print_phase5_validation_report(
        X, y, X_train, X_test, y_train, y_test, num_features, cat_features
    )

    # 9. Build Preprocessing and Sample Model Pipeline (Phase 6)
    preprocessor = build_preprocessing_pipeline(
        numerical_features=num_features,
        categorical_features=cat_features,
    )

    sample_classifier = LogisticRegression(random_state=42)
    sample_pipeline = build_model_pipeline(
        preprocessor=preprocessor,
        classifier=sample_classifier,
    )

    # 10. Print official Phase 6 validation report
    print_phase6_validation_report(
        preprocessor=preprocessor,
        sample_pipeline=sample_pipeline,
        num_features=num_features,
        cat_features=cat_features,
    )

    # 11. Run Phase 7 Model Comparison & Cross-Validation
    fitted_models, cv_results_df = compare_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    # 12. Print official Phase 7 validation report
    print_phase7_validation_report(
        results_df=cv_results_df,
        fitted_models=fitted_models,
        n_train=len(X_train),
    )

    # 13. Run Phase 8 Hyperparameter Tuning for Random Forest and XGBoost
    tuned_models = tune_candidate_models(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    # 14. Compare Tuned Models across multiple metrics using 5-Fold Stratified CV
    tuned_comparison_df = compare_tuned_models(
        X_train=X_train,
        y_train=y_train,
        tuned_models=tuned_models,
    )

    # 15. Print official Phase 8 validation report
    print_phase8_validation_report(
        tuned_models=tuned_models,
        comparison_df=tuned_comparison_df,
    )

