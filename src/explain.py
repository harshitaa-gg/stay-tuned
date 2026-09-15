"""Phase 11: Model Interpretability and SHAP Explainability Module for Stay-Tuned.

This module computes SHAP (SHapley Additive exPlanations) values to explain feature contributions
to disengagement risk predictions at both the global (cohort) and local (student) levels.

Strict Governance & Anti-Leakage Rules:
---------------------------------------
1. Frozen Model: Uses the frozen Random Forest pipeline from models/best_model.joblib.
   No model retraining, hyperparameter tuning, or threshold adjustments are performed.
2. Training Set Only: SHAP values and background distributions are derived strictly from X_train.
   X_test is NOT accessed or used for computing explanations.
3. Descriptive Nature: SHAP values explain statistical associations and model decision logic;
   they describe model contribution and DO NOT establish causal relationships.
4. Actionable Mapping: Feature explanations are mapped back to canonical human-interpretable
   feature names rather than raw preprocessor artifact tags.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# Canonical original features used in the project
CANONICAL_FEATURES = [
    "total_early_clicks",
    "clicks_per_active_day",
    "active_days",
    "longest_inactivity_gap",
    "days_since_last_active",
    "activity_delta",
    "weekly_click_ratio",
    "studied_credits",
]


def map_feature_to_original(post_feature_name, original_features=None):
    """Map a post-preprocessing feature name back to the canonical project feature name.

    Handles column transformer prefixes (e.g. 'num__', 'cat__', 'remainder__')
    and one-hot encoded suffixes (e.g. 'categorical__feature_value').

    Parameters
    ----------
    post_feature_name : str
        The transformed feature name from preprocessor.get_feature_names_out().
    original_features : list of str, optional
        List of canonical feature names. Defaults to CANONICAL_FEATURES.

    Returns
    -------
    str
        The matched original feature name.
    """
    if original_features is None:
        original_features = CANONICAL_FEATURES

    clean_name = post_feature_name
    for prefix in ["num__", "cat__", "remainder__"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]

    # Direct match
    if clean_name in original_features:
        return clean_name

    # Prefix match for categorical one-hot encoded columns (e.g., 'region_London')
    for orig in original_features:
        if clean_name.startswith(orig + "_") or clean_name.startswith(orig + "__") or clean_name == orig:
            return orig

    return clean_name


def compute_shap_values(best_pipeline, X_train, feature_names=None):
    """Compute SHAP values for all training samples using the appropriate SHAP explainer.

    Parameters
    ----------
    best_pipeline : sklearn.pipeline.Pipeline
        Fitted Scikit-Learn pipeline (loaded from best_model.joblib).
    X_train : pandas.DataFrame
        Training feature matrix (N=297).
    feature_names : list of str, optional
        Original feature names. If None, derived from X_train.columns.

    Returns
    -------
    tuple
        (shap_values, X_train_transformed, feature_names_after_preprocessing)
        - shap_values : numpy.ndarray of shape (n_samples, n_features), positive class contributions
        - X_train_transformed : numpy.ndarray of shape (n_samples, n_features)
        - feature_names_after_preprocessing : list of str
    """
    # 1. Extract fitted preprocessor and classifier from Pipeline
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]

    # 2. Transform X_train using the fitted preprocessor
    X_train_transformed = preprocessor.transform(X_train)

    # Convert sparse matrices to dense array if necessary
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    # 3. Obtain post-preprocessing feature names
    feature_names_after_preprocessing = list(preprocessor.get_feature_names_out())

    # 4. Instantiate the appropriate SHAP Explainer
    clf_class_name = classifier.__class__.__name__
    if "Forest" in clf_class_name or "XGB" in clf_class_name or "Tree" in clf_class_name:
        explainer = shap.TreeExplainer(classifier)
    elif "Logistic" in clf_class_name or "Linear" in clf_class_name or "Ridge" in clf_class_name:
        explainer = shap.LinearExplainer(classifier, X_train_transformed)
    else:
        explainer = shap.Explainer(classifier, X_train_transformed)

    # 5. Compute SHAP values for ALL X_train samples
    raw_shap = explainer.shap_values(X_train_transformed)

    # 6. Normalize SHAP output into 2D array (n_samples, n_features) for positive class (disengaged)
    if isinstance(raw_shap, list):
        # List of arrays [class_0, class_1]: index 1 corresponds to disengaged
        shap_values = np.array(raw_shap[1])
    elif isinstance(raw_shap, np.ndarray):
        if raw_shap.ndim == 3:
            # Shape (n_samples, n_features, n_classes): index 1 is positive class
            shap_values = raw_shap[:, :, 1]
        elif raw_shap.ndim == 2:
            shap_values = raw_shap
        else:
            raise ValueError(f"Unexpected SHAP array dimensions: {raw_shap.shape}")
    else:
        # Explanation object in modern SHAP
        if hasattr(raw_shap, "values"):
            values = raw_shap.values
            if values.ndim == 3:
                shap_values = values[:, :, 1]
            else:
                shap_values = values
        else:
            raise TypeError(f"Unrecognized SHAP output type: {type(raw_shap)}")

    # 7. Robust Validation Checks
    n_samples, n_features = shap_values.shape
    assert n_samples == len(X_train), f"Sample count mismatch: {n_samples} vs {len(X_train)}"
    assert n_features == len(feature_names_after_preprocessing), (
        f"Feature count mismatch: {n_features} vs {len(feature_names_after_preprocessing)}"
    )
    assert not np.isnan(shap_values).any(), "Found unexpected NaN in SHAP values."
    assert not np.isinf(shap_values).any(), "Found unexpected inf in SHAP values."

    print("Computed SHAP values successfully on training data:")
    print(f"  - Explainer Type:          {type(explainer).__name__}")
    print(f"  - Classifier:              {clf_class_name}")
    print(f"  - Samples Explained:       {n_samples}")
    print(f"  - Features:                {n_features}")

    return shap_values, X_train_transformed, feature_names_after_preprocessing


def plot_global_importance(
    shap_values,
    features,
    feature_names,
    save_path=None,
):
    """Generate and save global SHAP summary plot, and return top ranked features.

    Parameters
    ----------
    shap_values : numpy.ndarray of shape (n_samples, n_features)
        SHAP values for positive class.
    features : numpy.ndarray or pandas.DataFrame
        Transformed feature matrix.
    feature_names : list of str
        Post-preprocessing feature names.
    save_path : Path or str, optional
        Destination image path. Defaults to models/shap_global.png.

    Returns
    -------
    list of str
        Top 10 canonical feature names ranked by mean absolute SHAP value.
    """
    if save_path is None:
        save_path = PROJECT_ROOT / "models" / "shap_global.png"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Map display names to readable original feature names
    display_names = [map_feature_to_original(fn) for fn in feature_names]

    # 1. Create SHAP global beeswarm summary plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=display_names,
        show=False,
    )
    plt.title(
        "SHAP Global Feature Importance (Summary Plot) — Training Set",
        fontsize=12,
        weight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Global SHAP summary plot saved to: {save_path}")

    # 2. Calculate mean absolute SHAP value per feature
    mean_abs_per_col = np.mean(np.abs(shap_values), axis=0)

    # 3. Aggregate importance by canonical original feature
    orig_importance = {}
    for fn, val in zip(feature_names, mean_abs_per_col):
        orig_name = map_feature_to_original(fn)
        orig_importance[orig_name] = orig_importance.get(orig_name, 0.0) + float(val)

    # 4. Rank features by mean absolute SHAP impact descending
    ranked = sorted(orig_importance.items(), key=lambda x: x[1], reverse=True)
    top_10_features = [item[0] for item in ranked[:10]]

    return top_10_features, ranked


def explain_student(
    student_id,
    predictions_df,
    features_df,
    shap_values,
    feature_names,
    top_n=5,
):
    """Explain an individual student's risk prediction using local SHAP attribution.

    Parameters
    ----------
    student_id : int or str
        The student ID to explain.
    predictions_df : pandas.DataFrame
        Predictions DataFrame containing ['id_student', 'disengagement_probability', 'risk_level'].
    features_df : pandas.DataFrame
        Training feature matrix used during SHAP computation, preserving exact row alignment.
    shap_values : numpy.ndarray of shape (n_samples, n_features)
        SHAP values matrix aligned row-for-row with features_df.
    feature_names : list of str
        Post-preprocessing feature names aligned with columns of shap_values.
    top_n : int, default=5
        Number of top contributing factors to report.

    Returns
    -------
    dict
        Structured student explanation dictionary.
    """
    # 1. Locate student in features_df to preserve exact row alignment
    if "id_student" in features_df.columns:
        matching_indices = features_df.index[features_df["id_student"] == student_id].tolist()
        if not matching_indices:
            raise ValueError(f"Student ID {student_id} not found in features_df.")
        row_pos = features_df.index.get_loc(matching_indices[0])
    else:
        # id_student is the index
        if student_id not in features_df.index:
            raise ValueError(f"Student ID {student_id} not found in features_df index.")
        row_pos = features_df.index.get_loc(student_id)

    # 2. Extract the student's corresponding SHAP vector
    student_shap = shap_values[row_pos]

    # 3. Find prediction info in predictions_df
    pred_match = predictions_df[predictions_df["id_student"] == student_id]
    if pred_match.empty:
        raise ValueError(f"Student ID {student_id} not found in predictions_df.")
    prob = float(pred_match["disengagement_probability"].iloc[0])
    risk = str(pred_match["risk_level"].iloc[0])

    # Assert prediction domain validity
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"
    assert risk in {"LOW", "MEDIUM", "HIGH"}, f"Unexpected risk level: {risk}"

    # 4. Aggregate SHAP contributions by original feature name
    aggregated_shap = {}
    for fn, s_val in zip(feature_names, student_shap):
        orig_name = map_feature_to_original(fn)
        aggregated_shap[orig_name] = aggregated_shap.get(orig_name, 0.0) + float(s_val)

    # 5. Classify impact and direction
    factors = []
    for feat_name, s_val in aggregated_shap.items():
        if s_val > 1e-6:
            direction = "positive"
        elif s_val < -1e-6:
            direction = "negative"
        else:
            direction = "neutral"

        factors.append({
            "feature": feat_name,
            "direction": direction,
            "impact": float(np.round(abs(s_val), 4)),
            "shap_value": float(np.round(s_val, 4)),
        })

    # 6. Rank by absolute impact descending and select top_n
    factors_sorted = sorted(factors, key=lambda x: x["impact"], reverse=True)
    top_factors = factors_sorted[:top_n]

    explanation = {
        "id_student": int(student_id),
        "disengagement_probability": prob,
        "risk_level": risk,
        "top_risk_factors": top_factors,
    }

    return explanation


def save_shap_artifacts(shap_values, feature_names, output_dir=None):
    """Save computed SHAP values and feature names for downstream reproducibility and dashboards.

    Parameters
    ----------
    shap_values : numpy.ndarray
        Computed training SHAP values.
    feature_names : list of str
        Post-preprocessing feature names.
    output_dir : Path or str, optional
        Target directory. Defaults to models/.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    shap_path = output_dir / "shap_values.npy"
    names_path = output_dir / "shap_feature_names.npy"

    np.save(shap_path, shap_values)
    np.save(names_path, np.array(feature_names))

    print(f"SHAP values saved to:         {shap_path}")
    print(f"SHAP feature names saved to:  {names_path}")


def print_student_explanation_card(explanation):
    """Print an interpretable, human-readable student risk explanation card."""
    s_id = explanation["id_student"]
    prob = explanation["disengagement_probability"]
    risk = explanation["risk_level"]
    factors = explanation["top_risk_factors"]

    print("=" * 75)
    print(f"STUDENT RISK EXPLANATION CARD — ID: {s_id}")
    print("=" * 75)
    print(f"Disengagement Probability:       {prob:.4f}")
    print(f"Assigned Risk Tier:              {risk}")
    print("-" * 75)
    print("KEY MODEL CONTRIBUTIONS (TOP FACTORS):")
    print(f"{'Feature':<28} | {'Direction':<12} | {'Impact (abs)':<14} | {'SHAP Contribution'}")
    print("-" * 75)
    for f in factors:
        dir_symbol = "(+) Risk Increasing" if f["direction"] == "positive" else "(-) Protective / Decreasing" if f["direction"] == "negative" else "Neutral"
        print(f"{f['feature']:<28} | {dir_symbol:<25} | {f['impact']:<14.4f} | {f['shap_value']:+.4f}")
    print("-" * 75)
    print("Interpretation Guidance:")
    print("  * Positive (+) SHAP values pushed this student's estimated risk higher.")
    print("  * Negative (-) SHAP values acted as protective indicators, pushing risk lower.")
    print("  * Explanations describe model decision logic only; they do not establish causation.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    from src.data_loader import load_oulad_tables
    from src.problem_definition import scope_to_module, compute_target
    from src.feature_engineering import engineer_features
    from src.model import prepare_data, split_data
    from src.predict import load_model, generate_predictions

    # 1. Load the frozen model artifact (Random Forest, threshold 0.35)
    pipeline, threshold, artifact = load_model()

    # 2. Recreate training data using identical Day 0-14 pipeline
    data_dir = PROJECT_ROOT / "data" / "raw"
    raw_tables = load_oulad_tables(data_dir)
    scoped = scope_to_module(raw_tables, code_module="AAA", code_presentation="2013J")
    targets_df = compute_target(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
        student_vle_df=scoped["student_vle"],
        assessments_df=raw_tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
        code_module="AAA",
        code_presentation="2013J",
    )
    eligible_student_info = scoped["student_info"][
        scoped["student_info"]["id_student"].isin(targets_df["id_student"])
    ].copy()

    features_df = engineer_features(
        student_vle_df=scoped["student_vle"],
        student_info_df=eligible_student_info,
        obs_end_day=14,
    )

    X, y, _, _ = prepare_data(features_df, targets_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 3. Compute SHAP values strictly on X_train only (N=297)
    shap_vals, X_train_trans, feat_names_trans = compute_shap_values(
        best_pipeline=pipeline,
        X_train=X_train,
    )

    # 4. Save SHAP values and feature names
    save_shap_artifacts(shap_vals, feat_names_trans)

    # 5. Generate and save global importance summary plot
    top_10, ranked_all = plot_global_importance(
        shap_values=shap_vals,
        features=X_train_trans,
        feature_names=feat_names_trans,
    )

    print("\n" + "=" * 80)
    print("GLOBAL FEATURE IMPORTANCE RANKING (Mean Absolute SHAP Value):")
    print("=" * 80)
    for rank, (feat, imp) in enumerate(ranked_all, 1):
        print(f"  {rank:>2}. {feat:<28} : {imp:.4f}")
    print("-" * 80)
    print("SHAP values explain how each feature contributed to the model's prediction for a learner.")
    print("Positive values push predicted disengagement risk higher, while negative values push it lower.")
    print("SHAP explanations describe model behavior and do not establish causation.")
    print("=" * 80 + "\n")

    # 6. Generate predictions for training cohort to demonstrate student-level explanation
    train_predictions = generate_predictions(
        best_pipeline=pipeline,
        X=X_train,
        student_ids=list(X_train.index),
    )

    # 7. Select and explain representative students (one HIGH risk, one LOW risk)
    high_risk_candidates = train_predictions[train_predictions["risk_level"] == "HIGH"]
    low_risk_candidates = train_predictions[train_predictions["risk_level"] == "LOW"]

    if not high_risk_candidates.empty:
        example_high_id = high_risk_candidates.iloc[0]["id_student"]
        high_exp = explain_student(
            student_id=example_high_id,
            predictions_df=train_predictions,
            features_df=X_train,
            shap_values=shap_vals,
            feature_names=feat_names_trans,
            top_n=5,
        )
        print_student_explanation_card(high_exp)

    if not low_risk_candidates.empty:
        example_low_id = low_risk_candidates.iloc[0]["id_student"]
        low_exp = explain_student(
            student_id=example_low_id,
            predictions_df=train_predictions,
            features_df=X_train,
            shap_values=shap_vals,
            feature_names=feat_names_trans,
            top_n=5,
        )
        print_student_explanation_card(low_exp)

    # 8. Assertions verifying all saved files exist
    assert (PROJECT_ROOT / "models" / "shap_values.npy").exists(), "shap_values.npy was not created"
    assert (PROJECT_ROOT / "models" / "shap_feature_names.npy").exists(), "shap_feature_names.npy was not created"
    assert (PROJECT_ROOT / "models" / "shap_global.png").exists(), "shap_global.png was not created"
    print("ALL PHASE 11 EXPLAINABILITY CHECKS PASSED: SHAP artifacts created and verified successfully.")

