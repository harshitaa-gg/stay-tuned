"""Phase 10: Disengagement Probability & Risk Classification Module for Stay-Tuned.

This module provides the operational inference and scoring layer for the Early Learner
Disengagement Prediction project. It ingests Day 0–14 learner behavioral feature matrices,
generates continuous disengagement probabilities using the frozen model artifact
(models/best_model.joblib), and maps probabilities to decision-support risk tiers (LOW,
MEDIUM, HIGH) and operational early-warning intervention flags (predicted_label).

Operational Threshold & Risk Tier Architecture:
----------------------------------------------
- Operational Early-Warning Threshold: 0.35 (Frozen in Phase 8.2)
    - probability >= 0.35 -> predicted_label = 1 (Flagged for proactive intervention)
    - probability < 0.35  -> predicted_label = 0 (On-track / monitor)
- Decision-Support Risk Tiers:
    - LOW:    probability < 0.35
    - MEDIUM: 0.35 <= probability <= 0.65
    - HIGH:   probability > 0.65

Important Design Distinction (Risk Level vs. Predicted Label):
-------------------------------------------------------------
- predicted_label answers: "Does the student meet the operational threshold (0.35) for early outreach?"
- risk_level answers:      "What is the severity of the student's estimated disengagement risk?"
- Therefore:
    - LOW:    predicted_label = 0 (not flagged for immediate alert)
    - MEDIUM: predicted_label = 1 (flagged; standard supportive intervention)
    - HIGH:   predicted_label = 1 (flagged; intensive / urgent intervention)

No Model Training:
------------------
This module strictly performs inference. It does NOT train, fit, or calibrate models.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

# The 8 canonical features required row-for-row by the frozen pipeline
EXPECTED_FEATURES = [
    "total_early_clicks",
    "clicks_per_active_day",
    "active_days",
    "longest_inactivity_gap",
    "days_since_last_active",
    "activity_delta",
    "weekly_click_ratio",
    "studied_credits",
]

OPERATIONAL_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65


def load_model(model_path=None):
    """Load the frozen final model pipeline from disk and validate threshold.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to the saved model artifact. Defaults to models/best_model.joblib.

    Returns
    -------
    tuple
        (pipeline, threshold, artifact_dict)
    """
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen model artifact not found at: {model_path}. "
            "Ensure Phase 8.2 was completed successfully before running predictions."
        )

    artifact = joblib.load(model_path)

    if "pipeline" not in artifact:
        raise KeyError(f"Corrupt artifact at {model_path}: missing 'pipeline' key.")
    if "threshold" not in artifact:
        raise KeyError(f"Corrupt artifact at {model_path}: missing 'threshold' key.")

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    if not np.isclose(threshold, OPERATIONAL_THRESHOLD, atol=1e-4):
        raise ValueError(
            f"Model threshold mismatch! Expected {OPERATIONAL_THRESHOLD:.2f}, got {threshold:.2f}. "
            "Phase 10 requires the frozen 0.35 operating threshold selected in Phase 8.2."
        )

    return pipeline, threshold, artifact


def generate_predictions(best_pipeline, X, student_ids):
    """Generate estimated disengagement probabilities, risk levels, and operational labels.

    Parameters
    ----------
    best_pipeline : sklearn.pipeline.Pipeline
        Fitted final Scikit-Learn pipeline (loaded from best_model.joblib).
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix containing the 8 model features.
    student_ids : array-like or list
        Student identifier labels corresponding row-for-row with X.

    Returns
    -------
    pandas.DataFrame
        DataFrame with EXACTLY four columns:
        ['id_student', 'disengagement_probability', 'risk_level', 'predicted_label']
    """
    # 1. Input Shape & Alignment Validation
    student_ids_list = list(student_ids)
    n_samples = len(X)
    n_ids = len(student_ids_list)

    if n_samples != n_ids:
        raise ValueError(
            f"Dimension mismatch: X has {n_samples} rows, but student_ids has {n_ids} entries."
        )

    if isinstance(X, pd.DataFrame):
        missing_cols = [c for c in EXPECTED_FEATURES if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Feature matrix X is missing required feature columns: {missing_cols}. "
                f"Expected features: {EXPECTED_FEATURES}"
            )
        # Ensure exact column order expected by preprocessor
        X_input = X[EXPECTED_FEATURES].copy()
    else:
        if X.shape[1] != len(EXPECTED_FEATURES):
            raise ValueError(
                f"Feature array X has {X.shape[1]} columns; expected {len(EXPECTED_FEATURES)}."
            )
        X_input = X

    # 2. Predict continuous probabilities for Class 1 (Disengaged)
    # predict_proba() gives estimated posterior probabilities P(y = 1 | X)
    y_prob = best_pipeline.predict_proba(X_input)[:, 1]

    # Validate output length and probability domain [0, 1]
    if len(y_prob) != n_ids:
        raise ValueError(
            f"Prediction count mismatch: expected {n_ids}, got {len(y_prob)} probabilities."
        )

    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError("Model produced probabilities outside the valid range [0.0, 1.0].")

    # 3. Assign Risk Levels (decision support categories) using UNROUNDED probabilities
    # LOW:    prob < 0.35
    # MEDIUM: 0.35 <= prob <= 0.65
    # HIGH:   prob > 0.65
    risk_levels = []
    for p in y_prob:
        if p < OPERATIONAL_THRESHOLD:
            risk_levels.append("LOW")
        elif p <= HIGH_RISK_THRESHOLD:
            risk_levels.append("MEDIUM")
        else:
            risk_levels.append("HIGH")

    # 4. Assign Operational Early-Warning Predicted Label using UNROUNDED probabilities
    # probability >= 0.35 -> predicted_label = 1
    # probability <  0.35 -> predicted_label = 0
    # Note: 0.50 is intentionally NOT used because 0.35 was calibrated for sensitivity in Phase 8.2
    predicted_labels = (y_prob >= OPERATIONAL_THRESHOLD).astype(int)

    # 5. Domain validations
    valid_risk_levels = {"LOW", "MEDIUM", "HIGH"}
    if not set(risk_levels).issubset(valid_risk_levels):
        raise ValueError(f"Invalid risk_level detected. Must be in {valid_risk_levels}.")

    valid_labels = {0, 1}
    if not set(predicted_labels).issubset(valid_labels):
        raise ValueError(f"Invalid predicted_label detected. Must be in {valid_labels}.")

    # 6. Construct clean return DataFrame with exact required columns
    # Round probability to 4 decimals for presentation AFTER classification
    predictions_df = pd.DataFrame({
        "id_student": student_ids_list,
        "disengagement_probability": np.round(y_prob, 4),
        "risk_level": risk_levels,
        "predicted_label": predicted_labels,
    })

    return predictions_df


def get_risk_summary(predictions_df):
    """Compute and display summary statistics and validation checks for risk predictions.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        Predictions DataFrame generated by generate_predictions().

    Returns
    -------
    dict
        Summary metrics dictionary.
    """
    total_students = len(predictions_df)
    if total_students == 0:
        raise ValueError("predictions_df is empty.")

    # 1. Risk Tier Distributions
    counts = predictions_df["risk_level"].value_counts()
    count_low = int(counts.get("LOW", 0))
    count_med = int(counts.get("MEDIUM", 0))
    count_high = int(counts.get("HIGH", 0))

    pct_low = float(count_low / total_students * 100)
    pct_med = float(count_med / total_students * 100)
    pct_high = float(count_high / total_students * 100)

    # 2. Probability Means
    mean_prob_all = float(predictions_df["disengagement_probability"].mean())
    mean_prob_low = float(predictions_df.loc[predictions_df["risk_level"] == "LOW", "disengagement_probability"].mean()) if count_low > 0 else 0.0
    mean_prob_med = float(predictions_df.loc[predictions_df["risk_level"] == "MEDIUM", "disengagement_probability"].mean()) if count_med > 0 else 0.0
    mean_prob_high = float(predictions_df.loc[predictions_df["risk_level"] == "HIGH", "disengagement_probability"].mean()) if count_high > 0 else 0.0

    # 3. Operational Flagging Summary (predicted_label = 1)
    flagged_count = int((predictions_df["predicted_label"] == 1).sum())
    flagged_pct = float(flagged_count / total_students * 100)

    # 4. Strict Validation Checks
    assert count_low + count_med + count_high == total_students, "Risk counts do not sum to total students."
    assert np.isclose(pct_low + pct_med + pct_high, 100.0, atol=1e-2), "Risk percentages do not sum to ~100%."
    assert ((predictions_df["disengagement_probability"] >= 0.0) & (predictions_df["disengagement_probability"] <= 1.0)).all(), "Probabilities outside [0, 1]."

    if count_low > 0:
        assert (predictions_df.loc[predictions_df["risk_level"] == "LOW", "disengagement_probability"] < OPERATIONAL_THRESHOLD).all(), "LOW risk probability >= 0.35."
        assert (predictions_df.loc[predictions_df["risk_level"] == "LOW", "predicted_label"] == 0).all(), "LOW risk student has predicted_label == 1."

    if count_med > 0:
        assert ((predictions_df.loc[predictions_df["risk_level"] == "MEDIUM", "disengagement_probability"] >= OPERATIONAL_THRESHOLD) &
                (predictions_df.loc[predictions_df["risk_level"] == "MEDIUM", "disengagement_probability"] <= HIGH_RISK_THRESHOLD)).all(), "MEDIUM risk outside [0.35, 0.65]."
        assert (predictions_df.loc[predictions_df["risk_level"] == "MEDIUM", "predicted_label"] == 1).all(), "MEDIUM risk student has predicted_label == 0."

    if count_high > 0:
        assert (predictions_df.loc[predictions_df["risk_level"] == "HIGH", "disengagement_probability"] > HIGH_RISK_THRESHOLD).all(), "HIGH risk probability <= 0.65."
        assert (predictions_df.loc[predictions_df["risk_level"] == "HIGH", "predicted_label"] == 1).all(), "HIGH risk student has predicted_label == 0."

    # 5. Print Risk Summary Report
    print("=" * 80)
    print("PHASE 10: DISENGAGEMENT RISK CLASSIFICATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Evaluated Students:        {total_students:,}")
    print(f"Operational Decision Threshold:  {OPERATIONAL_THRESHOLD:.2f} (Frozen in Phase 8.2)")
    print(f"High-Risk Tier Boundary:         {HIGH_RISK_THRESHOLD:.2f}")
    print("-" * 80)
    print("RISK TIER DISTRIBUTION:")
    print(f"  - LOW    (< 0.35):             {count_low:>4} students ({pct_low:5.2f}%) | Mean Probability: {mean_prob_low:.4f}")
    print(f"  - MEDIUM [0.35, 0.65]:         {count_med:>4} students ({pct_med:5.2f}%) | Mean Probability: {mean_prob_med:.4f}")
    print(f"  - HIGH   (> 0.65):             {count_high:>4} students ({pct_high:5.2f}%) | Mean Probability: {mean_prob_high:.4f}")
    print(f"  - OVERALL COHORT AVERAGE:      {total_students:>4} students (100.00%) | Mean Probability: {mean_prob_all:.4f}")
    print("-" * 80)
    print("OPERATIONAL EARLY-WARNING FLAGGING:")
    print(f"  - Operationally flagged at-risk learners (threshold = 0.35): {flagged_count:>4} / {total_students} ({flagged_pct:.2f}%)")
    print("    * LOW risk (unflagged, on-track):                           {count_low:>4} ({pct_low:.2f}%)".format(count_low=count_low, pct_low=pct_low))
    print(f"    * MEDIUM + HIGH risk (flagged for outreach):               {flagged_count:>4} ({flagged_pct:.2f}%)")
    print("-" * 80)
    print("GOVERNANCE & DECISION SUPPORT NOTES:")
    print("  1. The disengagement probability is an estimated risk propensity, not a predetermined outcome.")
    print("  2. LOW / MEDIUM / HIGH categories are human-in-the-loop decision-support tiers for academic staff.")
    print("  3. Threshold 0.35 is the frozen early-warning cutoff selected in Phase 8.2 to optimize learner recall.")
    print("  4. Threshold 0.65 serves as a high-urgency boundary to be refined based on institutional capacity.")
    print("=" * 80 + "\n")

    summary_dict = {
        "total_students": total_students,
        "count_low": count_low,
        "pct_low": pct_low,
        "mean_prob_low": mean_prob_low,
        "count_med": count_med,
        "pct_med": pct_med,
        "mean_prob_med": mean_prob_med,
        "count_high": count_high,
        "pct_high": pct_high,
        "mean_prob_high": mean_prob_high,
        "mean_prob_all": mean_prob_all,
        "flagged_count": flagged_count,
        "flagged_pct": flagged_pct,
    }
    return summary_dict


if __name__ == "__main__":
    from src.data_loader import load_oulad_tables
    from src.problem_definition import scope_to_module, compute_target
    from src.feature_engineering import engineer_features
    from src.model import prepare_data, split_data

    # 1. Load the frozen model artifact without retraining
    pipeline, threshold, artifact = load_model()

    # 2. Recreate feature data without retraining
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
    _, X_test, _, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 3. Demonstrate inference on the holdout test set (75 students)
    student_ids_test = list(X_test.index)
    test_predictions = generate_predictions(
        best_pipeline=pipeline,
        X=X_test,
        student_ids=student_ids_test,
    )

    # 4. Print risk summary
    risk_summary = get_risk_summary(test_predictions)

    # 5. Display sample predictions table
    print("SAMPLE PREDICTIONS TABLE (First 10 Learners):")
    print(test_predictions.head(10).to_string(index=False))
    print("\nSAMPLE PREDICTIONS TABLE (High/Medium Risk Cases):")
    high_med_cases = test_predictions[test_predictions["risk_level"].isin(["HIGH", "MEDIUM"])].head(10)
    print(high_med_cases.to_string(index=False))
    print("-" * 80)
