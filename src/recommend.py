"""Phase 12: Rule-Based Intervention Recommendations Module for Stay-Tuned.

This module provides deterministic, rule-based heuristic guidance to support
advisors, tutors, and academic support staff in deciding how to follow up with
learners based on Phase 10 risk tier classifications and Day 0–14 behavioral indicators.

Strict Governance, Non-ML, and Anti-Leakage Architecture:
---------------------------------------------------------
1. Rule-Based Heuristics (NOT Machine Learning / AI):
   - This engine consists strictly of transparent if/else decision logic.
   - It does NOT train, fit, tune, or modify any machine learning models.
   - It does NOT alter the frozen operational threshold (0.35) or risk tiers.
   - The recommendation logic is NOT an ML prediction or an algorithmic decision.

2. Decision Support Only:
   - Recommendations provide triage and communication guidance for human educators.
   - They do NOT automatically determine academic standing, grades, or administrative outcomes.

3. Experimental Disclaimer:
   - "These recommendations are rule-based heuristics for demonstration purposes.
      They are NOT validated interventions. Their effectiveness in reducing actual
      disengagement has not been tested experimentally."

4. Dataset Constraints & Assessment Signal Handling:
   - In Module AAA, Presentation 2013J, no assessments are scheduled or due during
     the Day 0–14 observation window (TMA 01 is due on Day 19).
   - Therefore, 'submitted_any_early_assessment' is NOT part of the current canonical
     feature set for AAA-2013J.
   - The engine treats 'submitted_any_early_assessment' as an OPTIONAL feature.
   - If 'submitted_any_early_assessment' is unavailable, values are NEVER fabricated
     or defaulted; the engine seamlessly applies activity-based fallback rules.
   - The engine architecture is extensible to courses/presentations where early
     assessment signals are available.

5. Anti-Leakage Compliance:
   - The holdout test set (X_test / y_test) is NEVER used for tuning or validating
     these rules.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

# Allowed recommendation types specified by system architecture
ALLOWED_RECOMMENDATION_TYPES = {
    "monitor",
    "nudge",
    "reminder",
    "outreach",
    "urgent_outreach",
}

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def get_intervention(
    student_id: Union[int, str],
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate a rule-based intervention recommendation for an individual learner.

    These recommendations are rule-based heuristics for demonstration purposes.

    They are NOT validated interventions. Their effectiveness in reducing actual
    disengagement has not been tested experimentally.

    The rules provide decision support for academic advisors and support teams,
    and do not automatically determine academic outcomes.

    Parameters
    ----------
    student_id : int or str
        Unique student identifier.
    predictions_df : pandas.DataFrame
        DataFrame containing learner disengagement predictions from Phase 10.
        Must contain 'risk_level' ('LOW', 'MEDIUM', or 'HIGH').
        May have 'id_student' as a column or as the DataFrame index.
    features_df : pandas.DataFrame
        DataFrame containing Day 0–14 behavioral features.
        Must contain 'activity_delta' and 'longest_inactivity_gap'.
        May optionally contain 'submitted_any_early_assessment'.
        May have 'id_student' as a column or as the DataFrame index.

    Returns
    -------
    dict
        Dictionary containing:
        - 'id_student': student identifier
        - 'risk_level': 'LOW', 'MEDIUM', or 'HIGH'
        - 'recommendation': action guidance text
        - 'recommendation_type': one of {'monitor', 'nudge', 'reminder', 'outreach', 'urgent_outreach'}

    Raises
    ------
    KeyError
        If student_id is not found in predictions_df or features_df.
    ValueError
        If risk_level is invalid, or required features are missing/non-numeric.
    """
    # -------------------------------------------------------------------------
    # 1. Locate Student Record in predictions_df
    # -------------------------------------------------------------------------
    if not isinstance(predictions_df, pd.DataFrame):
        raise TypeError("predictions_df must be a pandas DataFrame.")

    if "id_student" in predictions_df.columns:
        pred_matches = predictions_df[predictions_df["id_student"] == student_id]
        if pred_matches.empty:
            raise KeyError(
                f"Student ID {student_id} not found in predictions_df column 'id_student'."
            )
        pred_row = pred_matches.iloc[0]
    elif student_id in predictions_df.index:
        pred_row = predictions_df.loc[student_id]
        if isinstance(pred_row, pd.DataFrame):  # duplicate index safeguard
            pred_row = pred_row.iloc[0]
    else:
        raise KeyError(
            f"Student ID {student_id} not found in predictions_df (checked index and 'id_student' column)."
        )

    # -------------------------------------------------------------------------
    # 2. Locate Student Record in features_df
    # -------------------------------------------------------------------------
    if not isinstance(features_df, pd.DataFrame):
        raise TypeError("features_df must be a pandas DataFrame.")

    if "id_student" in features_df.columns:
        feat_matches = features_df[features_df["id_student"] == student_id]
        if feat_matches.empty:
            raise KeyError(
                f"Student ID {student_id} not found in features_df column 'id_student'."
            )
        feat_row = feat_matches.iloc[0]
    elif student_id in features_df.index:
        feat_row = features_df.loc[student_id]
        if isinstance(feat_row, pd.DataFrame):  # duplicate index safeguard
            feat_row = feat_row.iloc[0]
    else:
        raise KeyError(
            f"Student ID {student_id} not found in features_df (checked index and 'id_student' column)."
        )

    # -------------------------------------------------------------------------
    # 3. Read and Validate Inputs
    # -------------------------------------------------------------------------
    if "risk_level" not in pred_row:
        raise KeyError(f"predictions_df for student {student_id} missing 'risk_level'.")

    risk_level = str(pred_row["risk_level"]).strip().upper()
    if risk_level not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Invalid risk_level '{risk_level}' for student {student_id}. "
            f"Expected one of {sorted(VALID_RISK_LEVELS)}."
        )

    # Validate activity_delta
    if "activity_delta" not in feat_row:
        raise KeyError(f"features_df for student {student_id} missing 'activity_delta'.")
    raw_delta = feat_row["activity_delta"]
    if pd.isna(raw_delta) or not isinstance(raw_delta, (int, float, np.number)):
        raise ValueError(
            f"activity_delta for student {student_id} must be numeric, got: {raw_delta}"
        )
    activity_delta = float(raw_delta)

    # Validate longest_inactivity_gap
    if "longest_inactivity_gap" not in feat_row:
        raise KeyError(
            f"features_df for student {student_id} missing 'longest_inactivity_gap'."
        )
    raw_gap = feat_row["longest_inactivity_gap"]
    if pd.isna(raw_gap) or not isinstance(raw_gap, (int, float, np.number)):
        raise ValueError(
            f"longest_inactivity_gap for student {student_id} must be numeric, got: {raw_gap}"
        )
    longest_inactivity_gap = float(raw_gap)

    # -------------------------------------------------------------------------
    # 4. Derive Indicators
    # -------------------------------------------------------------------------
    activity_trend_negative = 1 if activity_delta < 0 else 0

    # -------------------------------------------------------------------------
    # 5. Check Optional Assessment Signal (Without Fabricating Missing Data)
    # -------------------------------------------------------------------------
    has_assessment_signal = False
    submitted_any_early_assessment: Optional[int] = None

    if "submitted_any_early_assessment" in feat_row:
        val = feat_row["submitted_any_early_assessment"]
        if val is not None and not pd.isna(val):
            try:
                submitted_any_early_assessment = int(val)
                has_assessment_signal = True
            except (ValueError, TypeError):
                # If non-convertible, treat as unavailable signal rather than crashing
                has_assessment_signal = False
                submitted_any_early_assessment = None

    # -------------------------------------------------------------------------
    # 6. Apply Rule-Based Decision Logic
    # -------------------------------------------------------------------------
    recommendation: str
    recommendation_type: str

    if risk_level == "LOW":
        # LOW-risk rules
        recommendation = "No immediate action needed. Monitor weekly engagement."
        recommendation_type = "monitor"

    elif risk_level == "MEDIUM":
        # MEDIUM-risk rules
        if activity_delta < 0:
            recommendation = (
                "Send a motivational engagement nudge — weekly progress reminder."
            )
            recommendation_type = "nudge"
        elif has_assessment_signal and submitted_any_early_assessment == 0:
            recommendation = (
                "Send an assessment reminder — highlight the upcoming assessment deadline."
            )
            recommendation_type = "reminder"
        else:
            recommendation = "Monitor closely. Check engagement next week."
            recommendation_type = "monitor"

    elif risk_level == "HIGH":
        # HIGH-risk rules
        if longest_inactivity_gap >= 7:
            recommendation = (
                "Priority outreach — student has been inactive for 7+ days. "
                "Direct learner support contact recommended."
            )
            recommendation_type = "outreach"
        elif (
            activity_delta < 0
            and has_assessment_signal
            and submitted_any_early_assessment == 0
        ):
            recommendation = (
                "Urgent: both engagement and assessment completion are low. "
                "Recommend immediate instructor or support outreach."
            )
            recommendation_type = "urgent_outreach"
        elif activity_delta < 0:
            recommendation = (
                "Declining engagement detected. Send personalized re-engagement "
                "email with course highlights."
            )
            recommendation_type = "outreach"
        elif has_assessment_signal and submitted_any_early_assessment == 0:
            recommendation = (
                "No early assessments submitted. Remind student of upcoming "
                "deadlines and course value."
            )
            recommendation_type = "reminder"
        else:
            recommendation = (
                "High risk flagged. Consider reaching out with course resources "
                "and motivational content."
            )
            recommendation_type = "outreach"

    else:
        # Unreachable due to earlier check, but defensive safeguard
        raise ValueError(f"Unhandled risk_level: {risk_level}")

    # -------------------------------------------------------------------------
    # 7. Validate Return Object Structure
    # -------------------------------------------------------------------------
    if recommendation_type not in ALLOWED_RECOMMENDATION_TYPES:
        raise ValueError(
            f"Generated recommendation_type '{recommendation_type}' is not allowed. "
            f"Must be one of {sorted(ALLOWED_RECOMMENDATION_TYPES)}."
        )

    result = {
        "id_student": student_id,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "recommendation_type": recommendation_type,
    }

    # Strict dictionary structure verification
    required_keys = {"id_student", "risk_level", "recommendation", "recommendation_type"}
    if set(result.keys()) != required_keys:
        raise ValueError(
            f"Result dictionary keys {set(result.keys())} do not match expected {required_keys}."
        )

    return result


def generate_recommendations(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate rule-based intervention recommendations for all learners in predictions_df.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        DataFrame containing learner predictions and risk levels.
    features_df : pandas.DataFrame
        DataFrame containing learner behavioral features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['id_student', 'risk_level', 'recommendation_type', 'recommendation']
        (plus 'disengagement_probability' if present in predictions_df).
    """
    if "id_student" in predictions_df.columns:
        student_ids = predictions_df["id_student"].tolist()
    else:
        student_ids = predictions_df.index.tolist()

    records: List[Dict[str, Any]] = []
    for sid in student_ids:
        rec = get_intervention(sid, predictions_df, features_df)

        # Include disengagement_probability if present for context
        if "id_student" in predictions_df.columns:
            row = predictions_df[predictions_df["id_student"] == sid].iloc[0]
        else:
            row = predictions_df.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

        if "disengagement_probability" in row:
            rec["disengagement_probability"] = row["disengagement_probability"]

        records.append(rec)

    recs_df = pd.DataFrame(records)

    # Reorder columns cleanly
    col_order = ["id_student", "risk_level"]
    if "disengagement_probability" in recs_df.columns:
        col_order.append("disengagement_probability")
    col_order.extend(["recommendation_type", "recommendation"])

    return recs_df[col_order].copy()


def print_recommendation_card(intervention: Dict[str, Any]) -> None:
    """Print an individual learner intervention recommendation card.

    Parameters
    ----------
    intervention : dict
        Intervention dictionary returned by get_intervention().
    """
    sid = intervention["id_student"]
    risk = intervention["risk_level"]
    rec_type = intervention["recommendation_type"]
    text = intervention["recommendation"]

    print("=" * 80)
    print(f"INTERVENTION RECOMMENDATION CARD: Learner ID {sid}")
    print("=" * 80)
    print(f"Risk Tier:            {risk}")
    print(f"Recommendation Type:  {rec_type.upper()}")
    print(f"Action Guidance:      {text}")
    print("-" * 80)
    print("GOVERNANCE & ETHICAL DISCLAIMER:")
    print("  * These recommendations are rule-based heuristics for demonstration.")
    print("  * They are NOT validated interventions; effectiveness has not been tested.")
    print("  * This is decision support only and does not determine academic outcomes.")
    print("=" * 80 + "\n")


def print_recommendation_summary_report(recs_df: pd.DataFrame) -> None:
    """Print a cohort-level summary of rule-based intervention recommendations.

    Parameters
    ----------
    recs_df : pandas.DataFrame
        DataFrame of recommendations from generate_recommendations().
    """
    total = len(recs_df)
    type_counts = recs_df["recommendation_type"].value_counts()
    risk_type_cross = pd.crosstab(
        recs_df["risk_level"],
        recs_df["recommendation_type"],
        margins=True,
    )

    print("\n" + "=" * 80)
    print("PHASE 12: INTERVENTION RECOMMENDATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Learners Evaluated: {total:,}")
    print("-" * 80)
    print("RECOMMENDATION TYPE DISTRIBUTION:")
    for rtype in ["monitor", "nudge", "reminder", "outreach", "urgent_outreach"]:
        cnt = int(type_counts.get(rtype, 0))
        pct = (cnt / total * 100) if total > 0 else 0.0
        print(f"  - {rtype:<16}: {cnt:>4} learners ({pct:5.2f}%)")
    print("-" * 80)
    print("CROSS-TABULATION (Risk Tier vs. Recommendation Type):")
    print(risk_type_cross.to_string())
    print("-" * 80)
    print("ETHICAL & OPERATIONAL PRINCIPLES:")
    print("  1. Heuristic Nature: Simple deterministic rules, NOT machine learning / AI.")
    print("  2. Decision Support: Intended to aid human judgment, not replace educator discretion.")
    print("  3. Non-Causal: Signals reflect past activity; intervention efficacy is not guaranteed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import joblib
    from src.data_loader import load_oulad_tables
    from src.problem_definition import scope_to_module, compute_target
    from src.feature_engineering import engineer_features
    from src.model import prepare_data, split_data
    from src.predict import generate_predictions

    print("=" * 80)
    print("PHASE 12: RULE-BASED INTERVENTION RECOMMENDATIONS VERIFICATION")
    print("=" * 80)
    print("Confirming operational boundaries:")
    print("  - Rule-based logic (heuristic if/else conditions).")
    print("  - NOT an ML model, NOT an AI prediction.")
    print("  - Recommendations have NOT been experimentally validated.")
    print("  - Models and thresholds are FROZEN; holdout test set is UNTOUCHED.")
    print("=" * 80)

    # 1. Load frozen model artifact without retraining or tuning
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    artifact = joblib.load(model_path)
    best_pipeline = artifact["pipeline"]

    # 2. Recreate feature data strictly from Day 0-14 observation window
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

    # Use training split (X_train) for demonstration to ensure test set remains untouched
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    student_ids_train = list(X_train.index)
    train_predictions = generate_predictions(
        best_pipeline=best_pipeline,
        X=X_train,
        student_ids=student_ids_train,
    )

    # -------------------------------------------------------------------------
    # Demonstrate Required Target Scenarios
    # -------------------------------------------------------------------------
    # Join features with predictions to locate specific archetypes
    merged = train_predictions.merge(
        features_df, left_on="id_student", right_index=True
    )

    # 1. LOW-risk student
    low_candidates = merged[merged["risk_level"] == "LOW"]
    low_student_id = low_candidates.iloc[0]["id_student"]

    # 2. MEDIUM-risk student
    med_candidates = merged[merged["risk_level"] == "MEDIUM"]
    med_student_id = med_candidates.iloc[0]["id_student"]

    # 3. HIGH-risk student
    high_candidates = merged[merged["risk_level"] == "HIGH"]
    high_student_id = high_candidates.iloc[0]["id_student"]

    # 4. HIGH-risk student with longest_inactivity_gap >= 7
    high_gap_candidates = merged[
        (merged["risk_level"] == "HIGH") & (merged["longest_inactivity_gap"] >= 7)
    ]
    high_gap_student_id = high_gap_candidates.iloc[0]["id_student"]

    print("\n" + "#" * 80)
    print("DEMONSTRATION 1: LOW-Risk Student")
    print("#" * 80)
    low_interv = get_intervention(low_student_id, train_predictions, features_df)
    print_recommendation_card(low_interv)

    print("#" * 80)
    print("DEMONSTRATION 2: MEDIUM-Risk Student")
    print("#" * 80)
    med_interv = get_intervention(med_student_id, train_predictions, features_df)
    print_recommendation_card(med_interv)

    print("#" * 80)
    print("DEMONSTRATION 3: HIGH-Risk Student (General)")
    print("#" * 80)
    high_interv = get_intervention(high_student_id, train_predictions, features_df)
    print_recommendation_card(high_interv)

    print("#" * 80)
    print("DEMONSTRATION 4: HIGH-Risk Student with Inactivity Gap >= 7 Days")
    print("#" * 80)
    high_gap_interv = get_intervention(
        high_gap_student_id, train_predictions, features_df
    )
    print_recommendation_card(high_gap_interv)

    print("#" * 80)
    print("DEMONSTRATION 5: Case Where Assessment Information is Unavailable")
    print("#" * 80)
    # Confirm 'submitted_any_early_assessment' is absent from AAA-2013J features_df
    assert "submitted_any_early_assessment" not in features_df.columns, (
        "Expected submitted_any_early_assessment to be absent in AAA-2013J feature matrix."
    )
    print("Status: 'submitted_any_early_assessment' is confirmed absent in features_df.")
    print("Evaluating fallback rule on representative student:")
    demo_fallback = get_intervention(med_student_id, train_predictions, features_df)
    print(f"Student ID:           {demo_fallback['id_student']}")
    print(f"Risk Level:           {demo_fallback['risk_level']}")
    print(f"Recommendation Type:  {demo_fallback['recommendation_type']}")
    print(f"Recommendation:       {demo_fallback['recommendation']}")
    print("Assessment Signal:    UNAVAILABLE (Handled cleanly without fabrication)")
    print("-" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Unit Verification: Synthetic Scenarios to Validate All Rule Branches
    # -------------------------------------------------------------------------
    print("#" * 80)
    print("DEMONSTRATION 6: Synthetic Assessment Signal Handling Verification")
    print("#" * 80)
    # Verify that when assessment information IS present, the assessment rules activate properly
    synthetic_preds = pd.DataFrame([
        {"id_student": 99901, "risk_level": "MEDIUM"},
        {"id_student": 99902, "risk_level": "HIGH"},
        {"id_student": 99903, "risk_level": "HIGH"},
    ])
    synthetic_features = pd.DataFrame([
        # Case A: MEDIUM risk, positive delta, submitted_any_early_assessment == 0 -> reminder
        {
            "id_student": 99901,
            "activity_delta": 5.0,
            "longest_inactivity_gap": 3.0,
            "submitted_any_early_assessment": 0,
        },
        # Case B: HIGH risk, gap < 7, negative delta, submitted_any_early_assessment == 0 -> urgent_outreach
        {
            "id_student": 99902,
            "activity_delta": -10.0,
            "longest_inactivity_gap": 4.0,
            "submitted_any_early_assessment": 0,
        },
        # Case C: HIGH risk, gap < 7, delta >= 0, submitted_any_early_assessment == 0 -> reminder
        {
            "id_student": 99903,
            "activity_delta": 2.0,
            "longest_inactivity_gap": 4.0,
            "submitted_any_early_assessment": 0,
        },
    ])

    synth_res_a = get_intervention(99901, synthetic_preds, synthetic_features)
    synth_res_b = get_intervention(99902, synthetic_preds, synthetic_features)
    synth_res_c = get_intervention(99903, synthetic_preds, synthetic_features)

    assert synth_res_a["recommendation_type"] == "reminder", f"Expected reminder, got {synth_res_a}"
    assert synth_res_b["recommendation_type"] == "urgent_outreach", f"Expected urgent_outreach, got {synth_res_b}"
    assert synth_res_c["recommendation_type"] == "reminder", f"Expected reminder, got {synth_res_c}"

    print(f"Synthetic Case A (MEDIUM, delta>=0, assessment=0):      -> {synth_res_a['recommendation_type']} [PASS]")
    print(f"Synthetic Case B (HIGH, gap<7, delta<0, assessment=0):  -> {synth_res_b['recommendation_type']} [PASS]")
    print(f"Synthetic Case C (HIGH, gap<7, delta>=0, assessment=0): -> {synth_res_c['recommendation_type']} [PASS]")
    print("-" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Edge Cases & Defensive Error Handling
    # -------------------------------------------------------------------------
    print("#" * 80)
    print("DEMONSTRATION 7: Edge Case and Defensive Error Handling")
    print("#" * 80)

    # 1. Missing student in predictions
    try:
        get_intervention(888888, train_predictions, features_df)
        print("Missing student in predictions: FAIL (did not raise KeyError)")
    except KeyError:
        print("Missing student in predictions: PASS (KeyError raised)")

    # 2. Missing student in features
    try:
        dummy_pred = pd.DataFrame([{"id_student": 777777, "risk_level": "LOW"}])
        get_intervention(777777, dummy_pred, features_df)
        print("Missing student in features: FAIL (did not raise KeyError)")
    except KeyError:
        print("Missing student in features: PASS (KeyError raised)")

    # 3. Invalid risk level
    try:
        dummy_pred = pd.DataFrame([{"id_student": low_student_id, "risk_level": "CRITICAL"}])
        get_intervention(low_student_id, dummy_pred, features_df)
        print("Invalid risk level: FAIL (did not raise ValueError)")
    except ValueError:
        print("Invalid risk level: PASS (ValueError raised)")

    # 4. Zero clicks and zero active days student
    zero_click_students = features_df[features_df["total_early_clicks"] == 0]
    if not zero_click_students.empty:
        z_sid = zero_click_students.index[0]
        if z_sid in train_predictions["id_student"].values:
            z_res = get_intervention(z_sid, train_predictions, features_df)
            print(f"Zero-click student ({z_sid}) handled cleanly: Risk={z_res['risk_level']}, Type={z_res['recommendation_type']} [PASS]")

    # -------------------------------------------------------------------------
    # Full Cohort Summary Generation
    # -------------------------------------------------------------------------
    all_recs = generate_recommendations(train_predictions, features_df)
    print_recommendation_summary_report(all_recs)

    print("Phase 12 execution and verification successfully completed!")
