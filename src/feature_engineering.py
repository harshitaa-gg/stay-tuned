"""Phase 4: Temporal Feature Engineering module for Stay-Tuned disengagement prediction.

This module derives 8 numerical features per learner strictly from the Day 0–14 observation
window. No data after Day 14 or from the outcome window (Days 15–42) is accessed or used.

Assessment Features Policy:
---------------------------
Phase 3 EDA confirmed that for Module AAA, Presentation 2013J, zero assessments are scheduled
with due dates in Days 0–14 (the first milestone, TMA 01, is due on Day 19). Therefore, early
assessment submission and score features are omitted to avoid uninformative zero-variance inputs.

Zero-Variance Features Policy:
------------------------------
'num_of_prev_attempts' is 0 for all 372 eligible students in AAA-2013J (zero variance / std = 0.00).
It is explicitly excluded from the model feature matrix to prevent uninformative constant inputs.

Responsible AI Policy:
----------------------
Demographic and background attributes ('gender', 'region', 'imd_band', 'age_band', 'disability',
'highest_education') are strictly excluded from predictive features to prevent algorithmic bias.
'final_result' is an end-of-course outcome and is strictly excluded to prevent label leakage.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data_loader import load_oulad_tables
from src.problem_definition import scope_to_module, compute_target


def compute_longest_inactivity_gap(active_days_set, obs_start_day=0, obs_end_day=14):
    """Compute the longest consecutive sequence of days with zero clicks in the observation window.

    Parameters
    ----------
    active_days_set : set
        Set of day integers where the student had at least one click (sum_click > 0).
    obs_start_day : int, default=0
        Start day of observation window.
    obs_end_day : int, default=14
        End day of observation window (inclusive).

    Returns
    -------
    int
        Longest consecutive inactive days (0 to 15 for a 15-day window).
    """
    max_gap = 0
    curr_gap = 0
    for day in range(obs_start_day, obs_end_day + 1):
        if day not in active_days_set:
            curr_gap += 1
            if curr_gap > max_gap:
                max_gap = curr_gap
        else:
            curr_gap = 0
    return max_gap


def engineer_features(student_vle_df, student_info_df, obs_end_day=14):
    """Extract strictly temporal, leakage-free behavioral and academic context features (Days 0-14).

    Parameters
    ----------
    student_vle_df : pandas.DataFrame
        Scoped studentVle table.
    student_info_df : pandas.DataFrame
        Scoped studentInfo table (filtered to eligible cohort).
    obs_end_day : int, default=14
        Observation window upper bound (inclusive).

    Returns
    -------
    pandas.DataFrame
        Feature matrix indexed by 'id_student' containing exactly 8 numerical features:
        1. total_early_clicks
        2. clicks_per_active_day
        3. active_days
        4. longest_inactivity_gap
        5. days_since_last_active
        6. activity_delta
        7. weekly_click_ratio
        8. studied_credits
    """
    # 1. Enforce strict temporal boundary: Days 0 to obs_end_day inclusive
    early_vle = student_vle_df[
        (student_vle_df["date"] >= 0) & (student_vle_df["date"] <= obs_end_day)
    ].copy()

    # Verify no future date leaked into early_vle
    if not early_vle.empty:
        max_date_in_vle = int(early_vle["date"].max())
        if max_date_in_vle > obs_end_day:
            raise ValueError(
                f"Temporal leakage violation! Found date {max_date_in_vle} > {obs_end_day}."
            )

    # 2. Get eligible student cohort and academic context features (studied_credits only)
    base_students = (
        student_info_df[["id_student", "studied_credits"]]
        .drop_duplicates(subset=["id_student"])
        .copy()
    )

    # 3. Aggregate activity metrics per student
    # A. Total clicks in Days 0-14
    total_clicks_df = (
        early_vle.groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "total_early_clicks"})
    )

    # B. Active daily interactions (days with sum_click > 0)
    active_daily = early_vle[early_vle["sum_click"] > 0]
    
    # Active days count
    active_days_df = (
        active_daily.groupby("id_student")["date"]
        .nunique()
        .reset_index()
        .rename(columns={"date": "active_days"})
    )

    # Map of active days set per student for inactivity gap & recency
    active_days_by_student = (
        active_daily.groupby("id_student")["date"]
        .apply(set)
        .to_dict()
    )

    # Last active day per student
    last_active_by_student = (
        active_daily.groupby("id_student")["date"]
        .max()
        .to_dict()
    )

    # C. Week 1 (Days 0-6) and Week 2 (Days 7-14) clicks for trend features
    w1_df = (
        early_vle[early_vle["date"] <= 6]
        .groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "week1_clicks"})
    )

    w2_df = (
        early_vle[(early_vle["date"] >= 7) & (early_vle["date"] <= obs_end_day)]
        .groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "week2_clicks"})
    )

    # 4. Merge all components onto base_students
    features_df = (
        base_students
        .merge(total_clicks_df, on="id_student", how="left")
        .merge(active_days_df, on="id_student", how="left")
        .merge(w1_df, on="id_student", how="left")
        .merge(w2_df, on="id_student", how="left")
    )

    # Fill zero activity values
    features_df["total_early_clicks"] = features_df["total_early_clicks"].fillna(0).astype(int)
    features_df["active_days"] = features_df["active_days"].fillna(0).astype(int)
    features_df["week1_clicks"] = features_df["week1_clicks"].fillna(0).astype(int)
    features_df["week2_clicks"] = features_df["week2_clicks"].fillna(0).astype(int)

    # 5. Compute derived features
    # Intensity: clicks_per_active_day
    features_df["clicks_per_active_day"] = (
        features_df["total_early_clicks"] / features_df["active_days"].clip(lower=1)
    ).astype(float)
    # Ensure students with 0 active days get 0.0
    features_df.loc[features_df["active_days"] == 0, "clicks_per_active_day"] = 0.0

    # Inactivity gap and recency (days_since_last_active)
    gap_list = []
    recency_list = []
    for s_id in features_df["id_student"]:
        act_days = active_days_by_student.get(s_id, set())
        gap = compute_longest_inactivity_gap(act_days, obs_start_day=0, obs_end_day=obs_end_day)
        gap_list.append(gap)

        if s_id in last_active_by_student:
            recency = obs_end_day - last_active_by_student[s_id]
        else:
            # 0 active days: default to full window duration (15 days)
            recency = obs_end_day + 1
        recency_list.append(recency)

    features_df["longest_inactivity_gap"] = gap_list
    features_df["days_since_last_active"] = recency_list

    # Trend: activity_delta and weekly_click_ratio
    features_df["activity_delta"] = (
        features_df["week2_clicks"] - features_df["week1_clicks"]
    ).astype(int)

    features_df["weekly_click_ratio"] = (
        (features_df["week2_clicks"] + 1.0) / (features_df["week1_clicks"] + 1.0)
    ).astype(float)

    # 6. Select and order exactly the 8 specified model features
    feature_columns = [
        "total_early_clicks",
        "clicks_per_active_day",
        "active_days",
        "longest_inactivity_gap",
        "days_since_last_active",
        "activity_delta",
        "weekly_click_ratio",
        "studied_credits",
    ]

    final_df = features_df.set_index("id_student")[feature_columns].copy()
    return final_df


def print_feature_validation_report(features_df, max_vle_date_used):
    """Print a comprehensive Phase 4 validation report for the engineered feature matrix.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Engineered feature matrix.
    max_vle_date_used : int
        Maximum date value observed in the filtered VLE data used for feature extraction.
    """
    n_students, n_features = features_df.shape
    zero_click_students = int((features_df["total_early_clicks"] == 0).sum())
    total_nulls = int(features_df.isnull().sum().sum())
    min_gap = int(features_df["longest_inactivity_gap"].min())
    max_gap = int(features_df["longest_inactivity_gap"].max())
    neg_delta_count = int((features_df["activity_delta"] < 0).sum())

    # Detect zero-variance features
    zero_var_cols = [col for col in features_df.columns if features_df[col].std() == 0]

    print("\n" + "=" * 80)
    print("PHASE 4: FEATURE ENGINEERING VALIDATION REPORT")
    print("=" * 80)
    print(f"Total Eligible Students in Matrix:   {n_students:,} (Expected: 372)")
    print(f"Total Model Features:                {n_features} (Expected: 8)")
    print(f"Total Missing Values in Matrix:      {total_nulls} (Expected: 0)")
    print("-" * 80)
    print("Engineered Model Features:")
    for i, col in enumerate(features_df.columns, 1):
        dtype_str = str(features_df[col].dtype)
        mean_val = features_df[col].mean()
        std_val = features_df[col].std()
        print(f"  {i}. {col:<26} | dtype: {dtype_str:<8} | mean: {mean_val:8.2f} | std: {std_val:8.2f}")
    print("-" * 80)
    print("Quality & Distribution Checks:")
    print(f"  - Students with zero early clicks:           {zero_click_students:>4} ({zero_click_students / n_students * 100:.2f}%)")
    print(f"  - Longest inactivity gap range:              [{min_gap}, {max_gap}] days (Valid: 0 to 15)")
    print(f"  - Students with negative activity delta:     {neg_delta_count:>4} ({neg_delta_count / n_students * 100:.2f}%)")
    print(f"  - Maximum VLE date used in features:         Day {max_vle_date_used} (Confirmed <= 14)")
    print(f"  - Zero-variance features:                    {zero_var_cols if zero_var_cols else 'None (All 8 features have variance)'}")
    print("-" * 80)

    # Assertions for safety
    assert n_students == 372, f"Expected 372 students, got {n_students}"
    assert n_features == 8, f"Expected 8 features, got {n_features}"
    assert total_nulls == 0, f"Expected 0 missing values, got {total_nulls}"
    assert 0 <= min_gap <= max_gap <= 15, "Inactivity gap out of bounds [0, 15]"
    assert max_vle_date_used <= 14, "Temporal leakage! VLE date > 14"
    assert len(zero_var_cols) == 0, f"Found zero-variance features: {zero_var_cols}"

    print("ALL PHASE 4 QUALITY CHECKS PASSED: Final 8-feature matrix is clean, leakage-free, and ready.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 1. Resolve raw data directory
    data_dir = PROJECT_ROOT / "data" / "raw"

    # 2. Load raw tables
    raw_tables = load_oulad_tables(data_dir)

    # 3. Scope to AAA-2013J
    scoped = scope_to_module(raw_tables, code_module="AAA", code_presentation="2013J")

    # 4. Compute target to resolve the 372 eligible students
    target_df = compute_target(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
        student_vle_df=scoped["student_vle"],
        assessments_df=raw_tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
        code_module="AAA",
        code_presentation="2013J",
    )

    # Filter scoped student_info to only eligible students
    eligible_student_info = scoped["student_info"][
        scoped["student_info"]["id_student"].isin(target_df["id_student"])
    ].copy()

    # 5. Extract temporal features (Days 0-14)
    features_matrix = engineer_features(
        student_vle_df=scoped["student_vle"],
        student_info_df=eligible_student_info,
        obs_end_day=14,
    )

    # 6. Verify max VLE date used
    early_vle_check = scoped["student_vle"][
        (scoped["student_vle"]["date"] >= 0) & (scoped["student_vle"]["date"] <= 14)
    ]
    max_vle_date = int(early_vle_check["date"].max()) if not early_vle_check.empty else 0

    # 7. Print validation report
    print_feature_validation_report(features_matrix, max_vle_date)
