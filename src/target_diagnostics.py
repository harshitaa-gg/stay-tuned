"""Phase 2 Target Definition Diagnostics for Stay-Tuned Disengagement Prediction.

This module performs diagnostic analysis on the scoped AAA-2013J cohort across
multiple dimensions of learner engagement:
1. Cohort eligibility (pre-Day 14 withdrawal exclusion).
2. Early engagement volume (Days 0-14).
3. Engagement decline ratio between observation (Days 0-14) and outcome (Days 15-42) windows.
4. Longest consecutive inactivity streaks during the outcome window (Days 15-42).
5. Assessment submissions and missed deadlines during the outcome window.
6. Official withdrawals during the outcome window (Days 15-42).

This script is strictly read-only and diagnostic; it does not train models or modify data.
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
from src.problem_definition import scope_to_module


# ==============================================================================
# DIAGNOSTIC 1: ELIGIBLE COHORT
# ==============================================================================
def diagnose_eligible_cohort(student_info_df, registration_df):
    """Determine and report the eligible student cohort after excluding early withdrawals.

    Parameters
    ----------
    student_info_df : pandas.DataFrame
        Scoped student_info table.
    registration_df : pandas.DataFrame
        Scoped studentRegistration table.

    Returns
    -------
    pandas.DataFrame
        Eligible cohort DataFrame with ['id_student', 'date_unregistration'].
    """
    students = student_info_df[["id_student"]].drop_duplicates()
    reg_subset = registration_df[["id_student", "date_unregistration"]].drop_duplicates(subset=["id_student"])
    cohort = students.merge(reg_subset, on="id_student", how="left")

    total_before = len(cohort)
    excluded_mask = cohort["date_unregistration"] <= 14
    excluded_count = int(excluded_mask.sum())

    eligible_df = cohort[~excluded_mask].copy()
    total_eligible = len(eligible_df)

    print("=" * 80)
    print("DIAGNOSTIC 1: ELIGIBLE COHORT (Prediction Point: End of Day 14)")
    print("=" * 80)
    print(f"Total students before exclusion:                     {total_before:,}")
    print(f"Excluded (unregistered on or before Day 14):          {excluded_count:,}")
    print(f"Total eligible students:                             {total_eligible:,}")
    print("=" * 80 + "\n")

    return eligible_df


# ==============================================================================
# DIAGNOSTIC 2: EARLY ENGAGEMENT (DAYS 0-14)
# ==============================================================================
def diagnose_early_engagement(eligible_df, student_vle_df):
    """Analyze clickstream engagement in the Day 0-14 observation window.

    Parameters
    ----------
    eligible_df : pandas.DataFrame
        Eligible cohort DataFrame.
    student_vle_df : pandas.DataFrame
        Scoped studentVle table.

    Returns
    -------
    pandas.DataFrame
        Eligible DataFrame merged with 'early_clicks'.
    """
    early_vle = student_vle_df[
        (student_vle_df["date"] >= 0) & (student_vle_df["date"] <= 14)
    ]
    early_clicks = (
        early_vle.groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "early_clicks"})
    )

    df = eligible_df.merge(early_clicks, on="id_student", how="left")
    df["early_clicks"] = df["early_clicks"].fillna(0).astype(int)

    clicks = df["early_clicks"]
    zero_count = int((clicks == 0).sum())
    active_count = int((clicks > 0).sum())
    total = len(df)

    print("=" * 80)
    print("DIAGNOSTIC 2: EARLY ENGAGEMENT (Day 0-14 Observation Window)")
    print("=" * 80)
    print(f"Minimum early clicks:                                {clicks.min():,}")
    print(f"Maximum early clicks:                                {clicks.max():,}")
    print(f"Mean early clicks:                                   {clicks.mean():.2f}")
    print(f"Median early clicks:                                 {clicks.median():.2f}")
    print(f"25th percentile:                                     {clicks.quantile(0.25):.2f}")
    print(f"75th percentile:                                     {clicks.quantile(0.75):.2f}")
    print("-" * 80)
    print(f"Students with 0 early clicks:                        {zero_count:>4} ({zero_count / total * 100:.2f}%)")
    print(f"Students with > 0 early clicks:                      {active_count:>4} ({active_count / total * 100:.2f}%)")
    print("=" * 80 + "\n")

    return df


# ==============================================================================
# DIAGNOSTIC 3: ENGAGEMENT DECLINE (OBSERVATION VS OUTCOME WINDOW)
# ==============================================================================
def diagnose_engagement_decline(eligible_df_with_early, student_vle_df):
    """Analyze click rate decline from Day 0-14 (15 days) to Day 15-42 (28 days).

    Parameters
    ----------
    eligible_df_with_early : pandas.DataFrame
        Eligible cohort DataFrame with 'early_clicks'.
    student_vle_df : pandas.DataFrame
        Scoped studentVle table.

    Returns
    -------
    tuple
        (DataFrame with decline metrics, decline_75th percentile float value)
    """
    late_vle = student_vle_df[
        (student_vle_df["date"] >= 15) & (student_vle_df["date"] <= 42)
    ]
    late_clicks = (
        late_vle.groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "late_clicks"})
    )

    df = eligible_df_with_early.merge(late_clicks, on="id_student", how="left")
    df["late_clicks"] = df["late_clicks"].fillna(0).astype(int)

    # Calculate daily rates
    df["early_rate"] = df["early_clicks"] / 15.0
    df["late_rate"] = df["late_clicks"] / 28.0

    # Calculate decline ratio only where early_rate > 0
    df["decline_ratio"] = np.nan
    mask_positive = df["early_rate"] > 0
    df.loc[mask_positive, "decline_ratio"] = (
        df.loc[mask_positive, "early_rate"] - df.loc[mask_positive, "late_rate"]
    ) / df.loc[mask_positive, "early_rate"]

    valid_declines = df.loc[mask_positive, "decline_ratio"]
    n_defined = len(valid_declines)
    decline_75th = float(valid_declines.quantile(0.75))
    above_75th_count = int((valid_declines >= decline_75th).sum())

    print("=" * 80)
    print("DIAGNOSTIC 3: ENGAGEMENT DECLINE (Day 0-14 vs Day 15-42)")
    print("=" * 80)
    print(f"Number of students with defined decline_ratio:       {n_defined:,} / {len(df):,}")
    print("-" * 80)
    print(f"Minimum decline ratio:                               {valid_declines.min():.4f}")
    print(f"Maximum decline ratio:                               {valid_declines.max():.4f}")
    print(f"Mean decline ratio:                                  {valid_declines.mean():.4f}")
    print(f"Median decline ratio (50th percentile):              {valid_declines.median():.4f}")
    print(f"25th percentile:                                     {valid_declines.quantile(0.25):.4f}")
    print(f"50th percentile:                                     {valid_declines.quantile(0.50):.4f}")
    print(f"75th percentile:                                     {decline_75th:.4f}")
    print("-" * 80)
    print(f"Exact 75th percentile value (decline_75th):          {decline_75th:.4f}")
    print(f"Number of students with decline_ratio >= 75th pct:   {above_75th_count:,} ({above_75th_count / n_defined * 100:.2f}% of defined)")
    print("=" * 80 + "\n")

    return df, decline_75th


# ==============================================================================
# DIAGNOSTIC 4: INACTIVITY STREAKS (DAY 15-42)
# ==============================================================================
def diagnose_inactivity_streaks(eligible_df, student_vle_df):
    """Compute longest consecutive streak of inactive days in the Day 15-42 outcome window.

    Parameters
    ----------
    eligible_df : pandas.DataFrame
        Eligible cohort DataFrame.
    student_vle_df : pandas.DataFrame
        Scoped studentVle table.

    Returns
    -------
    tuple
        (DataFrame with 'max_inactivity_streak', streak_75th percentile float value)
    """
    outcome_vle = student_vle_df[
        (student_vle_df["date"] >= 15) & (student_vle_df["date"] <= 42)
    ]

    # Find daily click sums per student
    daily_clicks = (
        outcome_vle.groupby(["id_student", "date"])["sum_click"]
        .sum()
        .reset_index()
    )

    # Active days per student where sum_click > 0
    active_daily = daily_clicks[daily_clicks["sum_click"] > 0]
    active_days_by_student = active_daily.groupby("id_student")["date"].apply(set).to_dict()

    streak_records = []
    for student_id in eligible_df["id_student"]:
        active_days = active_days_by_student.get(student_id, set())

        max_streak = 0
        current_streak = 0
        for day in range(15, 43):  # Day 15 to Day 42 inclusive (28 days)
            if day not in active_days:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0

        streak_records.append({"id_student": student_id, "max_inactivity_streak": max_streak})

    streak_df = pd.DataFrame(streak_records)
    df = eligible_df.merge(streak_df, on="id_student", how="left")

    streaks = df["max_inactivity_streak"]
    streak_75th = float(streaks.quantile(0.75))
    total = len(df)

    streak_0 = int((streaks == 0).sum())
    streak_gte_7 = int((streaks >= 7).sum())
    streak_gte_14 = int((streaks >= 14).sum())

    print("=" * 80)
    print("DIAGNOSTIC 4: INACTIVITY STREAKS (Day 15-42 Outcome Window, 28 Days Total)")
    print("=" * 80)
    print(f"Minimum streak:                                      {streaks.min():,} days")
    print(f"Maximum streak:                                      {streaks.max():,} days")
    print(f"Mean streak:                                         {streaks.mean():.2f} days")
    print(f"Median streak (50th percentile):                     {streaks.median():.2f} days")
    print(f"25th percentile:                                     {streaks.quantile(0.25):.2f} days")
    print(f"50th percentile:                                     {streaks.quantile(0.50):.2f} days")
    print(f"75th percentile:                                     {streak_75th:.2f} days")
    print("-" * 80)
    print(f"Students with inactivity streak = 0 days:            {streak_0:>4} ({streak_0 / total * 100:.2f}%)")
    print(f"Students with inactivity streak >= 7 days:           {streak_gte_7:>4} ({streak_gte_7 / total * 100:.2f}%)")
    print(f"Students with inactivity streak >= 14 days:          {streak_gte_14:>4} ({streak_gte_14 / total * 100:.2f}%)")
    print("-" * 80)
    print(f"Exact 75th percentile of inactivity streak:          {streak_75th:.2f} days")
    print("=" * 80 + "\n")

    return df, streak_75th


# ==============================================================================
# DIAGNOSTIC 5: ASSESSMENTS IN THE OUTCOME WINDOW
# ==============================================================================
def diagnose_outcome_assessments(eligible_df, assessments_df, student_assessment_df):
    """Analyze assessment deadlines and student submission rates in Days 15-42.

    Parameters
    ----------
    eligible_df : pandas.DataFrame
        Eligible cohort DataFrame.
    assessments_df : pandas.DataFrame
        Raw or scoped assessments table.
    student_assessment_df : pandas.DataFrame
        Scoped student_assessment table.

    Returns
    -------
    tuple
        (Number of window assessments, missed assessment count, missed percentage)
    """
    # Filter assessments for AAA-2013J due in Days 15-42
    window_assessments = assessments_df[
        (assessments_df["code_module"] == "AAA")
        & (assessments_df["code_presentation"] == "2013J")
        & (assessments_df["date"] >= 15)
        & (assessments_df["date"] <= 42)
    ].copy()

    n_assessments = len(window_assessments)

    print("=" * 80)
    print("DIAGNOSTIC 5: ASSESSMENTS IN OUTCOME WINDOW (Days 15-42)")
    print("=" * 80)
    print(f"Number of assessments due in Days 15-42:             {n_assessments}")
    print("-" * 80)

    if n_assessments > 0:
        print(f"{'id_assessment':<15} | {'assessment_type':<16} | {'date (due day)':<16} | {'weight'}")
        print("-" * 80)
        for _, row in window_assessments.iterrows():
            print(f"{row['id_assessment']:<15} | {row['assessment_type']:<16} | {row['date']:<16} | {row['weight']}")
        print("-" * 80)

        # Check which eligible students submitted each assessment
        window_assessment_ids = set(window_assessments["id_assessment"])
        
        # Submissions by eligible students for these specific assessments
        submissions = student_assessment_df[
            student_assessment_df["id_assessment"].isin(window_assessment_ids)
            & student_assessment_df["id_student"].isin(eligible_df["id_student"])
        ]

        # For each student, check if they have submitted all assessments in this window
        # A student misses an assessment if they did not submit any of the due assessments
        submitted_students_per_assessment = {
            aid: set(submissions[submissions["id_assessment"] == aid]["id_student"])
            for aid in window_assessment_ids
        }

        missed_students = set()
        for student_id in eligible_df["id_student"]:
            for aid in window_assessment_ids:
                if student_id not in submitted_students_per_assessment[aid]:
                    missed_students.add(student_id)
                    break

        n_missed = len(missed_students)
        pct_missed = (n_missed / len(eligible_df)) * 100
    else:
        print("No assessments are scheduled in the Day 15-42 window for AAA-2013J.")
        n_missed = 0
        pct_missed = 0.0

    print(f"Eligible students who missed >= 1 assessment in window: {n_missed:,} ({pct_missed:.2f}%)")
    print("=" * 80 + "\n")

    return n_assessments, n_missed, pct_missed


# ==============================================================================
# DIAGNOSTIC 6: WITHDRAWALS
# ==============================================================================
def diagnose_withdrawals(eligible_df):
    """Report unregistrations/withdrawals occurring strictly within Days 15-42.

    Parameters
    ----------
    eligible_df : pandas.DataFrame
        Eligible cohort DataFrame.

    Returns
    -------
    int
        Count of students withdrawing during Days 15-42.
    """
    withdrawals_mask = (
        (eligible_df["date_unregistration"] >= 15)
        & (eligible_df["date_unregistration"] <= 42)
    )
    n_withdrawn = int(withdrawals_mask.sum())
    total = len(eligible_df)

    print("=" * 80)
    print("DIAGNOSTIC 6: WITHDRAWALS (Day 15-42 Outcome Window)")
    print("=" * 80)
    print(f"Eligible students who withdrew during Days 15-42:     {n_withdrawn:,} ({n_withdrawn / total * 100:.2f}%)")
    print("=" * 80 + "\n")

    return n_withdrawn


# ==============================================================================
# MAIN EXECUTION & SUMMARY
# ==============================================================================
def run_all_diagnostics():
    """Run all 6 target diagnostics for AAA-2013J and print the final summary."""
    # 1. Resolve raw data directory
    data_dir = PROJECT_ROOT / "data" / "raw"

    # 2. Load raw tables
    tables = load_oulad_tables(data_dir)

    # 3. Scope to AAA-2013J
    scoped = scope_to_module(tables, code_module="AAA", code_presentation="2013J")

    # Diagnostic 1: Eligible Cohort
    eligible_df = diagnose_eligible_cohort(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
    )

    # Diagnostic 2: Early Engagement (Day 0-14)
    early_df = diagnose_early_engagement(
        eligible_df=eligible_df,
        student_vle_df=scoped["student_vle"],
    )

    # Diagnostic 3: Engagement Decline (Observation vs Outcome)
    decline_df, decline_75th = diagnose_engagement_decline(
        eligible_df_with_early=early_df,
        student_vle_df=scoped["student_vle"],
    )

    # Diagnostic 4: Inactivity Streaks (Day 15-42)
    streak_df, streak_75th = diagnose_inactivity_streaks(
        eligible_df=eligible_df,
        student_vle_df=scoped["student_vle"],
    )

    # Diagnostic 5: Assessments in Outcome Window
    n_assessments, n_missed_assessments, pct_missed = diagnose_outcome_assessments(
        eligible_df=eligible_df,
        assessments_df=tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
    )

    # Diagnostic 6: Withdrawals (Day 15-42)
    n_withdrawn = diagnose_withdrawals(eligible_df)

    # Final Concise Summary
    total_eligible = len(eligible_df)
    zero_early = int((early_df["early_clicks"] == 0).sum())
    valid_declines = decline_df["decline_ratio"].dropna()
    above_decline_75th = int((valid_declines >= decline_75th).sum())

    print("=" * 80)
    print("FINAL SUMMARY OF TARGET DIAGNOSTIC METRICS")
    print("=" * 80)
    print(f"- Total eligible students:                          {total_eligible:,}")
    print(f"- Students with zero early activity (Days 0-14):    {zero_early:,}")
    print(f"- 75th percentile decline_ratio:                    {decline_75th:.4f}")
    print(f"- Number above the decline 75th percentile:         {above_decline_75th:,}")
    print(f"- 75th percentile inactivity streak:                {streak_75th:.2f} days")
    print(f"- Number of assessments due during Days 15-42:      {n_assessments}")
    print(f"- Number of students who missed >= 1 assessment:    {n_missed_assessments:,} ({pct_missed:.2f}%)")
    print(f"- Number who withdrew during Days 15-42:            {n_withdrawn:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_diagnostics()
