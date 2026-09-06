"""Cohort scoping module for Stay-Tuned disengagement prediction system (Phase 2A).

This module filters the raw OULAD tables to a specific module and presentation
(default: AAA, 2013J) without modifying the original DataFrames in-place.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import load_oulad_tables


def scope_to_module(tables, code_module="AAA", code_presentation="2013J"):
    """Filter OULAD tables to the specified module and presentation cohort.

    Parameters
    ----------
    tables : dict
        Dictionary of raw OULAD DataFrames containing at least:
        - 'student_info'
        - 'registration'
        - 'student_vle'
        - 'student_assessment'
        - 'assessments' (used to resolve assessment IDs for student_assessment)
    code_module : str, default="AAA"
        The module code to scope to.
    code_presentation : str, default="2013J"
        The presentation code to scope to.

    Returns
    -------
    dict
        A new dictionary containing the scoped DataFrames:
        - 'student_info'
        - 'registration'
        - 'student_vle'
        - 'student_assessment'
    """
    scoped_tables = {}

    # 1. Scope student_info
    student_info_df = tables["student_info"]
    scoped_tables["student_info"] = student_info_df[
        (student_info_df["code_module"] == code_module)
        & (student_info_df["code_presentation"] == code_presentation)
    ].copy()

    # 2. Scope registration (studentRegistration)
    registration_df = tables["registration"]
    scoped_tables["registration"] = registration_df[
        (registration_df["code_module"] == code_module)
        & (registration_df["code_presentation"] == code_presentation)
    ].copy()

    # 3. Scope student_vle
    student_vle_df = tables["student_vle"]
    scoped_tables["student_vle"] = student_vle_df[
        (student_vle_df["code_module"] == code_module)
        & (student_vle_df["code_presentation"] == code_presentation)
    ].copy()

    # 4. Scope student_assessment via valid assessment IDs in the assessments table
    assessments_df = tables["assessments"]
    valid_assessment_ids = assessments_df[
        (assessments_df["code_module"] == code_module)
        & (assessments_df["code_presentation"] == code_presentation)
    ]["id_assessment"]
    scoped_tables["student_assessment"] = tables["student_assessment"][
        tables["student_assessment"]["id_assessment"].isin(valid_assessment_ids)
    ].copy()

    return scoped_tables


def print_scoping_validation(scoped_tables, code_module="AAA", code_presentation="2013J", raw_assessments_df=None):
    """Print validation information for the scoped tables.

    Parameters
    ----------
    scoped_tables : dict
        Dictionary of filtered DataFrames.
    code_module : str
        The target module code.
    code_presentation : str
        The target presentation code.
    raw_assessments_df : pandas.DataFrame, optional
        The assessments DataFrame to map assessment IDs if needed.
    """
    print("\n" + "=" * 80)
    print("PHASE 2A: COHORT SCOPING VALIDATION REPORT")
    print("=" * 80)
    print(f"Selected Module:       {code_module}")
    print(f"Selected Presentation: {code_presentation}")
    print(f"Total Scoped Students: {len(scoped_tables['student_info']):,}")
    print("-" * 80)

    print(f"{'Table Key':<22} | {'Shape':<18} | {'Unique Modules':<16} | {'Unique Presentations'}")
    print("-" * 80)

    for key, df in scoped_tables.items():
        shape_str = f"{df.shape[0]:,} rows, {df.shape[1]} cols"

        if "code_module" in df.columns and "code_presentation" in df.columns:
            mod_str = ", ".join(map(str, df["code_module"].unique()))
            pres_str = ", ".join(map(str, df["code_presentation"].unique()))
        elif key == "student_assessment" and raw_assessments_df is not None:
            # Derive module and presentation from id_assessment mapping
            mapped = df.merge(raw_assessments_df, on="id_assessment", how="left")
            mod_str = ", ".join(map(str, mapped["code_module"].unique()))
            pres_str = ", ".join(map(str, mapped["code_presentation"].unique()))
        else:
            mod_str = "N/A (by id)"
            pres_str = "N/A (by id)"

        print(f"{key:<22} | {shape_str:<18} | {mod_str:<16} | {pres_str}")

    print("=" * 80 + "\n")


def compute_target(
    student_info_df,
    registration_df,
    student_vle_df,
    assessments_df,
    student_assessment_df,
    code_module="AAA",
    code_presentation="2013J",
):
    """Compute the finalized early disengagement target label for the scoped cohort (Phase 2B).

    Temporal setup:
    - Observation window: Days 0 to 14
    - Prediction point: End of Day 14
    - Outcome window: Days 15 to 42

    Disengagement Criteria (label = 1 if ANY condition is met):
    1. Official Withdrawal: date_unregistration is between Day 15 and Day 42 inclusive.
    2. Prolonged Inactivity: Longest zero-click streak during Days 15-42 is >= 14 days.
    3. Missed Academic Milestone: Did not submit TMA 01 (due on Day 19 for AAA-2013J).

    Parameters
    ----------
    student_info_df : pandas.DataFrame
        Scoped student_info table.
    registration_df : pandas.DataFrame
        Scoped studentRegistration table with 'date_unregistration'.
    student_vle_df : pandas.DataFrame
        Scoped studentVle table with 'date' and 'sum_click'.
    assessments_df : pandas.DataFrame
        Assessments metadata table.
    student_assessment_df : pandas.DataFrame
        Scoped studentAssessment submissions table.
    code_module : str, default="AAA"
        Module code.
    code_presentation : str, default="2013J"
        Presentation code.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        ['id_student', 'label', 'withdrew_in_window', 'inactivity_streak',
         'prolonged_inactivity', 'missed_assessment']
    """
    # 1. Total cohort before exclusion
    students = student_info_df[["id_student"]].drop_duplicates()
    reg_subset = registration_df[["id_student", "date_unregistration"]].drop_duplicates(subset=["id_student"])
    cohort = students.merge(reg_subset, on="id_student", how="left")

    total_before = len(cohort)

    # 2. Exclude students who unregistered on or before Day 14
    early_unreg_mask = cohort["date_unregistration"] <= 14
    excluded_count = int(early_unreg_mask.sum())

    eligible = cohort[~early_unreg_mask].copy()
    total_eligible = len(eligible)

    # 3. Condition 1: Official Withdrawal in Days 15-42
    eligible["withdrew_in_window"] = eligible["date_unregistration"].apply(
        lambda d: True if (pd.notna(d) and 15 <= d <= 42) else False
    )

    # 4. Condition 2: Prolonged Inactivity (streak >= 14 days in Days 15-42)
    outcome_vle = student_vle_df[
        (student_vle_df["date"] >= 15)
        & (student_vle_df["date"] <= 42)
        & (student_vle_df["sum_click"] > 0)
    ]
    active_days_by_student = outcome_vle.groupby("id_student")["date"].apply(set).to_dict()

    streak_records = []
    for student_id in eligible["id_student"]:
        active_days = active_days_by_student.get(student_id, set())

        max_streak = 0
        current_streak = 0
        for day in range(15, 43):  # 28 days: Day 15 to Day 42 inclusive
            if day not in active_days:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0

        streak_records.append({
            "id_student": student_id,
            "inactivity_streak": max_streak,
            "prolonged_inactivity": (max_streak >= 14),
        })

    streak_df = pd.DataFrame(streak_records)
    eligible = eligible.merge(streak_df, on="id_student", how="left")

    # 5. Condition 3: Missed Academic Milestone (TMA due on Day 19)
    # Dynamically find the TMA due on Day 19 for this module/presentation
    day19_assessments = assessments_df[
        (assessments_df["code_module"] == code_module)
        & (assessments_df["code_presentation"] == code_presentation)
        & (assessments_df["assessment_type"] == "TMA")
        & (assessments_df["date"] == 19)
    ]
    day19_assessment_ids = set(day19_assessments["id_assessment"])

    submitted_students = set(
        student_assessment_df[
            student_assessment_df["id_assessment"].isin(day19_assessment_ids)
        ]["id_student"]
    )

    eligible["missed_assessment"] = ~eligible["id_student"].isin(submitted_students)

    # 6. Final Disengagement Target (label = 1 if ANY condition met)
    eligible["label"] = (
        eligible["withdrew_in_window"]
        | eligible["prolonged_inactivity"]
        | eligible["missed_assessment"]
    ).astype(int)

    # Select required columns
    result_df = eligible[
        [
            "id_student",
            "label",
            "withdrew_in_window",
            "inactivity_streak",
            "prolonged_inactivity",
            "missed_assessment",
        ]
    ].copy()

    # 7. Validation Output & Overlap Reporting
    disengaged_count = int((result_df["label"] == 1).sum())
    non_disengaged_count = int((result_df["label"] == 0).sum())
    disengagement_pct = (disengaged_count / total_eligible * 100) if total_eligible > 0 else 0.0

    count_withdrew = int(result_df["withdrew_in_window"].sum())
    count_prolonged = int(result_df["prolonged_inactivity"].sum())
    count_missed_tma = int(result_df["missed_assessment"].sum())

    print("\n" + "=" * 80)
    print("PHASE 2B: FINAL DISENGAGEMENT TARGET VALIDATION REPORT")
    print("=" * 80)
    print(f"Total students before exclusion:                     {total_before:,}")
    print(f"Excluded (unregistered on or before Day 14):          {excluded_count:,}")
    print(f"Total eligible students:                             {total_eligible:,}")
    print("-" * 80)
    print("Breakdown of Individual Trigger Conditions:")
    print(f"  1. Official Withdrawal (Days 15-42):               {count_withdrew:>4} students ({count_withdrew / total_eligible * 100:.2f}%)")
    print(f"  2. Prolonged Inactivity (Streak >= 14 days):       {count_prolonged:>4} students ({count_prolonged / total_eligible * 100:.2f}%)")
    print(f"  3. Missed Academic Milestone (TMA 01 on Day 19):   {count_missed_tma:>4} students ({count_missed_tma / total_eligible * 100:.2f}%)")
    print("-" * 80)
    print(f"Final Disengaged Students (label = 1, Unique):       {disengaged_count:>4} ({disengagement_pct:.2f}%)")
    print(f"Final Non-disengaged Students (label = 0):           {non_disengaged_count:>4} ({100 - disengagement_pct:.2f}%)")
    print("=" * 80 + "\n")

    return result_df


def diagnose_outcome_activity(student_vle_df, registration_df, student_info_df):
    """Diagnose learner outcome activity and unregistration patterns (Days 15-42).

    Parameters
    ----------
    student_vle_df : pandas.DataFrame
        Scoped studentVle table.
    registration_df : pandas.DataFrame
        Scoped studentRegistration table.
    student_info_df : pandas.DataFrame
        Scoped studentInfo table.
    """
    # 1. Scoped cohort and unregistration merge
    students = student_info_df[["id_student"]].drop_duplicates()
    reg_subset = registration_df[["id_student", "date_unregistration"]].drop_duplicates(subset=["id_student"])
    cohort = students.merge(reg_subset, on="id_student", how="left")

    total_scoped = len(cohort)

    # 2. Early unregistrations (date_unregistration <= 14)
    early_unreg_mask = cohort["date_unregistration"] <= 14
    early_unreg_count = int(early_unreg_mask.sum())

    # 3. Eligible students at Day 14
    eligible = cohort[~early_unreg_mask].copy()
    total_eligible = len(eligible)

    # 4. Outcome clicks in Days 15-42
    outcome_vle = student_vle_df[
        (student_vle_df["date"] >= 15) & (student_vle_df["date"] <= 42)
    ]
    clicks_per_student = (
        outcome_vle.groupby("id_student")["sum_click"]
        .sum()
        .reset_index()
        .rename(columns={"sum_click": "outcome_clicks"})
    )
    eligible = eligible.merge(clicks_per_student, on="id_student", how="left")
    eligible["outcome_clicks"] = eligible["outcome_clicks"].fillna(0).astype(int)

    # Click brackets among eligible students
    c_0 = int((eligible["outcome_clicks"] == 0).sum())
    c_1_5 = int(((eligible["outcome_clicks"] >= 1) & (eligible["outcome_clicks"] <= 5)).sum())
    c_6_10 = int(((eligible["outcome_clicks"] >= 6) & (eligible["outcome_clicks"] <= 10)).sum())
    c_11_25 = int(((eligible["outcome_clicks"] >= 11) & (eligible["outcome_clicks"] <= 25)).sum())
    c_26_50 = int(((eligible["outcome_clicks"] >= 26) & (eligible["outcome_clicks"] <= 50)).sum())
    c_gt_50 = int((eligible["outcome_clicks"] > 50).sum())

    # Descriptive statistics for outcome_clicks
    click_count = int(eligible["outcome_clicks"].count())
    click_mean = float(eligible["outcome_clicks"].mean())
    click_median = float(eligible["outcome_clicks"].median())
    click_min = int(eligible["outcome_clicks"].min())
    click_max = int(eligible["outcome_clicks"].max())

    # Unregistration distribution among eligible students
    unreg_15_42 = int(((eligible["date_unregistration"] >= 15) & (eligible["date_unregistration"] <= 42)).sum())
    unreg_after_42 = int((eligible["date_unregistration"] > 42).sum())
    unreg_none = int(eligible["date_unregistration"].isna().sum())

    # Print diagnostic report
    print("=" * 80)
    print("PHASE 2B: OUTCOME ACTIVITY & WITHDRAWAL DIAGNOSTIC REPORT (Days 15-42)")
    print("=" * 80)
    print(f"1. Total Students in Scoped Cohort:                  {total_scoped:,}")
    print(f"2. Students Unregistered on or before Day 14:        {early_unreg_count:,}")
    print(f"   Eligible Students at Day 14:                      {total_eligible:,}")
    print("-" * 80)
    print("3. VLE Click Distribution during Days 15-42 (Eligible Students):")
    print(f"   - 0 clicks:                                       {c_0:>4} students ({c_0 / total_eligible * 100:.2f}%)")
    print(f"   - 1-5 clicks:                                     {c_1_5:>4} students ({c_1_5 / total_eligible * 100:.2f}%)")
    print(f"   - 6-10 clicks:                                    {c_6_10:>4} students ({c_6_10 / total_eligible * 100:.2f}%)")
    print(f"   - 11-25 clicks:                                   {c_11_25:>4} students ({c_11_25 / total_eligible * 100:.2f}%)")
    print(f"   - 26-50 clicks:                                   {c_26_50:>4} students ({c_26_50 / total_eligible * 100:.2f}%)")
    print(f"   - > 50 clicks:                                    {c_gt_50:>4} students ({c_gt_50 / total_eligible * 100:.2f}%)")
    print("-" * 80)
    print("4. Descriptive Statistics for Outcome Clicks (Days 15-42):")
    print(f"   - Count:                                          {click_count:,}")
    print(f"   - Mean:                                           {click_mean:.2f}")
    print(f"   - Median:                                         {click_median:.2f}")
    print(f"   - Min:                                            {click_min:,}")
    print(f"   - Max:                                            {click_max:,}")
    print("-" * 80)
    print("5. Unregistration Distribution among Eligible Students:")
    print(f"   - Withdrawing during Days 15-42:                  {unreg_15_42:>4} students ({unreg_15_42 / total_eligible * 100:.2f}%)")
    print(f"   - Withdrawing after Day 42:                       {unreg_after_42:>4} students ({unreg_after_42 / total_eligible * 100:.2f}%)")
    print(f"   - No Recorded Unregistration (Completed/Active):  {unreg_none:>4} students ({unreg_none / total_eligible * 100:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 1. Resolve project root
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data" / "raw"

    # 2. Load raw OULAD tables using src.data_loader
    raw_tables = load_oulad_tables(data_directory)

    # 3. Scope to AAA-2013J (Phase 2A)
    target_module = "AAA"
    target_presentation = "2013J"
    scoped = scope_to_module(
        raw_tables,
        code_module=target_module,
        code_presentation=target_presentation,
    )

    # 4. Print scoping validation report
    print_scoping_validation(
        scoped,
        code_module=target_module,
        code_presentation=target_presentation,
        raw_assessments_df=raw_tables.get("assessments"),
    )

    # 5. Compute disengagement target (Phase 2B Finalized Target)
    target_df = compute_target(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
        student_vle_df=scoped["student_vle"],
        assessments_df=raw_tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
        code_module=target_module,
        code_presentation=target_presentation,
    )

    # 6. Run outcome activity diagnostic (Phase 2B Diagnostic)
    diagnose_outcome_activity(
        student_vle_df=scoped["student_vle"],
        registration_df=scoped["registration"],
        student_info_df=scoped["student_info"],
    )

