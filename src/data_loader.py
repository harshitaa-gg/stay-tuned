"""Data loading, table inspection, and schema description module for OULAD."""

from pathlib import Path
import pandas as pd


def load_oulad_tables(data_dir):
    """Load and inspect the core OULAD CSV tables from the specified directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory path where raw OULAD CSV files are stored.

    Returns
    -------
    dict
        Dictionary containing loaded pandas DataFrames with keys:
        - 'student_info'
        - 'registration'
        - 'student_vle'
        - 'vle'
        - 'student_assessment'
        - 'assessments'
    """
    data_path = Path(data_dir)

    file_mapping = {
        "student_info": "studentInfo.csv",
        "registration": "studentRegistration.csv",
        "student_vle": "studentVle.csv",
        "vle": "vle.csv",
        "student_assessment": "studentAssessment.csv",
        "assessments": "assessments.csv",
    }

    tables = {}

    print("=" * 80)
    print("LOADING AND INSPECTING OULAD TABLES")
    print("=" * 80)

    for key, filename in file_mapping.items():
        file_path = data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {file_path}")

        print(f"\nLoading table: {filename} (key: '{key}')...")
        df = pd.read_csv(file_path)
        tables[key] = df

        print(f"-> Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print("\nColumns, Data Types, and Missing Values:")
        
        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        print(info_df.to_string(index=False))
        print("-" * 60)

    print("\nAll 6 OULAD tables loaded and inspected successfully.")
    return tables


def describe_relationships():
    """Print a detailed description of key entity relationships and OULAD data semantics."""
    description = """
================================================================================
OULAD ENTITY RELATIONSHIPS & DATA SEMANTICS
================================================================================

1. studentInfo <-> studentRegistration:
   - Relationship Keys: ('id_student', 'code_module', 'code_presentation')
   - Role: 'studentInfo' contains demographic and final outcome data for each
     student enrolled in a specific module presentation. 'studentRegistration'
     tracks when the student registered (date_registration) and unregistered
     (date_unregistration) for that same module presentation.

2. studentVle <-> studentInfo:
   - Relationship Keys: ('id_student', 'code_module', 'code_presentation')
   - Role: 'studentVle' records log-level clickstream interactions with course
     materials (VLE resources). Each interaction is associated with a student,
     the module presentation, the specific VLE material (id_site), the relative
     day of interaction (date), and the number of clicks (sum_click).

3. studentAssessment <-> assessments:
   - Relationship Key: 'id_assessment'
   - Role: 'assessments' provides course-level metadata about each assessment
     (assessment type such as TMA, CMA, Exam; scheduled due date; and weight in
     final score). 'studentAssessment' records individual learner submission
     instances, including submission date (date_submitted), banked status, and
     achieved score (0-100).

4. Meaning of studentVle.date:
   - It represents the relative day number with respect to the official course
     start date (Day 0).
   - It is NOT a calendar date.
   - Negative values (e.g., -14, -5) represent learner interactions that took
     place BEFORE the official start date of the module presentation.

5. Meaning of final_result values in studentInfo:
   - 'Pass': Student completed the course and met the passing criteria.
   - 'Distinction': Student completed the course with exceptional performance.
   - 'Fail': Student completed the course assessments but did not achieve
     passing grades.
   - 'Withdrawn': Student formally dropped or withdrew from the course prior
     to completion.

6. Meaning of unregistration_date (date_unregistration) in studentRegistration:
   - Represents the relative day number (relative to course start, Day 0) on
     which the student unregistered/withdrew from the course.
   - Missing / NaN values indicate that the student remained registered throughout
     the course presentation and did not unregister.
   - Negative values indicate unregistration before the course officially started.
================================================================================
"""
    print(description)


def get_module_summary(student_info_df):
    """Print module-level and presentation-level student count and outcome distributions.

    Parameters
    ----------
    student_info_df : pandas.DataFrame
        The studentInfo DataFrame.
    """
    print("=" * 80)
    print("OULAD MODULE & STUDENT SUMMARY")
    print("=" * 80)

    # 1. Unique code_module values
    unique_modules = sorted(student_info_df["code_module"].unique())
    print(f"\n1. Unique Modules ({len(unique_modules)} total):")
    print("   " + ", ".join(unique_modules))

    # 2. Unique code_presentation values
    unique_presentations = sorted(student_info_df["code_presentation"].unique())
    print(f"\n2. Unique Presentations ({len(unique_presentations)} total):")
    print("   " + ", ".join(unique_presentations))

    # 3. Student count per code_module + code_presentation combination
    print("\n3. Student Count per Module + Presentation:")
    module_pres_counts = (
        student_info_df.groupby(["code_module", "code_presentation"])
        .size()
        .reset_index(name="student_count")
    )
    print(module_pres_counts.to_string(index=False))

    # 4. Overall distribution of final_result values
    print("\n4. Overall Distribution of Final Results (final_result):")
    result_counts = student_info_df["final_result"].value_counts().reset_index()
    result_counts.columns = ["final_result", "count"]
    result_counts["percentage (%)"] = (
        result_counts["count"] / len(student_info_df) * 100
    ).round(2)
    print(result_counts.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    # Define data directory relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data" / "raw"

    # 1. Load and inspect tables
    oulad_tables = load_oulad_tables(data_directory)

    # 2. Describe entity relationships
    describe_relationships()

    # 3. Print module & student summary
    get_module_summary(oulad_tables["student_info"])
