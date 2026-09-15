"""Run Data & Prediction Pipeline for Stay-Tuned Dashboard.

This script executes the end-to-end data processing and inference pipeline for Module AAA,
Presentation 2013J, producing the necessary artifact files for the Streamlit dashboard:
  - models/predictions.csv: Scored learners with probability, risk level, predicted label,
    and rule-based recommendation.
  - models/features.csv: Day 0–14 canonical feature matrix for all analyzed learners.
  - models/shap_train_ids.npy: Identifiers of training learners with precomputed SHAP values.

Strict Architecture & Anti-Leakage Compliance:
----------------------------------------------
1. Frozen Model: Uses models/best_model.joblib without retraining or tuning.
2. Frozen Threshold: Applies the frozen 0.35 operational threshold from Phase 8.2.
3. Feature Integrity: Does not create new features specifically for the dashboard.
4. Module Reuse: Reuses src.data_loader, src.problem_definition, src.feature_engineering,
   src.predict, and src.recommend.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_oulad_tables
from src.problem_definition import scope_to_module, compute_target
from src.feature_engineering import engineer_features
from src.model import prepare_data, split_data
from src.predict import load_model, generate_predictions
from src.recommend import generate_recommendations


def run_pipeline(data_dir=None, output_dir=None):
    """Execute the pipeline to generate dashboard artifacts.

    Parameters
    ----------
    data_dir : Path or str, optional
        Path to raw OULAD CSV files. Defaults to data/raw.
    output_dir : Path or str, optional
        Destination directory for dashboard artifacts. Defaults to models/.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (predictions_df, features_df)
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAY-TUNED: DASHBOARD ARTIFACT GENERATION PIPELINE")
    print("=" * 80)
    print(f"Data Directory:    {data_dir}")
    print(f"Output Directory:  {output_dir}")
    print("Module / Cohort:   AAA 2013J")
    print("Prediction Point:  Day 14")
    print("-" * 80)

    # 1. Load raw OULAD tables
    print("\n[Step 1/6] Loading raw OULAD tables...")
    raw_tables = load_oulad_tables(data_dir)

    # 2. Scope to AAA 2013J
    print("\n[Step 2/6] Scoping data to Module AAA, Presentation 2013J...")
    scoped = scope_to_module(raw_tables, code_module="AAA", code_presentation="2013J")

    # 3. Generate Day 0–14 target and identify eligible population
    print("\n[Step 3/6] Generating target labels and filtering eligible learners...")
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
    print(f"-> Total eligible learners for Day 14 prediction: {len(eligible_student_info):,}")

    # 4. Derive Day 0–14 behavioral features
    print("\n[Step 4/6] Deriving Day 0–14 behavioral features...")
    features_df = engineer_features(
        student_vle_df=scoped["student_vle"],
        student_info_df=eligible_student_info,
        obs_end_day=14,
    )
    print(f"-> Engineered feature matrix shape: {features_df.shape}")

    # 5. Load frozen model artifact and score entire cohort
    print("\n[Step 5/6] Loading frozen model and generating predictions & risk tiers...")
    model_path = output_dir / "best_model.joblib"
    pipeline, threshold, artifact = load_model(model_path)
    print(f"-> Model architecture: {artifact.get('model_name', 'Pipeline')}")
    print(f"-> Frozen operational threshold: {threshold:.2f}")

    student_ids = list(features_df.index)
    preds_df = generate_predictions(
        best_pipeline=pipeline,
        X=features_df,
        student_ids=student_ids,
    )

    # 6. Generate rule-based intervention recommendations
    print("\n[Step 6/6] Generating rule-based intervention recommendations...")
    recs_df = generate_recommendations(
        predictions_df=preds_df,
        features_df=features_df,
    )

    # Merge recommendations into predictions
    # recs_df has ['id_student', 'risk_level', 'disengagement_probability', 'recommendation_type', 'recommendation']
    # preds_df has ['id_student', 'disengagement_probability', 'risk_level', 'predicted_label']
    dashboard_preds = preds_df.merge(
        recs_df[["id_student", "recommendation", "recommendation_type"]],
        on="id_student",
        how="left",
    )

    # Reorder columns cleanly as required
    pred_columns = [
        "id_student",
        "disengagement_probability",
        "risk_level",
        "predicted_label",
        "recommendation",
        "recommendation_type",
    ]
    dashboard_preds = dashboard_preds[pred_columns].copy()

    # Prepare features table with id_student as column
    dashboard_features = features_df.reset_index().copy()

    # Save artifacts to models/
    preds_path = output_dir / "predictions.csv"
    feat_path = output_dir / "features.csv"

    dashboard_preds.to_csv(preds_path, index=False)
    dashboard_features.to_csv(feat_path, index=False)
    print(f"\nSaved dashboard predictions to: {preds_path}")
    print(f"Saved dashboard features to:    {feat_path}")

    # Also record training split student IDs for SHAP lookup in the dashboard
    X, y, _, _ = prepare_data(features_df, targets_df)
    X_train, _, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    shap_ids_path = output_dir / "shap_train_ids.npy"
    np.save(shap_ids_path, np.array(X_train.index))
    print(f"Saved SHAP training student IDs to: {shap_ids_path}")

    # Summary Report
    total = len(dashboard_preds)
    counts = dashboard_preds["risk_level"].value_counts()
    low_cnt = int(counts.get("LOW", 0))
    med_cnt = int(counts.get("MEDIUM", 0))
    high_cnt = int(counts.get("HIGH", 0))
    at_risk_rate = (med_cnt + high_cnt) / total * 100

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Learners Processed:        {total:,}")
    print(f"  - LOW Risk:                    {low_cnt:>4} ({low_cnt / total * 100:5.2f}%)")
    print(f"  - MEDIUM Risk:                 {med_cnt:>4} ({med_cnt / total * 100:5.2f}%)")
    print(f"  - HIGH Risk:                   {high_cnt:>4} ({high_cnt / total * 100:5.2f}%)")
    print(f"Combined At-Risk Rate:           {med_cnt + high_cnt:>4} ({at_risk_rate:5.2f}%)")
    print(f"Mean Disengagement Probability:  {dashboard_preds['disengagement_probability'].mean():.4f}")
    print("=" * 80 + "\n")

    return dashboard_preds, dashboard_features


if __name__ == "__main__":
    run_pipeline()
