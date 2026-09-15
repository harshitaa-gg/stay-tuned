"""Phase 9: Final Test Evaluation Module for Stay-Tuned.

This module evaluates the frozen Random Forest model artifact (models/best_model.joblib)
at the selected operational decision threshold (0.35) on the holdout test set (X_test, y_test).

Strict Anti-Leakage & Governance Rules:
---------------------------------------
1. Single Test Evaluation: The test set is evaluated exactly once as a final benchmark.
2. Frozen Configuration: No model retraining, hyperparameter tuning, or threshold re-calibration
   is performed or permitted on the test set.
3. Decoupled Evaluation: The model is loaded directly from disk as an artifact rather than
   retrained in-memory, ensuring that the evaluated model is identical to the Phase 8.2 frozen artifact.
4. Operational Thresholding: Binary predictions use the frozen 0.35 threshold (probability >= 0.35 -> 1)
   rather than Scikit-Learn's default 0.50 threshold.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.data_loader import load_oulad_tables
from src.problem_definition import scope_to_module, compute_target
from src.feature_engineering import engineer_features
from src.model import prepare_data, split_data


def load_final_model(model_path=None):
    """Load the frozen model artifact and validate the operational decision threshold.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to the saved model artifact. Defaults to models/best_model.joblib.

    Returns
    -------
    tuple
        (pipeline, threshold, metadata)
        - pipeline : sklearn.pipeline.Pipeline, fitted model pipeline
        - threshold : float, operational decision threshold
        - metadata : dict, full artifact metadata dictionary
    """
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen model artifact not found at: {model_path}. "
            "Ensure Phase 8.2 was completed successfully before evaluating Phase 9."
        )

    artifact = joblib.load(model_path)

    if "pipeline" not in artifact or "threshold" not in artifact:
        raise KeyError(f"Corrupt artifact at {model_path}: missing 'pipeline' or 'threshold' keys.")

    pipeline = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    # Strict governance validation: threshold must be exactly 0.35
    expected_threshold = 0.35
    if not np.isclose(threshold, expected_threshold, atol=1e-4):
        raise ValueError(
            f"Model threshold mismatch! Expected {expected_threshold:.2f}, got {threshold:.2f}. "
            "Phase 9 requires the frozen 0.35 operating threshold selected in Phase 8.2."
        )

    print(f"Loaded frozen model artifact from: {model_path}")
    print(f"  - Model Architecture:      {artifact.get('model_name', 'Unknown')}")
    print(f"  - Operational Threshold:   {threshold:.2f}")
    print(f"  - Features:                {len(artifact.get('feature_names', []))} features")
    print(f"  - Training Set CV PR-AUC:  {artifact.get('cv_pr_auc', 0.0):.4f}")
    return pipeline, threshold, artifact


def recreate_test_set(
    data_dir=None,
    code_module="AAA",
    code_presentation="2013J",
    obs_end_day=14,
    test_size=0.2,
    random_state=42,
):
    """Recreate the untouched holdout test set using the standardized project data pipeline.

    Parameters
    ----------
    data_dir : Path or str, optional
        Path to raw OULAD CSV files. Defaults to data/raw.
    code_module : str, default='AAA'
        Module code.
    code_presentation : str, default='2013J'
        Presentation code.
    obs_end_day : int, default=14
        Observation window upper bound (Days 0-14).
    test_size : float, default=0.2
        Stratified test fraction.
    random_state : int, default=42
        Reproducibility seed matching Phase 5.

    Returns
    -------
    tuple
        (X_test, y_test)
        - X_test : pandas.DataFrame, shape (75, 8)
        - y_test : pandas.Series, shape (75,)
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    # 1. Load raw tables
    raw_tables = load_oulad_tables(data_dir)

    # 2. Scope to AAA-2013J
    scoped = scope_to_module(raw_tables, code_module=code_module, code_presentation=code_presentation)

    # 3. Compute target for eligible students
    targets_df = compute_target(
        student_info_df=scoped["student_info"],
        registration_df=scoped["registration"],
        student_vle_df=scoped["student_vle"],
        assessments_df=raw_tables["assessments"],
        student_assessment_df=scoped["student_assessment"],
        code_module=code_module,
        code_presentation=code_presentation,
    )

    # Filter scoped student_info to eligible students
    eligible_student_info = scoped["student_info"][
        scoped["student_info"]["id_student"].isin(targets_df["id_student"])
    ].copy()

    # 4. Extract temporal features (Days 0-14 only)
    features_df = engineer_features(
        student_vle_df=scoped["student_vle"],
        student_info_df=eligible_student_info,
        obs_end_day=obs_end_day,
    )

    # 5. Prepare feature matrix X and target y
    X, y, _, _ = prepare_data(features_df, targets_df)

    # 6. Execute identical stratified split
    _, X_test, _, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    # Validation assertions
    assert len(X_test) == 75, f"Expected 75 test samples, got {len(X_test)}"
    assert len(y_test) == 75, f"Expected 75 test labels, got {len(y_test)}"
    assert (y_test == 0).sum() == 64, f"Expected 64 class 0 in test, got {(y_test == 0).sum()}"
    assert (y_test == 1).sum() == 11, f"Expected 11 class 1 in test, got {(y_test == 1).sum()}"

    print(f"Recreated test set successfully: {len(X_test)} learners (64 Active, 11 Disengaged).")
    return X_test, y_test


def evaluate_final_model(pipeline, threshold, X_test, y_test):
    """Generate probability predictions, apply frozen threshold, and compute all test metrics.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted Scikit-Learn pipeline.
    threshold : float
        Decision threshold (must be 0.35).
    X_test : pandas.DataFrame
        Holdout test feature matrix (N=75).
    y_test : pandas.Series
        Holdout test target vector.

    Returns
    -------
    dict
        Comprehensive evaluation metrics dictionary.
    """
    # 1. Generate predicted probabilities for the positive class (disengaged)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 2. Apply frozen operational threshold
    # probability >= 0.35 -> 1 (disengaged / at-risk)
    # probability < 0.35  -> 0 (retained / active)
    y_pred = (y_prob >= threshold).astype(int)

    # 3. Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 4. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Not Disengaged (0)", "Disengaged (1)"],
        digits=4,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Not Disengaged (0)", "Disengaged (1)"],
        output_dict=True,
    )

    results = {
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cm": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "flagged_count": int(tp + fp),
        "flagged_pct": float((tp + fp) / len(y_test) * 100),
        "report_text": report_text,
        "report_dict": report_dict,
    }

    return results


def plot_confusion_matrix(cm, output_path=None):
    """Plot and save publication-quality confusion matrix heatmap.

    Parameters
    ----------
    cm : numpy.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]].
    output_path : Path or str, optional
        Target save path. Defaults to models/confusion_matrix.png.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tn, fp, fn, tp = cm.ravel()

    annot_matrix = np.array([
        [f"TN\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"FP\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
        [f"FN\n{fn}\n({fn/(fn+tp)*100:.1f}%)", f"TP\n{tp}\n({tp/(fn+tp)*100:.1f}%)"],
    ])

    plt.figure(figsize=(6.5, 5.5))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"fontsize": 12, "weight": "bold"},
    )

    ax.set_title(
        "Confusion Matrix — Test Set\nFrozen Random Forest (Threshold = 0.35)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=11, weight="bold", labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=11, weight="bold", labelpad=10)
    ax.set_xticklabels(["0: Not Disengaged", "1: Disengaged"], fontsize=10)
    ax.set_yticklabels(["0: Not Disengaged", "1: Disengaged"], fontsize=10, va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved successfully to: {output_path}")


def plot_roc_curve(y_test, y_prob, roc_auc, output_path=None):
    """Plot and save Receiver Operating Characteristic (ROC) curve using probability scores.

    Parameters
    ----------
    y_test : array-like
        Ground truth labels.
    y_prob : array-like
        Predicted probabilities for class 1.
    roc_auc : float
        ROC-AUC score.
    output_path : Path or str, optional
        Target save path. Defaults to models/roc_curve.png.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "models" / "roc_curve.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"Random Forest (ROC-AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=1.5,
        linestyle="--",
        label="Random Chance (ROC-AUC = 0.5000)",
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, weight="bold")
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve — Holdout Test Set",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved successfully to: {output_path}")


def plot_pr_curve(y_test, y_prob, pr_auc, output_path=None):
    """Plot and save Precision-Recall curve using probability scores.

    Parameters
    ----------
    y_test : array-like
        Ground truth labels.
    y_prob : array-like
        Predicted probabilities for class 1.
    pr_auc : float
        Average Precision / PR-AUC score.
    output_path : Path or str, optional
        Target save path. Defaults to models/pr_curve.png.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "models" / "pr_curve.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    baseline = (y_test == 1).mean()

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall_curve,
        precision_curve,
        color="#2ca02c",
        lw=2.5,
        label=f"Random Forest (PR-AUC = {pr_auc:.4f})",
    )
    plt.axhline(
        y=baseline,
        color="gray",
        lw=1.5,
        linestyle="--",
        label=f"No-Skill Prevalence Baseline ({baseline:.4f})",
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, weight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, weight="bold")
    plt.title(
        "Precision-Recall (PR) Curve — Holdout Test Set",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PR curve plot saved successfully to: {output_path}")


def print_evaluation_summary(results, metadata=None):
    """Print the official Phase 9 final test evaluation report and interpretation."""
    t = results["threshold"]
    acc = results["accuracy"]
    prec = results["precision"]
    rec = results["recall"]
    f1 = results["f1"]
    roc_auc = results["roc_auc"]
    pr_auc = results["pr_auc"]
    tn, fp, fn, tp = results["tn"], results["fp"], results["fn"], results["tp"]
    total = tn + fp + fn + tp
    flagged = results["flagged_count"]
    flagged_pct = results["flagged_pct"]

    print("\n" + "=" * 80)
    print("PHASE 9: FINAL TEST EVALUATION REPORT")
    print("=" * 80)
    print("Final test evaluation using frozen Random Forest threshold = 0.35")
    print(f"Holdout Test Cohort Size:        {total} students")
    print(f"Ground Truth Class Balance:      Class 0 (Active): {tn+fp} ({(tn+fp)/total*100:.2f}%), Class 1 (Disengaged): {fn+tp} ({(fn+tp)/total*100:.2f}%)")
    print(f"Evaluated Operational Threshold: {t:.2f} (Frozen from Phase 8.2)")
    print("-" * 80)
    print("FINAL TEST CLASSIFICATION METRICS:")
    print(f"  1. Accuracy:                   {acc:.4f} ({acc*100:.2f}%)")
    print(f"  2. Precision (Class 1):        {prec:.4f} ({prec*100:.2f}%)")
    print(f"  3. Recall (Class 1):           {rec:.4f} ({rec*100:.2f}%)")
    print(f"  4. F1-Score (Class 1):         {f1:.4f}")
    print(f"  5. ROC-AUC:                    {roc_auc:.4f}")
    print(f"  6. PR-AUC (Average Precision): {pr_auc:.4f}")
    print("-" * 80)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  - True Negatives (TN):         {tn:>2} students (Correctly identified as active/retained)")
    print(f"  - False Positives (FP):        {fp:>2} students (Active students flagged for proactive support)")
    print(f"  - False Negatives (FN):        {fn:>2} students (At-risk students missed by model)")
    print(f"  - True Positives (TP):         {tp:>2} students (At-risk students correctly detected)")
    print(f"  - Total Learners Flagged:      {flagged:>2} / {total} ({flagged_pct:.2f}% of test cohort)")
    print("-" * 80)
    print("DETAILED CLASSIFICATION REPORT:")
    print(results["report_text"])
    print("-" * 80)
    print("INTERPRETATION & DISCUSSION:")
    print("""
1. Recall (Sensitivity):
   The model achieved a test recall of {:.2f}% (identifying {} out of {} truly disengaged learners).
   In an early warning system, recall is paramount: missing an at-risk learner (False Negative)
   means that the learner may not receive timely support before the Day 19 assessment deadline.
   Capturing {}/{} of learners heading toward disengagement by Day 14 confirms
   the model's meaningful operational sensitivity.

2. Precision (Positive Predictive Value):
   The model achieved a test precision of {:.2f}% ({} out of {} flagged students were truly disengaged).
   Compared to the baseline positive prevalence of {:.2f}%, this represents a {:.2f}x lift in precision.
   In educational practice, a False Positive is not a harmful error—it simply triggers a low-cost,
   supportive nudge (e.g., an automated check-in email or advising message). A precision of ~{:.1f}%
   provides an actionable signal while avoiding advisor alert fatigue.

3. F1-Score:
   The F1-score of {:.4f} reflects a moderate balance between precision and recall
   under a 5.8:1 class imbalance, confirming that the threshold calibration at 0.35 maintained
   viable alert quality while dramatically boosting learner recall.

4. ROC-AUC:
   The test ROC-AUC of {:.4f} indicates useful global discriminative capacity across all possible
   decision thresholds. The model ranks randomly selected disengaged students higher than active
   students with ~{:.1f}% probability.

5. PR-AUC (Average Precision):
   The PR-AUC of {:.4f} demonstrates strong ranking quality in the positive (minority) class,
   significantly outperforming the no-skill baseline of {:.4f}. PR-AUC evaluates precision across
   all recall levels without being inflated by the large True Negative class, making it the most
   rigorous indicator of early-warning ranking effectiveness.

6. Confusion Matrix Operational Meaning:
   - TP ({}) = Disengaged students successfully flagged early for advising intervention.
   - TN ({}) = Engaged students correctly left undisturbed.
   - FP ({}) = Active students who received a supportive message (low-cost false alarm).
   - FN ({}) = Disengaged students missed by Day 14 behavioral indicators.

Note on AI Ethics & Intervention Scope:
The model predicts early disengagement risk so that potentially at-risk learners can be
identified for supportive outreach; it does not claim to prevent dropout or guarantee
academic success.
""".format(
        rec * 100, tp, tp + fn, tp, tp + fn,
        prec * 100, tp, tp + fp, (tp + fn) / total * 100, prec / ((tp + fn) / total) if (tp + fn) > 0 else 1.0, prec * 100,
        f1,
        roc_auc, roc_auc * 100,
        pr_auc, (tp + fn) / total,
        tp, tn, fp, fn,
    ))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 1. Load the frozen model artifact from disk
    pipeline, threshold, metadata = load_final_model()

    # 2. Recreate the untouched holdout test set (75 students)
    X_test, y_test = recreate_test_set()

    # 3. Evaluate the frozen model once on the holdout test set
    test_results = evaluate_final_model(
        pipeline=pipeline,
        threshold=threshold,
        X_test=X_test,
        y_test=y_test,
    )

    # 4. Generate the 3 required publication-quality plots
    plot_confusion_matrix(test_results["cm"])
    plot_roc_curve(y_test, test_results["y_prob"], test_results["roc_auc"])
    plot_pr_curve(y_test, test_results["y_prob"], test_results["pr_auc"])

    # 5. Print comprehensive evaluation report and interpretation
    print_evaluation_summary(test_results, metadata)

