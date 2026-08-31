# Stay-Tuned — Product Requirements Document (PRD)

**Product name:** Stay-Tuned
**Formal title:** Stay-Tuned: Early Learner Disengagement Prediction System
**Repository:** `stay-tuned`
**Document status:** Stable project-level context (not a step-by-step implementation guide)

---

## 1. Product Overview

**Product name:** Stay-Tuned

**Formal title:** Stay-Tuned: Early Learner Disengagement Prediction System

**Description:** Stay-Tuned is a machine learning system that predicts, at Day 14 of an online course, which learners are likely to disengage during the remainder of the course. It uses only behavioral and assessment signals available during the first 14 days of a learner's activity, drawing on the OULAD (Open University Learning Analytics Dataset), specifically the AAA module, 2013J presentation. Predictions are paired with SHAP-based explainability, a LOW/MEDIUM/HIGH risk classification, rule-based intervention recommendations, and a four-page Streamlit dashboard for exploring results.

**Product goal:** Enable early, explainable identification of learners at risk of disengagement — early enough that intervention is still possible — using only information that would realistically be available partway through a course.

**Intended users:** Developers/coding agents implementing the system; students/researchers working on the project; reviewers evaluating the project's requirements; and future contributors extending the work.

**High-level value proposition:** Retrospective disengagement detection (identifying at-risk learners only after they've already disengaged, e.g., via final course outcomes) is too late to act on. Stay-Tuned reframes the problem as an early-warning prediction task, answering a single, well-defined question — who is at risk by Day 14 — with a full pipeline from raw data to an interpretable, actionable dashboard.

---

## 2. Problem Statement

Learner disengagement in online/blended courses is frequently identified only in hindsight — after a course has ended, through final grades or withdrawal status. Retrospective detection has no practical value for the learner in question because there is no time window left to intervene.

**The gap:** Existing detection approaches label disengagement after the fact, using signals (like final outcomes) that are unavailable, by definition, at any point when intervention is still possible.

**The specific prediction question Stay-Tuned answers:**

> At Day 14 of a course, using only behavioral signals available during the first 14 days, which learners are likely to disengage during the subsequent outcome window (Days 15–42)?

This reframes disengagement detection as a forward-looking, decision-relevant prediction task rather than a retrospective labeling exercise.

---

## 3. Product Objective

**Central objective:**

> Predict, at Day 14 of a course, which learners are likely to disengage during the subsequent outcome window using behavioral signals available during the first 14 days.

All downstream requirements (feature engineering, modeling, evaluation, explainability, dashboard) exist in service of this single objective, and must respect the temporal boundary it implies (see Section 10, Temporal Leakage Requirements).

---

## 4. Scope

### 4.1 Initial / Core Scope

- Dataset: OULAD (Open University Learning Analytics Dataset)
- Single module/presentation: **AAA, 2013J**
- Observation window: **Day 0–14**
- Prediction point: **Day 14**
- Outcome window: **Day 15–42**
- Behavioral and assessment signals derived from the observation window
- Classification models: Logistic Regression, Random Forest, XGBoost
- SHAP explainability (global and per-learner)
- LOW / MEDIUM / HIGH risk classification
- Rule-based (heuristic) intervention recommendations
- Streamlit dashboard (four pages — see Section 6)

### 4.2 Stretch Scope

The provided source material does not enumerate specific stretch-scope items beyond the core scope above. **This is flagged in Section 18 (Open Questions).** No stretch items should be assumed or invented; any stretch scope must be explicitly defined before implementation.

### 4.3 Future Work

The provided source material does not enumerate specific future-work directions beyond what is implied by the core scope (e.g., extension beyond a single module/presentation). **This is flagged in Section 18 (Open Questions).** No future-work items should be assumed or invented.

---

## 5. Target Users / Stakeholders

**Actual users of the prototype (current scope):**
- The developer/coding agent implementing the system
- The student/researcher building and evaluating the project
- Reviewers assessing the project against its requirements

**Potential future users (not in current scope, not committed to):**
- EdTech platforms
- Universities and academic institutions
- Corporate learning & development (L&D) teams
- Instructors/educators
- Individuals or teams responsible for learner engagement outcomes

The prototype is a research/demonstration system built on a single historical dataset and module/presentation. It is not built, validated, or positioned for production deployment by any of the potential future users listed above.

---

## 6. Functional Requirements

### 6.1 Data Ingestion
- Load the relevant OULAD tables
- Validate loaded data (structure/integrity checks appropriate to the pipeline)
- Filter data to the single module/presentation in scope: AAA, 2013J

### 6.2 Target Generation
Must precisely implement and document:
- **Observation window:** Day 0–14
- **Prediction point:** Day 14
- **Outcome window:** Day 15–42
- **Disengagement label definition:** derived from learner behavior/outcomes within the outcome window (exact operational definition to be finalized during implementation — see Open Questions if ambiguity remains)
- **Withdrawal handling:** learners who withdrew **before** the prediction point (Day 14) must be excluded from the modeling population, since no meaningful Day 0–14 signal window exists for them relative to the prediction task

### 6.3 Feature Engineering
Feature categories to be derived from the observation window:
- Activity volume
- Temporal trends (within Day 0–14)
- Inactivity (gaps/lulls in engagement)
- Engagement consistency
- Assessment behavior (submissions, scores, timeliness within the window)
- Course context (module/presentation-level context as applicable)

**Hard constraint:** All features must be constructible using only information available by Day 14. No feature may use information that would only be known after Day 14.

### 6.4 Machine Learning
- Train/test split strategy respecting the temporal and leakage constraints (see Section 10)
- Preprocessing pipeline fitted only on training data
- Baseline model
- Logistic Regression
- Random Forest
- XGBoost
- Cross-validation for model comparison
- Hyperparameter tuning (without use of the test set)
- Final evaluation on the held-out test set only after model/hyperparameter selection is complete

### 6.5 Prediction and Risk Classification
- Output: disengagement probability per learner
- Output: predicted label (disengaged / not disengaged)
- Output: risk tier — **LOW / MEDIUM / HIGH**
- Thresholds for risk tiers: **not specified in the provided source material** — to be defined during implementation (see Open Questions)

### 6.6 Explainability
- Global SHAP feature importance across the model
- Per-learner (local) SHAP explanations
- Human-readable feature names (not raw column names) in explanations
- Explicit statement of explanation limitations: SHAP values describe model behavior/attribution, not causal relationships

### 6.7 Recommendations
- Rule-based (heuristic) intervention recommendation system tied to risk tier and/or key contributing signals
- Explicit, prominent statement that recommendations are heuristics, not validated interventions, and carry no guarantee of effectiveness

### 6.8 Dashboard
Four required Streamlit dashboard pages:
1. **Overview**
2. **Behavioral Analysis**
3. **At-Risk Learners**
4. **Individual Learner Deep-Dive**

### 6.9 Packaging / Deployment
- Saved model artifacts
- Saved prediction/feature outputs
- `requirements` file for dependency reproduction
- Streamlit deployment configuration
- Local execution support
- GitHub repository documentation

---

## 7. Non-Functional Requirements

- **Reproducibility:** pipeline steps and results should be reproducible given the same data and code
- **Leakage prevention:** enforced throughout data, feature, and modeling stages (see Section 10)
- **Interpretability:** predictions must be explainable, not just accurate
- **Maintainability:** code organized into clear modules/files
- **Documentation:** clear README and setup documentation
- **Reasonable execution time:** appropriate for the scope of a single-module, single-presentation prototype
- **Clean dashboard UX:** the four dashboard pages should be usable and clearly organized

No numerical performance or infrastructure requirements (e.g., specific latency, throughput, or uptime targets) are specified in the source material, and none are invented here.

---

## 8. Data Requirements

**Dataset:** OULAD (Open University Learning Analytics Dataset)

**Scope filter:** module/presentation = AAA / 2013J only

**Key identifiers and relationships:**
- `code_module` — identifies the course module (fixed to AAA in scope)
- `code_presentation` — identifies the specific run/offering of the module (fixed to 2013J in scope)
- `id_student` — unique learner identifier, used to join learner-level data across OULAD tables
- Relative day/date fields — OULAD represents time as **day offsets relative to the start of the module/presentation**, not calendar dates. These relative-day fields are what define the Day 0–14 observation window and the Day 15–42 outcome window used throughout the project.

The specific OULAD tables to be ingested (e.g., student registration, VLE interaction logs, assessment submissions) and exactly what each contributes should be finalized and documented during data-ingestion implementation, consistent with the feature categories in Section 6.3.

---

## 9. ML / Evaluation Requirements

- **Class imbalance:** must be explicitly considered, since disengagement is expected to be a minority outcome; this affects both modeling choices and metric selection.
- **Accuracy is not the primary metric** — with class imbalance, accuracy can be misleading (a model predicting the majority class can score highly while being useless).
- **Primary/required metrics:**
  - ROC-AUC
  - PR-AUC / Average Precision
  - Precision
  - Recall
  - F1
  - Confusion matrix
  - Threshold analysis
- **Metric priority:** For an early-warning use case, recall is prioritized more heavily than in a typical classification task, since missing an at-risk learner (false negative) is generally more costly than a false alarm. This does not mean recall is the *only* metric that matters — precision, PR-AUC, and threshold analysis remain necessary to keep the system usable and to avoid excessive false positives.

---

## 10. Temporal Leakage Requirements

**This is a critical, non-negotiable project constraint.**

- Features must use **only** data from Days 0–14.
- Target/outcome information must come **only** from Days 15–42.
- `final_result` (or any equivalent end-of-course outcome field) must **never** be used as a feature.
- Any information not actually available at Day 14 must not influence the prediction in any way.
- Preprocessing (scaling, encoding, imputation, etc.) must be **fit only on training data**, never on the full dataset or the test set.
- Test data must remain untouched until final evaluation — no peeking, no iteration against it.
- Model selection and hyperparameter tuning must use only training/validation data, never the test set.

**Why this matters:** If information from after Day 14 (including the outcome itself) leaks into the features or into model selection, the resulting evaluation numbers will look good but will not reflect real-world predictive performance — because in a real deployment, that future information would not yet exist at the moment the prediction is needed. Leakage would invalidate the project's central claim (that disengagement can be predicted *early*, from *only* early signals).

---

## 11. System Architecture

High-level pipeline (conceptual, not phase-by-phase implementation detail):

```
Data ingestion (OULAD tables, AAA/2013J filter)
        ↓
Problem definition / target generation
 (observation window, prediction point, outcome window, label, withdrawal handling)
        ↓
Temporal feature engineering (Day 0–14 only)
        ↓
Train/test split
        ↓
Preprocessing (fit on train only)
        ↓
Model comparison (Logistic Regression, Random Forest, XGBoost + baseline)
        ↓
Hyperparameter tuning (train/validation only)
        ↓
Final evaluation (held-out test set)
        ↓
Prediction (probability, label)
        ↓
SHAP explainability (global + per-learner)
        ↓
Risk classification (LOW / MEDIUM / HIGH)
        ↓
Rule-based recommendations
        ↓
Streamlit dashboard (Overview, Behavioral Analysis, At-Risk Learners, Individual Deep-Dive)
```

Major project modules/files should map roughly to these pipeline stages (e.g., a data-ingestion module, a target-generation module, a feature-engineering module, a modeling/evaluation module, an explainability module, a recommendations module, and a dashboard app), consistent with the phase-by-phase implementation guide used alongside this PRD.

---

## 12. Research Questions

- **RQ1:** How accurately can disengagement be predicted from early behavioral signals?
- **RQ2:** Which early signals are most predictive?
- **RQ3:** Can SHAP provide interpretable per-learner explanations?

---

## 13. Success Criteria

Stay-Tuned is considered successful when:

- The temporal formulation (observation window, prediction point, outcome window) is correctly implemented
- Features are demonstrably leakage-free
- A functioning comparison across the specified ML models is completed
- Final evaluation is performed honestly, on an untouched test set, after model/hyperparameter selection
- Meaningful behavioral analysis is produced (supporting RQ2)
- SHAP output is interpretable at both global and per-learner levels (supporting RQ3)
- Risk classification (LOW/MEDIUM/HIGH) functions end-to-end
- Rule-based recommendations function end-to-end and are clearly labeled as heuristic
- All four dashboard pages are functional
- The setup is reproducible from a clean environment
- Documentation (README, setup instructions) is complete

No specific target performance numbers (e.g., a minimum ROC-AUC) are defined as success criteria, consistent with the source material.

---

## 14. Constraints and Limitations

- **Single-module scope:** results are based on one module/presentation (AAA, 2013J) only and may not generalize to other modules, presentations, or institutions
- **Age of OULAD:** the dataset reflects a past cohort/period and online-learning behavior may have changed since
- **Heuristic thresholds:** risk-tier thresholds are rule-based/heuristic, not derived from a validated cost-benefit analysis
- **Unvalidated recommendations:** intervention recommendations are heuristic and have not been tested for real-world effectiveness
- **Demographic features intentionally excluded:** the system does not use learner demographic attributes as predictive features
- **Generalizability:** findings and model performance are specific to this scope and should not be assumed to transfer to other contexts

---

## 15. Security / Privacy / Responsible Use

- Predictions are for **research/demonstration purposes only**, not for operational/high-stakes decision-making
- Risk predictions must **not** be treated as definitive judgments about any individual learner
- SHAP explanations describe **model behavior**, not causal relationships in the real world
- Recommendations are **heuristic, not validated interventions**, and must be presented as such
- The system does not make, and should not be framed as making, high-stakes decisions about learners

---

## 16. Deliverables

- Source code
- Data pipeline
- EDA notebook
- Trained model(s)
- Evaluation artifacts
- SHAP artifacts
- Prediction outputs
- Recommendation engine
- Streamlit dashboard
- Requirements/setup files
- README
- Deployment configuration
- GitHub repository (`stay-tuned`)

---

## 17. Development Roadmap (High Level)

The implementation guide breaks the project into phase-by-phase prompts (provided separately to the coding agent). At a high level, the roadmap covers:

1. Environment / setup
2. Data ingestion
3. Problem definition (target generation)
4. Exploratory data analysis (EDA)
5. Feature engineering
6. ML pipeline (model comparison, tuning)
7. Evaluation
8. Prediction / explainability
9. Recommendations
10. Dashboard
11. Deployment / documentation

This PRD intentionally does not reproduce the detailed, phase-by-phase implementation prompts — those remain in the separate implementation guide provided to the coding agent one phase at a time.

---

## 18. Open Questions / Decisions Required

The following items are referenced by the project's requirements but are **not fully specified** in the source material provided for this PRD, and should be explicitly decided/verified during implementation rather than assumed:

- **Stretch scope items:** not enumerated in the source material — must be defined before any stretch work begins
- **Future-work items:** not enumerated in the source material — must be defined separately if desired
- **Exact disengagement label definition:** the precise rule for labeling a learner as "disengaged" within the Day 15–42 outcome window needs to be finalized and documented
- **Risk-tier thresholds:** the specific probability cutoffs for LOW / MEDIUM / HIGH are not given and must be set (and documented) during implementation
- **Actual class balance:** the real disengaged/not-disengaged split after filtering to AAA/2013J and applying withdrawal exclusions is not yet known
- **Best-performing model:** which of the baseline, Logistic Regression, Random Forest, or XGBoost ultimately performs best is not predetermined
- **Actual evaluation metric values:** no performance numbers should be assumed until final evaluation is run
- **Strongest behavioral signals:** which specific features are most predictive (RQ2) is an empirical outcome, not a given
- **Final threshold decisions:** if classification thresholds are later optimized (e.g., for recall), this should be documented as a decision, not assumed upfront
- **Deployment artifact size constraints:** any limits on model/artifact size for Streamlit deployment are not specified and should be checked against the actual deployment target
- **Specific OULAD tables and their exact contributions:** the source material references OULAD generally; the exact set of tables ingested and what each contributes to features should be finalized and documented during data-ingestion implementation

---

## 19. Glossary

- **OULAD:** Open University Learning Analytics Dataset — the dataset used as the basis for this project.
- **VLE:** Virtual Learning Environment — the online platform through which learner interaction/activity data is generated.
- **Disengagement:** A learner ceasing meaningful participation/engagement with the course, as operationally defined within the outcome window.
- **Observation window:** The period (Day 0–14) from which input features are derived.
- **Prediction point:** The point in time (Day 14) at which the prediction is made.
- **Outcome window:** The period (Day 15–42) during which the disengagement outcome is determined.
- **Temporal leakage:** The use of information not actually available at the prediction point, which would invalidate the evaluation of real-world predictive performance.
- **Class imbalance:** A situation where one outcome class (e.g., disengaged) is substantially less frequent than the other, requiring careful metric choice and modeling treatment.
- **PR-AUC:** Precision-Recall Area Under the Curve — an evaluation metric well-suited to imbalanced classification problems.
- **SHAP:** SHapley Additive exPlanations — a method for attributing model predictions to individual feature contributions, both globally and per instance.
- **Risk level:** The LOW / MEDIUM / HIGH classification assigned to a learner based on their predicted disengagement probability.
