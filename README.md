# Early Learner Disengagement Prediction

An early-warning machine learning system that predicts learner disengagement using behavioral activity from the first 14 days of an online course.

The project uses the Open University Learning Analytics Dataset (OULAD) and focuses on identifying learners who may become disengaged early enough for an institution to potentially intervene.

---

## Project Scope

- **Dataset:** Open University Learning Analytics Dataset (OULAD)
- **Module:** AAA
- **Presentation:** 2013J
- **Observation window:** Day 0–14
- **Prediction point:** End of Day 14
- **Outcome window:** Day 15–42
- **Model:** Random Forest
- **Operational prediction threshold:** 0.35

The prediction target is a custom early-disengagement definition based on learner behavior after the observation window.

---

## Features

The model uses eight features derived from learner activity during Days 0–14:

- Total early clicks
- Clicks per active day
- Active days
- Longest inactivity gap
- Days since last active
- Activity delta
- Weekly click ratio
- Studied credits

The project intentionally avoids using `final_result` as a predictor because it represents retrospective course outcomes.

---

## Target Definition

A learner is labeled as disengaged if at least one of the following occurs during the outcome period:

1. Official withdrawal during Days 15–42
2. A continuous inactivity streak of at least 14 days
3. Failure to submit TMA 01 by its due date

Learners who unregistered during the observation window (Day 0–14) are excluded from the modeling population.

---

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. Load and inspect OULAD data
2. Scope the dataset to AAA 2013J
3. Define the early-disengagement target
4. Perform exploratory data analysis
5. Engineer behavioral features
6. Split the data into training and holdout test sets
7. Build leakage-safe preprocessing pipelines
8. Compare baseline and machine learning models
9. Tune Random Forest and XGBoost models
10. Select an operational probability threshold using out-of-fold predictions
11. Evaluate the frozen model on the holdout test set
12. Generate learner-level predictions
13. Generate SHAP explanations
14. Generate rule-based intervention recommendations
15. Present results through an interactive Streamlit dashboard

---

## Model

The final model is a Random Forest classifier.

The operational prediction threshold is:

```text
0.35
```

A learner is operationally flagged when the estimated disengagement probability is greater than or equal to 0.35.

The model was selected using training data and out-of-fold validation. The holdout test set was kept separate for final evaluation.

### Risk Levels

Predicted probabilities are also converted into three operational risk levels:

| Risk Level | Probability |
| :--- | :--- |
| **LOW** | < 0.35 |
| **MEDIUM** | 0.35–0.65 |
| **HIGH** | > 0.65 |

These risk levels are intended for demonstration and decision-support purposes.

---

## Explainability

The project uses SHAP (SHapley Additive exPlanations) to explain model predictions.

Two types of explanations are provided:

### Global Explainability

The global SHAP analysis shows which behavioral features have the largest average contribution to the model's predictions across the training data.

### Individual Explainability

For an individual learner, the dashboard displays the behavioral features that contributed most strongly to the learner's predicted disengagement probability.

> **Note:** SHAP values describe model attribution. They do not establish that a feature causally caused disengagement.

---

## Intervention Recommendations

The project includes a rule-based recommendation engine that maps learner risk and behavioral signals to demonstration intervention types such as:

- **Monitor**
- **Nudge**
- **Reminder**
- **Outreach**
- **Urgent outreach**

These recommendations are deterministic rules built on top of the model outputs and behavioral features.

They are not validated interventions, and their effectiveness in reducing actual disengagement has not been experimentally or institutionally tested.

---

## Interactive Dashboard

The project includes a Streamlit dashboard with four main sections:

### 1. Overview
- Cohort size
- Operational disengagement rate
- High-risk learner count
- Average predicted probability
- Risk distribution

### 2. Behavioral Analysis
- Early click behavior
- Activity trends
- Inactivity patterns
- Risk-group comparisons

### 3. At-Risk Learners
- Learner ID
- Risk level
- Predicted probability
- Behavioral features
- Recommended action
- CSV export

### 4. Individual Learner
- Predicted disengagement probability
- Risk level
- Behavioral profile
- SHAP-based explanation
- Rule-based recommendation

---

## Project Structure

```text
early-disengagement/
│
├── data/
│   └── raw/
│       └── OULAD CSV files
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── best_model.joblib
│   ├── predictions.csv
│   ├── features.csv
│   ├── shap_values.npy
│   ├── shap_feature_names.npy
│   ├── shap_train_ids.npy
│   ├── shap_global.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── problem_definition.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   └── recommend.py
│
├── run_pipeline.py
├── start.bat
├── start.sh
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Local Setup

Clone the repository and create a Python virtual environment.

### Create virtual environment
```bash
python -m venv .venv
```

### Activate on Windows
```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## OULAD Dataset

The raw OULAD dataset is intentionally not included in this repository because the raw CSV files are relatively large.

Download the OULAD dataset separately and place the required CSV files inside:

```text
data/raw/
```

The project expects the standard OULAD CSV files used by the data-loading pipeline.

The raw dataset directory is excluded from Git using `.gitignore`.

---

## Generate Prediction Artifacts

After placing the OULAD files in `data/raw/`, run:

```bash
python run_pipeline.py
```

The pipeline:

1. Loads the OULAD data
2. Scopes the data to AAA 2013J
3. Computes the custom disengagement target
4. Engineers the Day 0–14 behavioral features
5. Loads the frozen trained model
6. Generates predictions for all eligible learners
7. Generates rule-based recommendations
8. Saves the artifacts used by the dashboard

The main generated dashboard artifacts are:

- `models/predictions.csv`
- `models/features.csv`

---

## Run the Dashboard

After generating the artifacts:

```bash
streamlit run dashboard/app.py
```

The dashboard can then be opened in the browser using the local Streamlit address shown in the terminal.

### One-Command Local Run

**Windows:**
```cmd
start.bat
```

**Mac/Linux:**
```bash
./start.sh
```

These scripts run the prediction pipeline first and then launch the Streamlit dashboard.

---

## Streamlit Cloud Deployment

The Streamlit application entry point is:

```text
dashboard/app.py
```

The deployed dashboard uses pre-generated artifacts rather than running the complete OULAD data-processing pipeline on Streamlit Cloud.

The required deployment artifacts are:

- `models/best_model.joblib`
- `models/predictions.csv`
- `models/features.csv`

The raw OULAD dataset is intentionally excluded from the repository and is not required by the dashboard itself.

### Reproducing the Prediction Artifacts

If the prediction artifacts need to be regenerated locally:

1. Download the OULAD dataset separately.
2. Place the required CSV files in `data/raw/`.
3. Run:
   ```bash
   python run_pipeline.py
   ```
4. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

---

## Deployment Configuration

The project includes a Streamlit configuration file:

```text
.streamlit/config.toml
```

This provides the dashboard theme and basic presentation settings.

Startup scripts are also included:

- `start.bat`
- `start.sh`

---

## Model Evaluation

The final model is evaluated on a separate holdout test set that was not used for model selection or threshold tuning.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- ROC curve
- Precision-Recall curve

Because the disengagement classes are imbalanced, precision, recall, F1-score, and PR-AUC are considered alongside ROC-AUC rather than relying on accuracy alone.

---

## Responsible Use

- This project is intended for research, educational, and demonstration purposes.
- The predictions should be treated as decision-support signals rather than definitive judgments about individual learners.
- Model explanations describe how the trained model uses available features and do not establish causation.
- The rule-based intervention recommendations are demonstration heuristics and have not been experimentally or institutionally validated.
- The project does not use demographic attributes as model features.

---

## Reproducibility

The trained model and dashboard artifacts are stored separately from the raw OULAD dataset.

A reviewer can:

1. Obtain the OULAD dataset separately.
2. Place the raw files in `data/raw/`.
3. Install the dependencies.
4. Run `run_pipeline.py`.
5. Launch the Streamlit dashboard.

The frozen model artifact allows the dashboard to generate predictions without retraining the model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Plotly
- Streamlit
- Matplotlib
- Seaborn
- Jupyter

---

## Disclaimer

This project is intended for research, educational, and demonstration purposes.

The disengagement predictions are generated by a machine learning model. SHAP explanations describe model attribution and do not establish causation. Intervention recommendations are rule-based heuristics and have not been experimentally or institutionally validated.

