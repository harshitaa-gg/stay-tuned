"""Phase 13: Interactive Streamlit Dashboard for Early Learner Disengagement Prediction.

This module provides an interactive decision-support visualization interface for academic advisors,
educators, and learning analysts exploring disengagement risk predictions for Module AAA,
Presentation 2013J at Day 14.

Strict Architecture & Governance Rules:
---------------------------------------
1. Pure Interface Layer: Does NOT train, tune, or alter any machine learning model or threshold.
2. Frozen Model & Threshold: Uses models/best_model.joblib and the frozen 0.35 threshold.
3. Module Reuse: Directly reuses src.predict, src.explain, and src.recommend.
4. Non-Causal & Ethical Disclaimers: Transparently distinguishes model attribution (SHAP)
   and rule-based intervention heuristics from causal assertions and validated clinical actions.
5. Robust Plotly Graphics: All charts are interactive Plotly visualizations (no matplotlib).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_model
from src.explain import explain_student
from src.recommend import get_intervention

# Human-readable feature display names mapping
FEATURE_DISPLAY_NAMES = {
    "total_early_clicks": "Total Early Clicks",
    "clicks_per_active_day": "Clicks per Active Day",
    "active_days": "Active Days",
    "longest_inactivity_gap": "Longest Inactivity Gap (Days)",
    "days_since_last_active": "Days Since Last Active",
    "activity_delta": "Activity Delta (W2 - W1 Clicks)",
    "weekly_click_ratio": "Weekly Click Ratio",
    "studied_credits": "Studied Credits",
}

# Standardized color scheme for risk tiers
RISK_COLORS = {
    "LOW": "#2ecc71",       # Emerald Green
    "MEDIUM": "#f39c12",    # Amber Yellow
    "HIGH": "#e74c3c",      # Crimson Red
}


# -----------------------------------------------------------------------------
# 1. Cached Data and Model Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_pipeline():
    """Load and cache the frozen final model pipeline."""
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        return None
    pipeline, threshold, artifact = load_model(model_path)
    return pipeline, threshold, artifact


@st.cache_data
def get_dashboard_data():
    """Load and cache predictions and feature artifacts."""
    preds_path = PROJECT_ROOT / "models" / "predictions.csv"
    feats_path = PROJECT_ROOT / "models" / "features.csv"

    if not preds_path.exists() or not feats_path.exists():
        return None, None

    preds_df = pd.read_csv(preds_path)
    feats_df = pd.read_csv(feats_path)
    return preds_df, feats_df


@st.cache_data
def get_shap_artifacts():
    """Load precomputed training SHAP values and feature names."""
    shap_vals_path = PROJECT_ROOT / "models" / "shap_values.npy"
    shap_names_path = PROJECT_ROOT / "models" / "shap_feature_names.npy"
    shap_ids_path = PROJECT_ROOT / "models" / "shap_train_ids.npy"

    if not (shap_vals_path.exists() and shap_names_path.exists() and shap_ids_path.exists()):
        return None, None, None

    shap_vals = np.load(shap_vals_path)
    shap_names = list(np.load(shap_names_path))
    shap_ids = np.load(shap_ids_path)
    return shap_vals, shap_names, shap_ids


# -----------------------------------------------------------------------------
# 2. Main Application Flow & Error Handling
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Early Learner Disengagement Dashboard",
        page_icon="🎓",
        layout="wide",
    )

    # Validate presence of necessary artifacts
    preds_df, feats_df = get_dashboard_data()
    model_tuple = get_model_pipeline()

    if preds_df is None or feats_df is None or model_tuple is None:
        st.error("### Dashboard Data or Model Not Found")
        st.info(
            "The required data artifacts could not be located. Please execute the pipeline "
            "before launching the dashboard:\n\n"
            "```bash\n"
            "python run_pipeline.py\n"
            "```"
        )
        st.stop()

    pipeline, threshold, artifact = model_tuple
    shap_vals, shap_names, shap_ids = get_shap_artifacts()

    # Merge features with predictions for unified cohort exploration
    merged_df = preds_df.merge(feats_df, on="id_student", how="left")

    # -------------------------------------------------------------------------
    # 3. Clean Sidebar Navigation
    # -------------------------------------------------------------------------
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Behavioral Analysis", "At-Risk Learners", "Individual Learner"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Stay-Tuned System**\n\n"
        "- Module: **AAA** | Presentation: **2013J**\n"
        "- Decision Point: **Day 14**\n"
        f"- Operational Cutoff: **{threshold:.2f}**\n"
        "- Model: **Random Forest**"
    )

    # -------------------------------------------------------------------------
    # PAGE 1: Overview
    # -------------------------------------------------------------------------
    if page == "Overview":
        st.title("Early Learner Disengagement Dashboard")
        st.subheader("AAA 2013J Module | Prediction at Day 14")
        st.markdown(
            "Early-warning disengagement intelligence for educators and support advisors. "
            "Evaluates student risk at Day 14 using behavioral engagement signals from the "
            "first two weeks of course activity."
        )

        total_learners = len(preds_df)
        counts = preds_df["risk_level"].value_counts()
        high_cnt = int(counts.get("HIGH", 0))
        med_cnt = int(counts.get("MEDIUM", 0))
        low_cnt = int(counts.get("LOW", 0))

        # Operational Disengagement Rate = (MEDIUM + HIGH) / Total
        overall_disengagement_rate = ((med_cnt + high_cnt) / total_learners * 100) if total_learners > 0 else 0.0
        high_pct = (high_cnt / total_learners * 100) if total_learners > 0 else 0.0
        avg_prob = preds_df["disengagement_probability"].mean()

        # Display four summary metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Learners Analyzed", f"{total_learners:,}")
        col2.metric(
            "Overall Disengagement Rate",
            f"{overall_disengagement_rate:.1f}%",
            help="Learners classified as MEDIUM or HIGH risk (threshold >= 0.35) requiring intervention.",
        )
        col3.metric("HIGH Risk Learners", f"{high_cnt} ({high_pct:.1f}%)")
        col4.metric("Average Disengagement Probability", f"{avg_prob:.1%}")

        st.markdown("---")

        # Visualizations: Risk Distribution and Proportion
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            risk_order = ["LOW", "MEDIUM", "HIGH"]
            bar_data = pd.DataFrame({
                "Risk Level": risk_order,
                "Learners": [int(counts.get(r, 0)) for r in risk_order],
                "Percentage": [float(counts.get(r, 0) / total_learners * 100) for r in risk_order],
            })
            fig_bar = px.bar(
                bar_data,
                x="Risk Level",
                y="Learners",
                color="Risk Level",
                color_discrete_map=RISK_COLORS,
                text="Learners",
                title="Risk Level Distribution",
            )
            fig_bar.update_traces(
                texttemplate="%{text} (%{customdata[0]:.1f}%)",
                customdata=bar_data[["Percentage"]],
                textposition="outside",
            )
            fig_bar.update_layout(
                showlegend=False,
                xaxis_title="Assigned Risk Tier",
                yaxis_title="Number of Learners",
                yaxis_range=[0, max(bar_data["Learners"]) * 1.15],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=risk_order,
                        values=[int(counts.get(r, 0)) for r in risk_order],
                        hole=0.48,
                        marker_colors=[RISK_COLORS[r] for r in risk_order],
                        textinfo="label+percent",
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig_donut.update_layout(
                title="Risk Proportion by Tier",
                annotations=[
                    dict(
                        text=f"{total_learners}<br>Students",
                        x=0.5,
                        y=0.5,
                        font_size=16,
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.info(
            "**Governance Note**: Risk tiers reflect the frozen operational cutoff (0.35) "
            "established in Phase 8.2 to prioritize learner support sensitivity. "
            "Probabilities are model propensity estimates and do not determine academic grades."
        )

    # -------------------------------------------------------------------------
    # PAGE 2: Behavioral Analysis
    # -------------------------------------------------------------------------
    elif page == "Behavioral Analysis":
        st.title("Behavioral Signals")
        st.markdown(
            "Explore early VLE behavioral indicators captured across Days 0–14 to "
            "understand the empirical separation between learner risk groups."
        )

        # Chart 1: Total Early Clicks Histogram (HIGH vs LOW)
        st.markdown("### 1. Total Early Clicks Distribution (HIGH vs. LOW Risk)")
        high_low_df = merged_df[merged_df["risk_level"].isin(["LOW", "HIGH"])].copy()
        if not high_low_df.empty:
            fig_hist = px.histogram(
                high_low_df,
                x="total_early_clicks",
                color="risk_level",
                barmode="overlay",
                opacity=0.6,
                color_discrete_map=RISK_COLORS,
                title="Distribution of Total Early Clicks: HIGH vs. LOW Risk",
                labels={"total_early_clicks": "Total Early Clicks (Days 0–14)", "risk_level": "Risk Level"},
                nbins=30,
            )
            fig_hist.update_layout(
                xaxis_title="Total Early Clicks",
                yaxis_title="Learner Count",
                legend_title="Risk Tier",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown(
                "Shows the distribution of early VLE activity for LOW- and HIGH-risk learners. "
                "Differences in early activity can provide an intuitive view of behavioral separation "
                "between risk groups."
            )
        else:
            st.warning("Insufficient learner data to render clicks distribution.")

        st.markdown("---")

        # Chart 2: Week 1 vs Week 2 Clicks Scatter Plot
        st.markdown("### 2. Week 1 vs. Week 2 Clicks Exploration")
        if "week1_clicks" in merged_df.columns and "week2_clicks" in merged_df.columns:
            fig_scatter = px.scatter(
                merged_df,
                x="week1_clicks",
                y="week2_clicks",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
                hover_data=["id_student", "disengagement_probability"],
                title="Week 1 Clicks vs. Week 2 Clicks by Risk Tier",
                labels={"week1_clicks": "Week 1 Clicks (Days 0–6)", "week2_clicks": "Week 2 Clicks (Days 7–14)"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info(
                "Raw weekly click components (`week1_clicks` and `week2_clicks`) are intermediate "
                "engineering variables from Phase 4 and are not part of the exported canonical "
                "feature artifact. The model directly utilizes their derived representations: "
                "`activity_delta` and `weekly_click_ratio`."
            )

        st.markdown("---")

        # Two-column layout for Activity Delta and Inactivity Gap
        col_c3, col_c5 = st.columns(2)

        # Chart 3: Average Activity Delta
        with col_c3:
            st.markdown("### 3. Average Activity Delta")
            delta_summary = (
                merged_df.groupby("risk_level", as_index=False)["activity_delta"]
                .mean()
            )
            # Ensure correct risk order
            delta_summary["risk_order"] = delta_summary["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
            delta_summary = delta_summary.sort_values("risk_order").drop(columns=["risk_order"])

            fig_delta = px.bar(
                delta_summary,
                x="risk_level",
                y="activity_delta",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                text="activity_delta",
                title="Average Activity Delta by Risk Level",
            )
            fig_delta.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_delta.update_layout(
                xaxis_title="Risk Level",
                yaxis_title="Mean Activity Delta (W2 - W1)",
                showlegend=False,
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.markdown(
                "Shows the average change in activity across risk groups. "
                "Negative values indicate declining activity over the observation window."
            )

        # Chart 5: Longest Inactivity Gap Box Plot
        with col_c5:
            st.markdown("### 4. Longest Inactivity Gap")
            fig_box = px.box(
                merged_df,
                x="risk_level",
                y="longest_inactivity_gap",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]},
                title="Longest Inactivity Gap by Risk Level",
                labels={"longest_inactivity_gap": "Longest Inactivity Gap (Days)", "risk_level": "Risk Level"},
            )
            fig_box.update_layout(
                xaxis_title="Risk Level",
                yaxis_title="Consecutive Inactive Days",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown(
                "Shows how inactivity gaps differ across predicted risk groups. "
                "Larger inactivity gaps indicate longer periods without observed VLE activity."
            )

        st.markdown("---")

        # Chart 4: Early Assessment Submission Status
        st.markdown("### 5. Early Assessment Submissions")
        st.warning(
            "Early assessment submission data is unavailable for the Day 0–14 observation "
            "window in AAA 2013J."
        )
        st.caption(
            "In Module AAA 2013J, no assessments were scheduled or due during Days 0–14 "
            "(the first course milestone, TMA 01, was due on Day 19). Missing assessment data "
            "is an institutional scheduling characteristic and does not indicate student omission."
        )

    # -------------------------------------------------------------------------
    # PAGE 3: At-Risk Learners
    # -------------------------------------------------------------------------
    elif page == "At-Risk Learners":
        st.title("At-Risk Learners")
        st.markdown(
            "Prioritized roster of learners flagged for proactive outreach "
            "(`risk_level` in **MEDIUM**, **HIGH**). Use this table for academic triage "
            "and intervention coordination."
        )

        # Filter strictly to MEDIUM and HIGH risk learners
        at_risk_df = merged_df[merged_df["risk_level"].isin(["MEDIUM", "HIGH"])].copy()
        at_risk_df = at_risk_df.sort_values("disengagement_probability", ascending=False).reset_index(drop=True)

        st.markdown(f"**Total Flagged Learners:** `{len(at_risk_df):,}` / `{len(merged_df):,}` ({len(at_risk_df) / len(merged_df) * 100:.1f}%)")

        if not at_risk_df.empty:
            # Prepare formatted display table
            display_df = pd.DataFrame({
                "Student ID": at_risk_df["id_student"],
                "Risk Level": at_risk_df["risk_level"],
                "Disengagement Probability": at_risk_df["disengagement_probability"].apply(lambda p: f"{p * 100:.1f}%"),
                "Total Clicks": at_risk_df["total_early_clicks"],
                "Activity Delta": at_risk_df["activity_delta"],
                "Recommendation": at_risk_df["recommendation"],
            })

            # Custom styling function for risk levels
            def highlight_risk(val):
                if val == "HIGH":
                    return "background-color: rgba(231, 76, 60, 0.25); color: #900C3F; font-weight: bold;"
                elif val == "MEDIUM":
                    return "background-color: rgba(243, 156, 18, 0.25); color: #B7950B; font-weight: bold;"
                return ""

            styler = display_df.style
            map_fn = getattr(styler, "map", getattr(styler, "applymap", None))
            if map_fn is not None:
                styled_table = map_fn(highlight_risk, subset=["Risk Level"])
            else:
                styled_table = styler

            st.dataframe(styled_table, use_container_width=True, height=520)

            # CSV Download Button for support teams
            csv_export = at_risk_df[[
                "id_student",
                "risk_level",
                "disengagement_probability",
                "total_early_clicks",
                "activity_delta",
                "recommendation",
                "recommendation_type",
            ]].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download At-Risk Learners CSV",
                data=csv_export,
                file_name="at_risk_learners_AAA_2013J.csv",
                mime="text/csv",
                help="Download the filtered at-risk student records for advisor follow-up.",
            )
        else:
            st.success("No learners are currently flagged in the MEDIUM or HIGH risk tiers.")

    # -------------------------------------------------------------------------
    # PAGE 4: Individual Learner Deep-Dive
    # -------------------------------------------------------------------------
    elif page == "Individual Learner":
        st.title("Individual Learner Analysis")
        st.markdown(
            "Detailed diagnostic view for an individual learner, combining predicted "
            "risk probability, behavioral metrics, model attribution (SHAP), and rule-based "
            "intervention recommendations."
        )

        # Default student: highest predicted disengagement probability
        sorted_preds = preds_df.sort_values("disengagement_probability", ascending=False)
        student_id_list = sorted_preds["id_student"].tolist()

        selected_student = st.selectbox(
            "Select Student ID for Deep-Dive Analysis:",
            options=student_id_list,
            index=0,  # highest risk learner by default
            help="Learners are ordered by predicted disengagement risk descending.",
        )

        student_pred = preds_df[preds_df["id_student"] == selected_student].iloc[0]
        student_feat = feats_df[feats_df["id_student"] == selected_student].iloc[0]

        prob = float(student_pred["disengagement_probability"])
        risk = str(student_pred["risk_level"])

        st.markdown("---")

        # 1. Probability Metric & 2. Risk Badge
        kpi_col1, kpi_col2, kpi_col3 = st.columns([1.5, 1.5, 3])

        with kpi_col1:
            st.metric("Disengagement Probability", f"{prob * 100:.1f}%")

        with kpi_col2:
            st.markdown("**Assigned Risk Tier**")
            badge_color = RISK_COLORS.get(risk, "#7f8c8d")
            st.markdown(
                f"<div style='background-color:{badge_color}; color:white; "
                f"padding:10px 18px; border-radius:8px; font-weight:bold; "
                f"font-size:1.1rem; text-align:center; width: fit-content;'>"
                f"{risk} RISK</div>",
                unsafe_allow_html=True,
            )

        with kpi_col3:
            st.markdown("**Intervention Need**")
            if risk == "HIGH":
                st.markdown("🚨 **Urgent Action Recommended**: Student displays critical disengagement patterns.")
            elif risk == "MEDIUM":
                st.markdown("⚠️ **Supportive Outreach Recommended**: Student is showing engagement decline.")
            else:
                st.markdown("✅ **On-Track**: Student exhibits consistent early engagement.")

        st.markdown("---")

        # 3. Behavioral Feature Summary Table
        st.markdown("### Behavioral Feature Profile")
        summary_rows = [
            ("Total Early Clicks", f"{int(student_feat['total_early_clicks']):,}"),
            ("Clicks per Active Day", f"{float(student_feat['clicks_per_active_day']):.2f}"),
            ("Active Days", f"{int(student_feat['active_days'])} / 15 days"),
            ("Longest Inactivity Gap", f"{int(student_feat['longest_inactivity_gap'])} consecutive days"),
            ("Days Since Last Active", f"{int(student_feat['days_since_last_active'])} days ago"),
            ("Activity Delta", f"{int(student_feat['activity_delta']):+d} clicks (W2 vs W1)"),
            ("Weekly Click Ratio", f"{float(student_feat['weekly_click_ratio']):.2f}"),
            ("Studied Credits", f"{int(student_feat['studied_credits'])} credits"),
        ]
        feat_summary_df = pd.DataFrame(summary_rows, columns=["Behavioral Feature", "Observed Value (Days 0–14)"])
        st.table(feat_summary_df)

        st.markdown("---")

        # 4. SHAP Local Attribution Explanation
        st.markdown("### Model Explanation (SHAP Attribution)")
        if shap_vals is not None and shap_ids is not None and selected_student in shap_ids:
            try:
                # Reconstruct training features row-aligned with precomputed SHAP matrix
                train_feats_subset = feats_df.set_index("id_student").loc[shap_ids]

                explanation = explain_student(
                    student_id=selected_student,
                    predictions_df=preds_df,
                    features_df=train_feats_subset,
                    shap_values=shap_vals,
                    feature_names=shap_names,
                    top_n=5,
                )

                top_factors = explanation["top_risk_factors"]
                st.markdown("**Top 5 Contributing Factors to Model Prediction:**")

                for factor in top_factors:
                    feat_raw = factor["feature"]
                    readable_feat = FEATURE_DISPLAY_NAMES.get(feat_raw, feat_raw.replace("_", " ").title())
                    direction = factor["direction"]
                    shap_val = factor["shap_value"]

                    if direction == "positive":
                        icon = "🔺"
                        st.markdown(
                            f"{icon} **↑ {readable_feat}** increased risk "
                            f"(SHAP impact: `+{abs(shap_val):.4f}`)"
                        )
                    elif direction == "negative":
                        icon = "🔻"
                        st.markdown(
                            f"{icon} **↓ {readable_feat}** reduced risk "
                            f"(SHAP impact: `-{abs(shap_val):.4f}`)"
                        )
                    else:
                        st.markdown(f"▫️ **{readable_feat}** neutral contribution (`{shap_val:.4f}`)")

                st.caption(
                    "**SHAP Attribution Guidance**: Directions describe the statistical influence on the "
                    "model's disengagement probability output: `↑` pushed the predicted risk higher, while "
                    "`↓` acted as a protective indicator pushing risk lower. "
                    "SHAP values explain model decision logic only and do not establish causation."
                )
            except Exception as e:
                st.info(f"SHAP explanation is currently unavailable for this learner.")
        else:
            st.info("SHAP explanation is currently unavailable for this learner.")

        st.markdown("---")

        # 5. Recommended Intervention
        st.markdown("### Recommended Intervention")
        try:
            intervention = get_intervention(
                student_id=selected_student,
                predictions_df=preds_df,
                features_df=feats_df.set_index("id_student"),
            )
            rec_text = intervention["recommendation"]
            rec_type = intervention["recommendation_type"]

            st.success(f"**Recommended Action**: {rec_text}")
            st.markdown(f"**Intervention Type**: `{rec_type}`")
        except Exception:
            rec_text = str(student_pred.get("recommendation", "Monitor engagement weekly."))
            rec_type = str(student_pred.get("recommendation_type", "monitor"))
            st.success(f"**Recommended Action**: {rec_text}")
            st.markdown(f"**Intervention Type**: `{rec_type}`")

        # 6. Governance Disclaimer
        st.markdown("---")
        st.caption(
            "**Ethical & Operational Disclaimer**: "
            "These predictions are generated by a machine learning model for research and demonstration purposes only. "
            "Recommendations are rule-based. They have not been clinically, experimentally, or institutionally validated."
        )


if __name__ == "__main__":
    main()
