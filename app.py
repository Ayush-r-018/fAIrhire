# ============================================================
# FAIRHIRE STREAMLIT APPLICATION
# ============================================================

import io
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

import pipeline

# ============================================================
# SECTION 2 : PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="FairHire",

    page_icon="📄",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ============================================================
# SECTION 3 : CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .decision-selected {
        padding: 15px;
        border-radius: 10px;
        background-color: #dff5e5;
        color: #087f23;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
    }

    .decision-rejected {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffe1e1;
        color: #b00020;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SECTION 5 : SIDEBAR
# ============================================================

with st.sidebar:

    st.header("FairHire")

    st.markdown(
        "### Model"
    )

    st.success(
        "XGBoost Classifier"
    )

    st.markdown(
        "### Explainability"
    )

    st.info(
        "LIME"
    )

    st.markdown(
        "### Decision Threshold"
    )

    st.write(
        "Candidate Score ≥ 75"
    )

    st.divider()

    st.caption(
        "Minor Project"
    )
    
# ============================================================
# SECTION 6 : RESUME UPLOAD
# ============================================================

st.header(
    "Upload Resume"
)

resume_file = st.file_uploader(

    "Choose Resume (PDF)",

    type=["pdf"],

    key="resume_uploader"

)

# ============================================================
# SECTION 7 : JOB DESCRIPTION UPLOAD
# ============================================================

st.header(
    "Upload Job Description"
)

job_file = st.file_uploader(

    "Choose Job Description (TXT)",

    type=["txt"],

    key="job_uploader"

)

# ============================================================
# SECTION 8 : PREDICT BUTTON
# ============================================================

predict_button = st.button(

    "🚀 Predict Candidate",

    width="stretch",

    type="primary",

    key="predict_candidate_button"

)

# ============================================================
# SECTION 9 : RUN PREDICTION
# ============================================================

if predict_button:

    if resume_file is None:

        st.error(
            "Please upload a resume PDF."
        )

        st.stop()

    if job_file is None:

        st.error(
            "Please upload a job description TXT file."
        )

        st.stop()

    # --------------------------------------------------------
    # Read JD
    # --------------------------------------------------------

    job_description = (
        job_file
        .getvalue()
        .decode("utf-8")
    )

    # --------------------------------------------------------
    # Save uploaded resume temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            resume_file.getvalue()
        )

        temp_resume_path = (
            temp_file.name
        )

    # --------------------------------------------------------
    # Pipeline Prediction
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Running FairHire prediction..."
        ):

            resume_text = (
                pipeline
                .extract_text_from_pdf(
                    temp_resume_path
                )
            )

            if not resume_text.strip():

                st.error(
                    "Unable to extract text from the resume."
                )

                st.stop()

            result = (
                pipeline
                .predict_candidate(
                    resume_text,
                    job_description
                )
            )

        st.session_state[
            "prediction_result"
        ] = result

        st.session_state[
            "resume_text"
        ] = resume_text

        st.session_state[
            "job_description"
        ] = job_description

        st.session_state[
            "candidate_name"
        ] = Path(
            resume_file.name
        ).stem

        st.success(
            "Prediction Completed Successfully."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
        
# ============================================================
# SECTION 10 : RESULT STATE
# ============================================================

if "prediction_result" not in st.session_state:

    st.info(
        "Upload a resume and job description to begin."
    )

    st.stop()

result = st.session_state[
    "prediction_result"
]

candidate_name = st.session_state[
    "candidate_name"
]

# ============================================================
# SECTION 11 : METRIC CARDS
# ============================================================

st.divider()

st.subheader(
    "Candidate Prediction"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Candidate Score",
        f"{result['Candidate Score']:.2f}"
    )

with col2:

    st.metric(
        "Selection Probability",
        f"{result['Probability']:.2f}%"
    )

with col3:

    st.metric(
        "Semantic Similarity",
        f"{result['Semantic Similarity']:.3f}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Keyword Coverage",
        f"{result['Keyword Coverage']:.2f}%"
    )

with col5:

    st.metric(
        "Experience Match",
        result["Experience Match"]
    )

with col6:

    st.metric(
        "Matched Skills",
        result["Matched Count"]
    )
    
# ============================================================
# SECTION 12 : FINAL DECISION
# ============================================================

decision = result["Decision"]

if decision == "SELECTED":

    st.markdown(
        '<div class="decision-selected">'
        '✓ SELECTED'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="decision-rejected">'
        '✗ REJECTED'
        '</div>',
        unsafe_allow_html=True
    )
    
# ============================================================
# SECTION 13 : SKILL ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "Skill Analysis"
)

skill_col1, skill_col2 = st.columns(2)

with skill_col1:

    st.markdown(
        "### ✓ Matched Skills"
    )

    if result["Matched Skills"]:

        st.write(
            ", ".join(
                result["Matched Skills"]
            )
        )

    else:

        st.write(
            "No matched skills found."
        )

with skill_col2:

    st.markdown(
        "### ✗ Missing Skills"
    )

    if result["Missing Skills"]:

        st.write(
            ", ".join(
                result["Missing Skills"]
            )
        )

    else:

        st.write(
            "No missing skills."
        )
        
# ============================================================
# SECTION 14 : CANDIDATE SCORE BREAKDOWN
# ============================================================

st.divider()

st.subheader(
    "Candidate Score Breakdown"
)

selection_probability = float(
    result["Probability"]
)

keyword_score = float(
    result["Keyword Coverage"]
)

semantic_score = (
    float(
        result["Semantic Similarity"]
    )
    * 100
)

experience_score = float(
    result["Experience Score"]
)

matched_count = int(
    result["Matched Count"]
)

missing_count = int(
    result["Missing Count"]
)

total_skills = (
    matched_count
    + missing_count
)

skill_score = (

    (
        matched_count
        / total_skills
    ) * 100

    if total_skills > 0

    else 0
)

components = [
    "Selection Probability",
    "Keyword Coverage",
    "Semantic Similarity",
    "Experience Match",
    "Skill Match"
]

weights = [
    0.05,
    0.30,
    0.30,
    0.15,
    0.20
]

values = [
    selection_probability,
    keyword_score,
    semantic_score,
    experience_score,
    skill_score
]

contributions = [
    value * weight
    for value, weight
    in zip(values, weights)
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

bars = ax.bar(
    components,
    contributions
)

ax.set_ylabel(
    "Contribution to Candidate Score"
)

ax.set_title(
    f"Weighted Candidate Score — {candidate_name}"
)

ax.axhline(
    result["Candidate Score"],
    linestyle="--",
    linewidth=1
)

for bar, value in zip(
    bars,
    contributions
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value:.2f}",

        ha="center",

        va="bottom"
    )

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)

# ============================================================
# SECTION 15 : LIME EXPLANATION
# ============================================================

from lime.lime_text import LimeTextExplainer

st.divider()

st.subheader(
    "LIME Explainability"
)

st.write(
    "LIME explains which resume words most influence "
    "the model prediction for this candidate."
)

explainer = LimeTextExplainer(
    class_names=[
        "Rejected",
        "Selected"
    ],
    split_expression=r"\W+",
    bow=True
)

# ============================================================
# SECTION 16 : LIME PREDICTION FUNCTION
# ============================================================

def lime_predict(texts):

    job_description = st.session_state[
        "job_description"
    ]

    predictions = []

    for text in texts:

        clean_resume = (
            pipeline
            .preprocess_text(text)
        )

        clean_job = (
            pipeline
            .preprocess_text(
                job_description
            )
        )

        combined_text = (
            clean_resume
            + " "
            + clean_job
        )

        tfidf_vector = (
            pipeline
            .vectorizer
            .transform(
                [combined_text]
            )
        )

        probabilities = (
            pipeline
            .best_model
            .predict_proba(
                tfidf_vector
            )
        )

        predictions.append(
            probabilities[0]
        )

    return predictions

# ============================================================
# SECTION 17 : GENERATE LIME EXPLANATION
# ============================================================

if st.button(
    "🔍 Generate LIME Explanation",
    key="lime_button"
):

    with st.spinner(
        "Generating LIME explanation..."
    ):

        lime_exp = explainer.explain_instance(

            st.session_state[
                "resume_text"
            ],

            lime_predict,

            num_features=10,

            num_samples=500
        )

    st.session_state[
        "lime_explanation"
    ] = lime_exp

    st.success(
        "LIME explanation generated."
    )
    
# ============================================================
# SECTION 18 : LIME CHART
# ============================================================

if "lime_explanation" in st.session_state:

    lime_exp = st.session_state[
        "lime_explanation"
    ]

    lime_items = lime_exp.as_list(
        label=1
    )

    lime_df = pd.DataFrame(
        lime_items,
        columns=[
            "Feature",
            "Weight"
        ]
    )

    lime_df = lime_df.sort_values(
        "Weight"
    )

    st.subheader(
        "Top LIME Features"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        lime_df["Feature"],
        lime_df["Weight"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "LIME Weight"
    )

    ax.set_title(
        "Local Explanation of Candidate Prediction"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    st.dataframe(
        lime_df,
        width="stretch"
    )
    
# ============================================================
# SECTION 19 : CSV REPORT
# ============================================================

st.divider()

st.subheader(
    "Download Reports"
)

report_row = {

    "Candidate Name":
        candidate_name,

    "Decision":
        result["Decision"],

    "Candidate Score":
        result["Candidate Score"],

    "Selection Probability":
        result["Probability"],

    "Keyword Coverage":
        result["Keyword Coverage"],

    "Semantic Similarity":
        result["Semantic Similarity"],

    "Experience Match":
        result["Experience Match"],

    "Experience Score":
        result["Experience Score"],

    "Matched Count":
        result["Matched Count"],

    "Missing Count":
        result["Missing Count"],

    "Matched Skills":
        ", ".join(
            result["Matched Skills"]
        ),

    "Missing Skills":
        ", ".join(
            result["Missing Skills"]
        )
}

report_df = pd.DataFrame(
    [report_row]
)

csv_data = report_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(

    label="📥 Download CSV",

    data=csv_data,

    file_name=(
        f"{candidate_name}_"
        "FairHire_Report.csv"
    ),

    mime="text/csv",

    key="download_csv"
)

# ============================================================
# SECTION 20 : PDF REPORT
# ============================================================

def create_pdf_report(
    candidate_name,
    result
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "FAIRHIRE",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "Candidate Explainability Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    data = [

        [
            "Candidate",
            candidate_name
        ],

        [
            "Decision",
            result["Decision"]
        ],

        [
            "Candidate Score",
            f"{result['Candidate Score']:.2f}"
        ],

        [
            "Selection Probability",
            f"{result['Probability']:.2f}%"
        ],

        [
            "Keyword Coverage",
            f"{result['Keyword Coverage']:.2f}%"
        ],

        [
            "Semantic Similarity",
            f"{result['Semantic Similarity']:.3f}"
        ],

        [
            "Experience Match",
            result["Experience Match"]
        ],

        [
            "Matched Skills",
            str(result["Matched Count"])
        ],

        [
            "Missing Skills",
            str(result["Missing Count"])
        ]
    ]

    table = Table(
        data,
        colWidths=[
            180,
            300
        ]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Matched Skills",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            ", ".join(
                result["Matched Skills"]
            )
            if result["Matched Skills"]
            else "None",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Missing Skills",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            ", ".join(
                result["Missing Skills"]
            )
            if result["Missing Skills"]
            else "None",
            styles["BodyText"]
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()

# ============================================================
# SECTION 21 : DOWNLOAD PDF
# ============================================================

pdf_data = create_pdf_report(
    candidate_name,
    result
)

st.download_button(

    label="📄 Download PDF Report",

    data=pdf_data,

    file_name=(
        f"{candidate_name}_"
        "FairHire_Report.pdf"
    ),

    mime="application/pdf",

    key="download_pdf"
)