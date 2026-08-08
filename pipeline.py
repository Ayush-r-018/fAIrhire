# ============================================================
# FAIRHIRE PIPELINE
# Exact prediction logic based on Notebook 06
# ============================================================

from pathlib import Path
import re
import string
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pdfplumber
from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# SECTION 2 : Project Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"

DATASET_DIR = PROJECT_ROOT / "dataset"

RESUME_DIR = DATASET_DIR / "resumes"

JOB_DIR = DATASET_DIR / "job_descriptions"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

REAL_TEST_DATA_DIR = PROJECT_ROOT / "real_test_data"

REAL_RESUME_DIR = (
    REAL_TEST_DATA_DIR
    / "resumes"
    / "Cloud, DevOps and SRE"
)

REAL_JOB_DIR = (
    REAL_TEST_DATA_DIR
    / "job_descriptions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SECTION 3 : Load Models
# ============================================================

best_model = joblib.load(
    MODELS_DIR / "best_model.pkl"
)

vectorizer = joblib.load(
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

skill_dictionary = joblib.load(
    MODELS_DIR / "skill_dictionary.pkl"
)

sentence_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# SECTION 4 : NLTK Resources
# ============================================================

nltk.download(
    "stopwords",
    quiet=True
)

nltk.download(
    "wordnet",
    quiet=True
)

nltk.download(
    "omw-1.4",
    quiet=True
)

STOP_WORDS = set(
    stopwords.words("english")
)

LEMMATIZER = WordNetLemmatizer()

# ============================================================
# SECTION 5 : Job Description
# ============================================================

def load_job_description(file_path):
    """
    Read Job Description text file.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()
    
    # ============================================================
# SECTION 6 : Resume Extraction
# ============================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF resume.

    Priority:
        1. pdfplumber
        2. PyPDF2 fallback
    """

    text = ""

    # --------------------------------------------------------
    # Try pdfplumber first
    # --------------------------------------------------------

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

    except Exception:

        text = ""

    # --------------------------------------------------------
    # PyPDF2 fallback
    # --------------------------------------------------------

    if len(text.strip()) == 0:

        try:

            reader = PdfReader(pdf_path)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"

        except Exception as e:

            print(
                f"Error reading PDF: {e}"
            )

            return ""

    # --------------------------------------------------------
    # Basic Cleaning
    # --------------------------------------------------------

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "\t",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    return text

# ============================================================
# SECTION 7 : Text Preprocessing
# ============================================================

def preprocess_text(text):
    """
    Clean Resume / Job Description text.
    """

    if text is None:

        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    # Remove Email
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove Phone Numbers
    text = re.sub(
        r"\+?\d[\d\-\(\)\s]{7,}",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove Numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Remove Extra Spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    words = []

    for word in text.split():

        if word not in STOP_WORDS:

            word = LEMMATIZER.lemmatize(
                word
            )

            words.append(word)

    return " ".join(words)

# ============================================================
# SECTION 8 : Experience Extraction
# ============================================================

def extract_years_of_experience(text):
    """
    Extract total years of experience.

    Returns integer years.
    """

    text = text.lower()

    patterns = [

        r'(\d+)\+?\s*years',

        r'(\d+)\+?\s*year',

        r'(\d+)\s*yrs',

        r'(\d+)\s*yr'

    ]

    years = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        years.extend(matches)

    years = [
        int(x)
        for x in years
    ]

    if len(years) == 0:

        return 0

    return max(years)

# ============================================================
# SECTION 9 : Experience Matching
# ============================================================

def experience_match_score(
    resume_text,
    job_text
):

    resume_exp = extract_years_of_experience(
        resume_text
    )

    job_exp = extract_years_of_experience(
        job_text
    )

    if job_exp == 0:

        score = 100

    else:

        score = min(
            (resume_exp / job_exp) * 100,
            100
        )

    return (
        resume_exp,
        job_exp,
        round(score, 2)
    )
    
# ============================================================
# SECTION 10 : Skill Extraction
# ============================================================

def extract_skills(text):
    """
    Extract skills using the trained skill dictionary.
    """

    text = text.lower()

    extracted = set()

    for skill in skill_dictionary:

        pattern = (
            r"\b"
            + re.escape(skill.lower())
            + r"\b"
        )

        if re.search(
            pattern,
            text
        ):

            extracted.add(
                skill.lower()
            )

    return sorted(extracted)

# ============================================================
# SECTION 11 : Skill Comparison
# ============================================================

def compare_skills(
    resume_text,
    job_text
):

    resume_skills = set(
        extract_skills(resume_text)
    )

    job_skills = set(
        extract_skills(job_text)
    )

    matched = sorted(
        resume_skills & job_skills
    )

    missing = sorted(
        job_skills - resume_skills
    )

    return (
        resume_skills,
        job_skills,
        matched,
        missing
    )
    
# ============================================================
# SECTION 12 : Keyword Coverage
# ============================================================

def keyword_coverage(
    resume_text,
    job_description
):
    """
    Calculate keyword coverage.
    """

    _, _, matched, missing = compare_skills(
        resume_text,
        job_description
    )

    total = (
        len(matched)
        + len(missing)
    )

    if total == 0:

        return (
            0.0,
            matched,
            missing
        )

    coverage = (
        len(matched)
        / total
    ) * 100

    return (
        round(coverage, 2),
        matched,
        missing
    )
    
# ============================================================
# SECTION 13 : Semantic Similarity
# ============================================================

def semantic_similarity(
    resume_text,
    job_description
):

    resume_embedding = sentence_model.encode(
        resume_text,
        convert_to_numpy=True
    )

    job_embedding = sentence_model.encode(
        job_description,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(

        [resume_embedding],

        [job_embedding]

    )[0][0]

    return round(
        float(similarity),
        4
    )
    
# ============================================================
# SECTION 14 : Candidate Score
# ============================================================

def calculate_candidate_score(
    selection_probability,
    keyword_coverage,
    semantic_similarity,
    experience_score,
    matched_count,
    missing_count
):

    keyword_score = float(
        keyword_coverage
    )

    semantic_score = (
        float(semantic_similarity)
        * 100
    )

    experience_score = float(
        experience_score
    )

    total_skills = (
        matched_count
        + missing_count
    )

    if total_skills == 0:

        skill_score = 0

    else:

        skill_score = (
            matched_count
            / total_skills
        ) * 100

    candidate_score = (

        0.05
        * float(selection_probability)

        +

        0.30
        * keyword_score

        +

        0.30
        * semantic_score

        +

        0.15
        * experience_score

        +

        0.20
        * skill_score

    )

    return round(
        candidate_score,
        2
    )
    
# ============================================================
# SECTION 15 : Candidate Prediction
# ============================================================

def predict_candidate(
    resume_text,
    job_description
):

    # --------------------------------------------------------
    # Clean Text
    # --------------------------------------------------------

    clean_resume = preprocess_text(
        resume_text
    )

    clean_job = preprocess_text(
        job_description
    )

    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    combined_text = (
        clean_resume
        + " "
        + clean_job
    )

    tfidf_vector = vectorizer.transform(
        [combined_text]
    )

    probability = (
        best_model
        .predict_proba(
            tfidf_vector
        )[0][1]
    )

    probability = round(
        probability * 100,
        2
    )

    # --------------------------------------------------------
    # Keyword Coverage
    # --------------------------------------------------------

    (
        keyword_score,
        matched,
        missing
    ) = keyword_coverage(
        clean_resume,
        clean_job
    )

    # --------------------------------------------------------
    # Semantic Similarity
    # --------------------------------------------------------

    semantic_score = semantic_similarity(
        clean_resume,
        clean_job
    )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    (
        resume_exp,
        job_exp,
        experience_score
    ) = experience_match_score(
        resume_text,
        job_description
    )

    # --------------------------------------------------------
    # Candidate Score
    # --------------------------------------------------------

    candidate_score = calculate_candidate_score(

        probability,

        keyword_score,

        semantic_score,

        experience_score,

        len(matched),

        len(missing)

    )

    # --------------------------------------------------------
    # Final Decision
    # --------------------------------------------------------

    decision = (

        "SELECTED"

        if candidate_score >= 75

        else

        "REJECTED"

    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "Decision": decision,

        "Probability": probability,

        "Keyword Coverage": keyword_score,

        "Semantic Similarity": semantic_score,

        "Experience Match":
            f"{resume_exp}/{job_exp}",

        "Experience Score":
            experience_score,

        "Candidate Score":
            candidate_score,

        "Matched Skills":
            matched,

        "Missing Skills":
            missing,

        "Matched Count":
            len(matched),

        "Missing Count":
            len(missing)

    }
    
# ============================================================
# SECTION 16 : Predict PDF Candidate
# ============================================================

def predict_pdf_candidate(
    resume_pdf_path,
    job_description
):

    resume_text = extract_text_from_pdf(
        resume_pdf_path
    )

    if not resume_text.strip():

        raise ValueError(
            "Could not extract text from the resume PDF."
        )

    result = predict_candidate(
        resume_text,
        job_description
    )

    return result

# ============================================================
# SECTION 17 : Rank Multiple Candidates
# ============================================================

def rank_candidates(
    resume_files,
    job_description
):

    results_list = []

    for idx, pdf_file in enumerate(
        resume_files,
        start=1
    ):

        resume_text = extract_text_from_pdf(
            pdf_file
        )

        result = predict_candidate(
            resume_text=resume_text,
            job_description=job_description
        )

        result["SN"] = idx

        result["Candidate Name"] = (
            Path(pdf_file).stem
        )

        results_list.append(
            result
        )

    ranking = pd.DataFrame(
        results_list
    )

    ranking.rename(
        columns={
            "Probability":
                "Selection Probability"
        },
        inplace=True
    )

    ranking = ranking.sort_values(
        by="Candidate Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    ranking["SN"] = range(
        1,
        len(ranking) + 1
    )

    return ranking

# ============================================================
# SECTION 18 : Export Ranking CSV
# ============================================================

def export_ranking_csv(
    ranking,
    output_dir=None
):

    from datetime import datetime

    if output_dir is None:

        output_dir = OUTPUT_DIR

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_file = (
        output_dir
        / (
            "candidate_ranking_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )
    )

    ranking.to_csv(
        csv_file,
        index=False
    )

    return csv_file

# ============================================================
# SECTION 19 : Verification
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "FAIRHIRE PIPELINE"
    )

    print(
        "Pipeline loaded successfully."
    )

    print()

    print(
        "Model :",
        type(best_model).__name__
    )

    print(
        "TF-IDF Features :",
        len(
            vectorizer
            .get_feature_names_out()
        )
    )

    print(
        "Skill Dictionary :",
        len(skill_dictionary)
    )

    print("=" * 70)