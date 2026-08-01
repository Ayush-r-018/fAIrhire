"""
FairHire
Keyword Coverage Module
"""

import re


def extract_keywords(text):
    """
    Extract unique keywords from cleaned text.
    """

    if not isinstance(text, str):
        return set()

    words = re.findall(r"\b[a-zA-Z0-9_+#.]+\b", text.lower())

    # Remove very short words
    return {
        word
        for word in words
        if len(word) > 2
    }


def calculate_keyword_coverage(resume_text, job_text):
    """
    Calculate keyword coverage percentage.

    Coverage =
        matched keywords /
        total job keywords
    """

    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_text)

    if len(job_keywords) == 0:
        return 0.0

    matched = resume_keywords.intersection(job_keywords)

    coverage = (
        len(matched) /
        len(job_keywords)
    ) * 100

    return round(coverage, 2)