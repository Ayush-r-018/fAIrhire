"""
FairHire
Skill Matching Module (Optimized)
"""

import re


def tokenize(text):
    """
    Convert text into a set of normalized words.
    """

    if not isinstance(text, str):
        return set()

    text = text.lower()

    return set(re.findall(r"[a-z0-9+#.]+", text))


def extract_skills(text, skill_dictionary):
    """
    Fast skill extraction using token lookup.
    """

    tokens = tokenize(text)

    found_skills = set()

    for skill in skill_dictionary:

        skill_tokens = set(skill.split())

        if skill_tokens.issubset(tokens):
            found_skills.add(skill)

    return found_skills


def calculate_skill_match(resume_text, job_text, skill_dictionary):
    """
    Calculate Skill Match Percentage.
    """

    resume_skills = extract_skills(resume_text, skill_dictionary)
    job_skills = extract_skills(job_text, skill_dictionary)

    if not job_skills:
        return 0.0

    matched = resume_skills.intersection(job_skills)

    return round(
        len(matched) / len(job_skills) * 100,
        2,
    )