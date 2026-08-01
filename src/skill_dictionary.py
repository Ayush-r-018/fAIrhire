"""
FairHire
Skill Dictionary Module
"""

import re


def normalize_skill(skill: str) -> str:
    """
    Normalize skills so that resume skills and
    job skills use the same format.
    """

    if not isinstance(skill, str):
        return ""

    skill = skill.lower().strip()

    # Technology normalization
    replacements = {
        "c++": "cplusplus",
        "c#": "csharp",
        ".net": "dotnet",
        "asp.net": "aspnet",
        "vb.net": "vbnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "next.js": "nextjs",
        "vue.js": "vuejs",
        "express.js": "expressjs",
        "sql server": "sqlserver",
        "machine learning": "machinelearning",
        "deep learning": "deeplearning",
        "computer vision": "computervision",
        "natural language processing": "nlp",
        "power bi": "powerbi",
    }

    for old, new in replacements.items():
        skill = skill.replace(old, new)

    # Remove proficiency words
    remove_words = [
        "basic",
        "basics",
        "fundamental",
        "fundamentals",
        "advanced",
        "expert",
        "experienced",
        "professional",
        "proficient",
        "strong",
        "excellent",
        "good",
        "knowledge of",
        "understanding of",
    ]

    for word in remove_words:
        skill = skill.replace(word, "")

    # Remove punctuation except space
    skill = re.sub(r"[^a-z0-9\s]", " ", skill)

    # Remove extra spaces
    skill = re.sub(r"\s+", " ", skill).strip()

    return skill


def build_skill_dictionary(job_df):
    """
    Build a unique skill dictionary from the job dataset.
    """

    skill_set = set()

    for skills in job_df["Skills"].fillna(""):

        # Split using ; or ,
        for skill in re.split(r"[;,]", skills):

            skill = normalize_skill(skill)

            if len(skill) > 1:
                skill_set.add(skill)

    return skill_set