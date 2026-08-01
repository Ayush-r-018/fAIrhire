"""
FairHire
Semantic Similarity Module
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_similarity(resume_text, job_text):
    """
    Calculate similarity for a single resume-job pair.
    """

    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    job_embedding = model.encode(
        job_text,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding],
    )[0][0]

    return round(float(similarity), 4)


def calculate_semantic_similarity_batch(
    resumes,
    jobs,
):
    """
    Calculate semantic similarity for an entire dataset.
    Much faster than calling the model row by row.
    """

    resume_embeddings = model.encode(
        resumes,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    job_embeddings = model.encode(
        jobs,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    similarities = []

    for resume_embedding, job_embedding in zip(
        resume_embeddings,
        job_embeddings,
    ):

        score = cosine_similarity(
            resume_embedding.reshape(1, -1),
            job_embedding.reshape(1, -1),
        )[0][0]

        similarities.append(round(float(score), 4))

    return similarities