import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean textual data while preserving important technical skills.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove Emails
    4. Remove Phone Numbers
    5. Preserve technical terms
    6. Remove unnecessary punctuation
    7. Remove extra whitespace
    8. Stopword removal
    9. Lemmatization
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove Phone Numbers
    text = re.sub(r"\b\d{10,}\b", "", text)

    # Preserve common technical terms
    replacements = {
        "c++": "cplusplus",
        "c#": "csharp",
        ".net": "dotnet",
        "asp.net": "aspnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "next.js": "nextjs",
        "vue.js": "vuejs",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation but keep + # .
    text = re.sub(r"[^\w\s+#.]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)