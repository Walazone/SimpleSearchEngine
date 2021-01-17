from typing import List, Tuple, Sequence
import numpy as np
from numpy import dot
from numpy.linalg import norm

from .vectorizers import TfIdfVectorizer


def cosine_similarity(
    query_vector: np.ndarray, corpus_vectors: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between query vector and corpus vectors

    Args:
        query_vector: query vector of dimensinality (1, D)
        corpus_vector: corpus matrix of dimensionality (N, D)
    Returns:
        The vector of (1, N) shape with values in range [-1, 1] representing similarity
        between a query and a given vector where 1 is max similarity i.e. two vectors are the same.
    """
    q, corpus = query_vector, corpus_vectors

    assert q.shape[0] == 1
    assert q.shape[1] == corpus_vectors.shape[1]
    # ignore division with 0 warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (q @ corpus.T) / (norm(q) * norm(corpus, axis=1))

    # replace nan values (result of divison with 0) with zeros
    result = np.nan_to_num(result)
    return result


class QuestionSearchEngine:
    """Initialize search engine by vectorizing question corpus.

    Input questions are used to fit the TF-IDF vectorizer.
    Vectorized question is used to find the top n most
    similar questions w.r.t. input query.

    Args:
        questions: The sequence of raw questions from corpus.
    """

    def __init__(self, questions: Sequence[str]) -> None:
        self.vectorizer = TfIdfVectorizer()
        self.questions = questions
        self.X = self.vectorizer.fit_transform(questions)

    def most_similar(self, query: str, n: int = 5) -> List[Tuple[float, int, str]]:
        """Return top n most similar questions from corpus.

        Input question are cleaned and vectorized with fitted
        TfIdfVectorizer to get query question vectors. After that, use
        cosine_similarity function to get the top n most similar
        questions from the corpus.

        Args:
            query: The raw query question input from the user.
            n: The number of similar questions returned from corpus.

        Returns:
            The list of top n most similar questions from corpus along
            with similarity scores. Note that returned questions are
            verbatim.
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.X).reshape(-1)
        indices = np.argsort(similarities)[::-1][:n]
        return [(similarities[idx], idx, self.questions[idx]) for idx in indices]
