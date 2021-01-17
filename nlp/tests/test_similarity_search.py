from nlp.similarity_search import cosine_similarity, QuestionSearchEngine

import pytest
import numpy as np

corpus = [
    "I want to use stdin in a pytest test",
    "pytest run only the changed file?",
    "Can I perform multiple assertions in pytest?",
    "Invoke pytest from python for current module only",
    "How do I import the pytest monkeypatch plugin?",
    "Some random words",
]


def test_cosine_similarity():
    a = np.array([1, 0, 2, 3])
    B = np.array([[2, 0, 4, 6], [0, 0, 0, 0], [1, 0, 1, 1]])
    similarities = cosine_similarity(a, B)

    assert similarities[0] > similarities[2] > similarities[1]
    assert similarities[1] == 0
    assert similarities[0] == 1


def test_search_engine_init():
    engine = QuestionSearchEngine(corpus)
    assert len(engine.questions) == engine.X.shape[0] == 6


def test_empty_search_engine_init():
    with pytest.raises(ValueError, match="empty vocabulary"):
        QuestionSearchEngine([])


def test_search_engine():
    engine = QuestionSearchEngine(corpus)

    query1 = "In pytest Can I perform multiple assertions?"

    most_similar = engine.most_similar(query1, 10)
    assert most_similar[0] == (1, 2, "Can I perform multiple assertions in pytest?")
    assert most_similar[-1] == (0, 5, "Some random words")
    assert len(most_similar) == 6

    most_similar = engine.most_similar(query1, 3)
    assert most_similar[0] == (1, 2, "Can I perform multiple assertions in pytest?")
    assert len(most_similar) == 3
