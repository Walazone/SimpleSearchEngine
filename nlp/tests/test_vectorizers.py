from nlp.vectorizers import VectorizerMixin, CountVectorizer, TfIdfVectorizer

import pytest

from numpy.testing import assert_array_equal

corpus = [
    "This is the first document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]


def test_preprocessing():
    vectorizer = VectorizerMixin()

    example1 = "This is the example\t with \n tabs, new lines\n\n"
    output1 = vectorizer._preprocess(example1)
    assert set(output1) == set(
        ["this", "is", "the", "example", "with", "tabs", "new", "lines"]
    )

    example2 = "// .! '!? "
    output2 = vectorizer._preprocess(example2)
    assert len(output2) == 0

    example3 = "EXAMPLE OF ALL CAPS"
    output3 = vectorizer._preprocess(example3)
    assert set(output3) == set(["example", "of", "all", "caps"])

    example4 = "example.with!punctuation?INSTEAD/of whitespaces"
    output4 = vectorizer._preprocess(example4)
    assert set(output4) == set(
        ["example", "with", "punctuation", "instead", "of", "whitespaces"]
    )


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfIdfVectorizer))
def test_vectorizers_vocabulary(Vectorizer):
    vectorizer = Vectorizer()
    vectorizer.fit(corpus)

    expected = [
        "this",
        "is",
        "the",
        "first",
        "document",
        "second",
        "and",
        "third",
        "one",
    ]
    assert set(vectorizer.get_feature_names()) == set(expected)


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfIdfVectorizer))
def test_maxfeature_vocabulary(Vectorizer):

    cv = Vectorizer(max_features=3)
    cv.fit(corpus)

    expected = ["this", "is", "the"]
    assert set(cv.get_feature_names()) == set(expected)


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfIdfVectorizer))
def test_empty_vocab(Vectorizer):
    vectorizer = Vectorizer()
    corpus = ["", ",,", ";"]

    with pytest.raises(ValueError, match="empty vocabulary"):
        vectorizer.fit(corpus)


@pytest.mark.parametrize("Vectorizer", (CountVectorizer, TfIdfVectorizer))
def test_fit_transform_twice(Vectorizer):
    vectorizer = Vectorizer()
    X1 = vectorizer.fit_transform(corpus[:2])
    assert X1.shape[0] == 2
    X2 = vectorizer.fit_transform(corpus[1:])
    assert X2.shape[0] == 3


def test_fit_fit_transform_same_state():

    cv1, cv2 = CountVectorizer(), CountVectorizer()
    cv1.fit(corpus)
    cv2.fit_transform(corpus)
    assert cv1.vocabulary == cv2.vocabulary

    tfidf1, tfidf2 = TfIdfVectorizer(), TfIdfVectorizer()
    tfidf1.fit(corpus)
    tfidf2.fit_transform(corpus)

    assert tfidf1.vocabulary == tfidf2.vocabulary
    assert_array_equal(tfidf1.idf, tfidf2.idf)
