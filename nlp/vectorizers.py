import re
import pickle
from typing import Sequence, List
from collections import defaultdict, Counter
from operator import itemgetter

import numpy as np


class VectorizerMixin:
    """Provides common code for text vectorizers (preprocessing logic)"""

    def _preprocess(self, document: str) -> List[str]:
        """Prepares textual document for vectorization.

        Preprocessing steps - lowercase and tokenization

        Args:
            document: textual document to be processed

        Returns:
            list of tokens, all lowercased
        """
        document = document.lower()
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        document = token_pattern.findall(document)
        return document


class CountVectorizer(VectorizerMixin):
    """Converts a collection of textual documents to a matrix of token counts.

    Args:
        max_features: maximum number of tokens used to represent each document. Single document
        representation will be of length min(n_tokens, max_features)
    """

    def __init__(self, max_features: int = 1000) -> None:
        self.max_features = max_features

    def fit(self, documents: Sequence[str]) -> None:
        """Fit vectorizer with the sequence of documents.

        Args:
            documents: sequence of textual documents used for fitting the vectorizer
        """
        self.vocabulary = self._build_vocabulary(documents, self.max_features)

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transforms documents to numerical representation.

        Each document is represented as a numpy vector, that counts occurences of each word
        in a corpus vocabulary. Length of each document representation is same as the length of
        the vocabulary.

        Args:
            documents: sequence of textual documents to be transformed
        Returns:
            matrix where each row is a bag of words representation of a input document
        """
        X = []
        for document in documents:
            document_representation = [0] * len(self.vocabulary)
            for token in self._preprocess(document):
                if token in self.vocabulary:
                    token_idx = self.vocabulary[token]
                    document_representation[token_idx] += 1

            X.append(document_representation)
        return np.array(X)

    def fit_transform(self, documents: Sequence[str]):
        """Fitting vectorizer with a sequence of documents and returning document-term matrix

        Equivalent of fit followed by transform.
        """
        self.fit(documents)
        X = self.transform(documents)
        return X

    def get_feature_names(self):
        """Gets a list of vocabulary tokens"""
        return [
            token for token, idx in sorted(self.vocabulary.items(), key=itemgetter(1))
        ]

    def _build_vocabulary(self, documents: Sequence[str], max_features: int) -> dict:
        """Builds a vocabulary from a sequence of documents.

        Vocabulary is a dictionary mapping tokens to their respective positions.
        Size of vocabulary is limited with a max_features. If max_features is greater than
        total number of tokens, only max_features most frequent tokens in a corpus are selected.

        Args:
            documents: sequence of documents which will be used for vocabulary creation
            max_features: maximum possible number of tokens in a vocabulary
        Returns:
            dictionary mapping tokens to their indices
        """
        feature_counter = Counter()

        for document in documents:
            feature_counter.update(self._preprocess(document))

        features = [el[0] for el in feature_counter.most_common(max_features)]
        vocabulary = {feature: idx for idx, feature in enumerate(features)}

        if not vocabulary:
            raise ValueError("empty vocabulary")

        return vocabulary


class TfIdfVectorizer(CountVectorizer):
    """Converts a collection of textual documents to a matrix of tfidf values.

    tfidf value is calculated as a combination of tf and idf values

    tf stands for term frequency and it counts number of occurences of a
    token in a documents with respect to total token number in a document.
    tf = (number of times token appears in a document) / (total number of tokens in a document)

    idf stands for inverse document frequency, and it counts inverse of a total
    number of documents token appears in, with respect to total number of documents in
    a corpus.
    idf = log((total number of documents) / (number of documents that contain token + 1) ) +1


    Args:
        max_features: maximum number of tokens used to represent each document. Single document
        representation will be of length min(n_tokens, max_features)
    """

    def __init__(self, max_features=1000):
        super().__init__(max_features)

    def fit(self, documents: Sequence[str]) -> None:
        """Fit vectorizer with the sequence of documents.

        Vectorizer stores document-term matrix and idf vector as its state.

        Args:
            documents: sequence of textual documents used for fitting the vectorizer
        """
        super().fit(documents)
        X = super().transform(documents)
        self.idf = self._idf(X)

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Transforms sequence of documents to a tfidf matrix representation.

        Each document is represented as a tfidf numpy vector.
        Length of each document representation is same as the length of
        the vocabulary.

        Args:
            documents: sequence of textual documents to be transformed
        Returns:
            matrix where each row is a tfidf representation of a input document
        """
        X = super().transform(documents)
        tf = self._tf(X)

        return tf * self.idf

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fitting vectorizer with a sequence of documents and returning tfidf matrix

        Equivalent of fit followed by transform.
        """
        self.fit(documents)
        return self.transform(documents)

    def _idf(self, X: np.ndarray) -> np.ndarray:
        """Calculates inverse document frequency vector given a document-term matrix

        Args:
            X: document-term matrix of shape (D,N) where D is number of documents, and N number of tokens
        Returns:
            idf vector of shape (N,) representing idf values for all vocabulary terms
        """
        df = np.count_nonzero(X, axis=0)
        n_doc = X.shape[0]
        return np.log(n_doc / (df + 1)) + 1

    def _tf(self, X: np.ndarray) -> np.ndarray:
        """Calculates term frequency matrix given a document-term matrix

        Args:
            X: document-term matrix of shape (D,N) where D is number of documents, and N number of tokens
        Returns:
            term frequency matrix of shape (D, N) representing count of tokens in a document with respect
            to total number of tokens in a document
        """
        feature_counts = np.sum(X, axis=1).reshape(-1, 1)
        # replacing zeros with ones in order to avoid zero divison error
        feature_counts = np.where(feature_counts == 0, 1, feature_counts)

        return X / feature_counts

    def save(self, filename: str) -> None:
        """Saves current state of a vectorizer in a file.

        Args:
            filename: name of the output file
        """
        with open(filename, "wb") as f:
            pickle.dump({"vocabulary": self.vocabulary, "idf": self.idf}, f)

    @classmethod
    def load(self, filename: str):
        """Loads vectorizer state from a file.

        Args:
            filename: name of the output file
        Returns:
            TfIdfVectorizer instance loaded with saved state.
        """
        with open(filename, "rb") as f:
            saved_state = pickle.load(f)
        if type(saved_state) != dict:
            raise ValueError("expected dictionary")

        vectorizer = TfIdfVectorizer()
        vectorizer.vocabulary = saved_state["vocabulary"]
        vectorizer.idf = saved_state["idf"]
        return vectorizer
