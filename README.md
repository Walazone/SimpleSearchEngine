# Simple Search Engine

Recommended python version 3.6.8

To run tests
```
python -m pytest
```

To use TfIdfVectorizer
```
from nlp.vectorizers import TfIdfVectorizer

questions = [
    "this is the question",
    "what is the answer to life, the universe and everything"
]
vectorizer = TfIdfVectorizer()
vectorizer.fit(questions)

query_vector = vectorizer.transform(["what is life"])
print(query_vector)
```

To use SearchEngine
```
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third document.",
    "Is this the first document?",
]
engine = QuestionSearchEngine(corpus)

query = "Question regarding third document?"
print(engine.most_similar(query))
```

