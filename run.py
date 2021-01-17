from nlp.similarity_search import QuestionSearchEngine

import json
from pprint import pprint


corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third document.",
    "Is this the first document?",
]

query = "Question regarding third document?"
engine = QuestionSearchEngine(corpus)
print("Question: ", query)
pprint(engine.most_similar(query))
print("=================================================================")

with open("questions.jsonl", "r") as f:
    questions = f.readlines()
questions = [json.loads(q)["question"] for q in questions]

engine = QuestionSearchEngine(questions)
query = "what is angular"
print("Question: ", query)
pprint(engine.most_similar(query))