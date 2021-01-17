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
questions = [json.loads(q) for q in questions]

# mapping from question index to question id
idx2id = {idx: question["id"] for idx, question in enumerate(questions)}
corpus = [item["question"] for item in questions]

engine = QuestionSearchEngine(corpus)
query = "What is angular?"
print("Question: ", query)
for similarity, uid, text in engine.most_similar(query):
    print(f"({similarity}, {idx2id[uid]}, {text})")