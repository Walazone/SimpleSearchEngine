from nlp.similarity_search import QuestionSearchEngine

import json

if __name__ == "__main__":

    # load questions from file
    with open("questions.jsonl", "r") as f:
        questions = f.readlines()
    questions = [json.loads(q) for q in questions]

    # mapping from question index to question id
    idx2id = {idx: question["id"] for idx, question in enumerate(questions)}
    corpus = [item["question"] for item in questions]

    # build search engine on a given corpus
    engine = QuestionSearchEngine(corpus)
    print("Search Engine loaded!")

    query = input("Input query question: ")

    for similarity, uid, text in engine.most_similar(query):
        print(f"({similarity}, {idx2id[uid]}, {text})")