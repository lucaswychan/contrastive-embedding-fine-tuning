from llama3p2 import Llama3P2
from utils import convert_json
import logging
import json

logging.basicConfig(filename="log/hard_neg.log", filemode="w", level=logging.INFO)


def generate_hard_neg(keywords):
    llm = Llama3P2()

    role = ""


if __name__ == "__main__":
    with open("data/keywords/json", "r") as f:
        data = json.load(f)

    term = "Covered entity"

    # keywords = data[term]
    keywords = ["doctor"]

    hard_negatives = generate_hard_neg(keywords)

    with open("data/hard_neg.json", "w") as f:
        json.dump(hard_negatives, f, indent=4)
