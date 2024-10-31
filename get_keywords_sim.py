import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import torch


def get_keywords_sim(term1, term2):
    with open("data/keywords.json", "r") as f:
        data = json.load(f)

    emb_model = SentenceTransformer("all-mpnet-base-v2", device="cuda:1")

    keyword_emb1 = {}
    for keyword in data[term1]:
        keyword_emb1[keyword] = emb_model.encode(keyword, convert_to_tensor=True)

    keyword_emb2 = {}
    for keyword in data[term2]:
        keyword_emb2[keyword] = emb_model.encode(keyword, convert_to_tensor=True)

    corr = {}
    for keyword1 in keyword_emb1:
        corr[keyword1] = {}

        for keyword2 in keyword_emb2:
            corr[keyword1][keyword2] = torch.cosine_similarity(
                keyword_emb1[keyword1], keyword_emb2[keyword2], dim=-1
            ).item()

    return corr


if __name__ == "__main__":
    keywords_sim = get_keywords_sim("Covered entity", "Individual")

    with open("data/keywords_sim.json", "w") as f:
        json.dump(keywords_sim, f, indent=4)

    print(keywords_sim)
