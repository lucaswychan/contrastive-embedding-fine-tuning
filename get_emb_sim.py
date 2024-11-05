import json
import logging

import torch
from sentence_transformers import SentenceTransformer

from utils import get_available_gpu_idx


def get_emb_sim(model, source, target):
    source_emb = model.encode(source, convert_to_tensor=True)
    target_emb = model.encode(target, convert_to_tensor=True)

    sim = torch.cosine_similarity(source_emb, target_emb, dim=-1).item()

    return sim


if __name__ == "__main__":
    available_gpu_idx = get_available_gpu_idx()
    if available_gpu_idx is None:
        raise ValueError("No available GPU found!")

    available_cuda = f"cuda:{available_gpu_idx}"
    print(f"Using GPU: {available_cuda}")

    device = torch.device(available_cuda)
    model = SentenceTransformer(
        'output/training/Nov-04_00-53/checkpoint-202', device=device
    )

    with open("data/filtered_keywords.json", "r") as f:
        data = json.load(f)

    source = "Covered entity"

    f = open("data/covered_entity.txt", "w")

    key_word = "Covered entity"

    for context in data[key_word]:
        target = context
        sim = get_emb_sim(model, source, target)

        f.write(f"{source}\t{target}  ---  {sim}\n")

    f.close()

    # sim = get_emb_sim("HIPAA", "medical record")
    """
    # HIPAA vs medical record = 0.47049009799957275
    # HIPAA vs privacy regulation = 0.39
    """

    # sim = get_emb_sim("HIPAA", "privacy regulation")

    # print(sim)
