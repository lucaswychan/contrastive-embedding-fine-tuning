import json
import logging

import torch
from sentence_transformers import SentenceTransformer

from utils import get_available_gpu_idx, get_emb_sim

if __name__ == "__main__":
    available_gpu_idx = get_available_gpu_idx()
    if available_gpu_idx is None:
        raise ValueError("No available GPU found!")

    available_cuda = f"cuda:{available_gpu_idx}"
    print(f"Using GPU: {available_cuda}")

    device = torch.device(available_cuda)
    model = SentenceTransformer(
        "output/training/Dec-12_16-06_no_labels/checkpoint-51", device=device
    )

    with open("data/filtered_keywords.json", "r") as f:
        data = json.load(f)

    source = "Individual"

    f = open("data/individual_no_label_with_rolekg.txt", "w")

    for context in data[source]:
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
