import argparse
import json
import re

import nvidia_smi
import torch
from sentence_transformers import SentenceTransformer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def print_json_error(text: str, error_msg: str) -> None:
    SEPARATE_LINE = "=" * 30
    print(f"{SEPARATE_LINE}\n{error_msg}\n{text}\n{SEPARATE_LINE}")


def convert_json(text: str) -> dict:
    # Check if the text is empty
    if not text:
        print_json_error(text, "Error: Empty text")
        return {"Empty Error": None}

    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    json_str = json_match.group(1) if json_match else text

    # preprocess the text to make it a valid JSON
    text = json_str.strip().replace("\n", "")

    open_bracket_idx = [i for i, char in enumerate(text) if char == "{"]
    close_bracket_idx = [i for i, char in enumerate(text) if char == "}"]

    text = text[open_bracket_idx[0] : close_bracket_idx[-1] + 1]

    text = text.replace(",]", "]").replace("`", "").replace('""', '"')

    # Load the JSON string into a dictionary
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print_json_error(text, f"JSONDecodeError: {e}")
        data = {"Decode Error": None}
    except Exception as e:
        print_json_error(text, f"Exception: {e}")
        data = {"Exception Error": None}

    return data


def get_available_gpu(use_cpu=False):
    """
    Get the GPU index which have more than 90% free memory
    """
    # Initialize NVIDIA-SMI
    nvidia_smi.nvmlInit()

    # Get the number of available GPUs
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # Check if the GPU has enough free memory (e.g., 90% free)
        if info.free / info.total > 0.9:
            # Try to allocate a small tensor on this GPU
            try:
                with torch.cuda.device(f"cuda:{i}"):
                    torch.cuda.current_stream().synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.memory.empty_cache()
                    test_tensor = torch.zeros((1,), device=f"cuda:{i}")
                    del test_tensor
                    return f"cuda:{i}"
            except RuntimeError:
                # If allocation fails, move to the next GPU
                continue

    if not use_cpu:
        raise ValueError("No available GPU found!")

    # If no available GPU is found
    return "cpu"


def get_emb_sim(model, source, target):
    source_emb = model.encode(source, convert_to_tensor=True)
    target_emb = model.encode(target, convert_to_tensor=True)

    sim = torch.cosine_similarity(source_emb, target_emb, dim=-1).item()

    return sim


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


# def check_kb_corr():
#     kb_embeddings = np.load(
#         f"data/{DATASET_NAME}/kb_embeddings_{DATASET_NAME.lower()}_{path_suffix}.npy"
#     )

#     correlations = np.corrcoef(kb_embeddings)
#     print(correlations)

#     np.save(
#         f"data/{DATASET_NAME}/kb_correlations_{DATASET_NAME.lower()}_{path_suffix}.npy",
#         correlations,
#     )
