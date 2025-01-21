import logging
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from model_factory import SentenceEmbeddingModelFactory
from pooler import pool_embeddings

from config import HF_cases_path, HF_KBs_path

use_my_model = False
encode_all_once = False
use_tsdae = True

path_suffix = ""
if encode_all_once:
    path_suffix += "_once"
if use_my_model:
    path_suffix += "_mymodel"
if use_tsdae:
    path_suffix += "_tsdae"

logger = logging.getLogger("sentence_embedding")
logging.basicConfig(
    filename=f"logs/sentence_embedding/regulation_rag_acc{path_suffix}.log",
    filemode="w",
    level=logging.INFO,
)


def get_embeddings(emb_model, sentences):
    embeddings = []

    # convert all cases content into embedding vectors. Each case correspond to one sentence vector
    for sents in sentences:
        sentences_array = pa.array(sents).tolist()

        # encode the sentences
        each_case_embeddings = emb_model.encode(sentences_array)

        # perform mean pooling to get a single embedding of a case
        each_case_embeddings_after_pooling = pool_embeddings(
            each_case_embeddings, pooling_type="mean"
        )

        embeddings.append(each_case_embeddings_after_pooling.cpu().detach().numpy())

    embeddings = torch.tensor(np.array(embeddings))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def get_embeddings_once(emb_model, contents: str):
    embeddings = emb_model.encode(pa.array(contents).tolist())
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach()

    print(embeddings.shape)

    return embeddings


def get_similarity():
    # load the table that have splitted sentences
    case_table = pq.read_table("data/splitted_hipaa_cases.parquet")
    kb_table = pq.read_table("data/splitted_hipaa_kb.parquet")

    # get the splitted sentences
    case_sentences = case_table["case_content_sentences"]
    kb_sentences = kb_table["regulation_content_sentences"]

    # load the model
    kwargs = {}
    if use_my_model:
        kwargs["model"] = "output/training/Dec-12_16-06_no_labels/checkpoint-51"
    elif use_tsdae:
        kwargs["model"] = "output/tsdae-model"
    emb_model = SentenceEmbeddingModelFactory.get_model("hf", **kwargs)

    case_embeddings = get_embeddings_once(emb_model, case_table["case_content"])
    kb_embeddings = get_embeddings_once(emb_model, kb_table["regulation_content"])

    print(f"case emb dim = {case_embeddings.shape}")
    print(f"kb emb dim = {kb_embeddings.shape}")

    print(f"case_embedding norm = {torch.norm(case_embeddings, dim=1)}")
    print(f"kb_embedding norm = {torch.norm(kb_embeddings, dim=1)}")

    np.save(f"data/case_embeddings{path_suffix}.npy", case_embeddings)
    np.save(f"data/kb_embeddings{path_suffix}.npy", kb_embeddings)

    similarity = case_embeddings @ kb_embeddings.T

    np.save(f"data/similarity{path_suffix}.npy", similarity)


def evaluate():
    similarity = np.load(f"data/similarity{path_suffix}.npy")

    # find the top-k regulations for each case
    k = 5
    top_k = np.argsort(similarity, axis=1)[:, -k:]

    print(top_k)

    KB = load_from_disk(HF_KBs_path)
    CASES = load_from_disk(HF_cases_path)

    hipaa_kb = KB["HIPAA"]
    hipaa_cases = CASES["HIPAA"]

    regulation_ids = np.array(hipaa_kb["regulation_id"])

    acc_list = []

    for i, k_idxs in enumerate(top_k):
        top_k_regulations = regulation_ids[k_idxs]

        logger.info(f"similarity scores : {similarity[i, k_idxs]}")
        print(f"similarity scores : {similarity[i, k_idxs]}")

        ground_truth = (
            hipaa_cases["followed_articles"][i] + hipaa_cases["violated_articles"][i]
        )
        
        if not ground_truth:
            continue

        acc = 0.0
        for gt in ground_truth:
            for reg in top_k_regulations:
                if gt in reg:
                    acc += 1
                    break

        acc /= len(ground_truth)
        acc_list.append(acc)

        print(f"acc for case {i + 1} = {acc}")

        logger.info(hipaa_cases["purpose"][i])
        logger.info(f"top k regulations for case {i + 1} = {top_k_regulations}")
        logger.info(f"ground truth for case {i + 1} = {ground_truth}")
        logger.info(f"acc for case {i + 1} = {acc}")
        logger.info("=" * 50)
        logger.info("\n")

    logger.info(f"average acc = {np.mean(acc_list)}")


def check_kb_corr():
    kb_embeddings = np.load(f"data/kb_embeddings{path_suffix}.npy")

    correlations = np.corrcoef(kb_embeddings)
    print(correlations)

    np.save(f"data/kb_correlations{path_suffix}.npy", correlations)


if __name__ == "__main__":
    get_similarity()
    evaluate()
