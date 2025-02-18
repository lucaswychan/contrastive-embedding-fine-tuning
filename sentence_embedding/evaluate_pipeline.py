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

from config import HF_cases_path, HF_KBs_path, DATASET_NAME

use_my_model = False
encode_all_once = True
use_tsdae = False

path_suffix = ""
if encode_all_once:
    path_suffix += "_once"
if use_my_model:
    path_suffix += "_mymodel"
if use_tsdae:
    path_suffix += "_tsdae"

if path_suffix != "":
    path_suffix = path_suffix[1:]

logger = logging.getLogger("sentence_embedding")
logging.basicConfig(
    filename=f"logs/sentence_embedding/{DATASET_NAME}/regulation_rag_acc_{DATASET_NAME.lower()}_{path_suffix}.log",
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
    case_table = pq.read_table(f"data/{DATASET_NAME}/splitted_{DATASET_NAME.lower()}_cases.parquet")
    kb_table = pq.read_table(f"data/{DATASET_NAME}/splitted_{DATASET_NAME.lower()}_kb.parquet")

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
    
    case_embeddings = None
    kb_embeddings = None
    
    if encode_all_once:
        case_embeddings = get_embeddings_once(emb_model, case_table["case_content"]) # instead of using sentences, we use the whole case content
        kb_embeddings = get_embeddings_once(emb_model, kb_table["regulation_content"])
    else:
        case_embeddings = get_embeddings(emb_model, case_sentences)
        kb_embeddings = get_embeddings(emb_model, kb_sentences)

    print(f"case emb dim = {case_embeddings.shape}")
    print(f"kb emb dim = {kb_embeddings.shape}")

    print(f"case_embedding norm = {torch.norm(case_embeddings, dim=1)}")
    print(f"kb_embedding norm = {torch.norm(kb_embeddings, dim=1)}")

    np.save(f"data/{DATASET_NAME}/case_embeddings_{DATASET_NAME.lower()}_{path_suffix}.npy", case_embeddings)
    np.save(f"data/{DATASET_NAME}/kb_embeddings_{DATASET_NAME.lower()}_{path_suffix}.npy", kb_embeddings)

    similarity = case_embeddings @ kb_embeddings.T

    np.save(f"data/{DATASET_NAME}/similarity_{DATASET_NAME.lower()}_{path_suffix}.npy", similarity)


def evaluate():
    similarity = np.load(f"data/{DATASET_NAME}/similarity_{DATASET_NAME.lower()}_{path_suffix}.npy")

    # find the top-k regulations for each case
    k = 5
    top_k = np.argsort(similarity, axis=1)[:, -k:][:, ::-1] # top-k regulations

    print(top_k)

    KB = load_from_disk(HF_KBs_path)
    CASES = load_from_disk(HF_cases_path)

    kb = KB[DATASET_NAME]
    cases = CASES[DATASET_NAME]

    regulation_ids = np.array(kb["regulation_id"])

    recall_list = []

    for i, k_idxs in enumerate(top_k):
        top_k_regulations = regulation_ids[k_idxs]

        logger.info(f"similarity scores : {similarity[i, k_idxs]}")
        print(f"similarity scores : {similarity[i, k_idxs]}")

        ground_truth = (
            cases["followed_articles"][i] + cases["violated_articles"][i]
        )
        
        if not ground_truth:
            continue

        recall = 0.0
        for gt in ground_truth:
            for reg in top_k_regulations:
                if gt in reg:
                    recall += 1
                    break

        recall /= len(ground_truth)
        recall_list.append(recall)

        print(f"recall for case {i + 1} = {recall}")

        logger.info(cases["purpose"][i])
        logger.info(f"top k regulations for case {i + 1} = {top_k_regulations}")
        logger.info(f"ground truth for case {i + 1} = {ground_truth}")
        logger.info(f"recall for case {i + 1} = {recall}")
        logger.info("=" * 50)
        logger.info("\n")

    logger.info(f"average recall = {np.mean(recall_list)}")

def full_evaluation_pipeline():
    get_similarity()
    evaluate()


def check_kb_corr():
    kb_embeddings = np.load(f"data/{DATASET_NAME}/kb_embeddings_{DATASET_NAME.lower()}_{path_suffix}.npy")

    correlations = np.corrcoef(kb_embeddings)
    print(correlations)

    np.save(f"data/{DATASET_NAME}/kb_correlations_{DATASET_NAME.lower()}_{path_suffix}.npy", correlations)


if __name__ == "__main__":
    full_evaluation_pipeline()
