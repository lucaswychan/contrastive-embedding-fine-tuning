import logging
import os
import sys
from collections import deque

sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

import argparse

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from pooler import pool_embeddings
from tqdm import tqdm

from config import HF_cases_path, HF_KBs_path
from sentence_embedding.emb_model_factory import SentenceEmbeddingModelFactory
from utils import str2bool

# the logger configuration is in the main function
logger = logging.getLogger("sentence_embedding")


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
    embeddings = F.normalize(embeddings, p=2, dim=-1).cpu().detach()

    return embeddings


def get_embeddings_once(emb_model, contents: str):
    embeddings = emb_model.encode(pa.array(contents).tolist())
    embeddings = F.normalize(embeddings, p=2, dim=-1).cpu().detach()

    print(embeddings.shape)

    return embeddings


def direct_search(args, emb_model) -> np.ndarray:
    # load the table that have splitted sentences
    case_table = pq.read_table(
        f"data/{args.dataset_name}/splitted_{args.dataset_name.lower()}_cases.parquet"
    )
    kb_table = pq.read_table(
        f"data/{args.dataset_name}/splitted_{args.dataset_name.lower()}_kb.parquet"
    )

    # get the splitted sentences
    case_sentences = case_table["case_content_sentences"]
    kb_sentences = kb_table["regulation_content_sentences"]

    case_embeddings = None
    kb_embeddings = None

    if args.encode_all_once:
        case_embeddings = get_embeddings_once(
            emb_model, case_table["case_content"]
        )  # instead of using sentences, we use the whole case content
        kb_embeddings = get_embeddings_once(emb_model, kb_table["regulation_content"])
    else:
        case_embeddings = get_embeddings(emb_model, case_sentences)
        kb_embeddings = get_embeddings(emb_model, kb_sentences)

    print(f"case emb dim = {case_embeddings.shape}")
    print(f"kb emb dim = {kb_embeddings.shape}")

    np.save(
        f"data/{args.dataset_name}/case_embeddings_{args.dataset_name.lower()}_{path_suffix}.npy",
        case_embeddings,
    )
    np.save(
        f"data/{args.dataset_name}/kb_embeddings_{args.dataset_name.lower()}_{path_suffix}.npy",
        kb_embeddings,
    )

    similarity_scores = case_embeddings @ kb_embeddings.T

    np.save(
        f"data/{args.dataset_name}/similarity_{args.dataset_name.lower()}_{path_suffix}.npy",
        similarity_scores,
    )

    return similarity_scores


def hierarchical_search(
    emb_model, kb_graph, case_content, k=3
) -> list[tuple[float, str]]:
    # we will add node only if it is a leaf node, which have to outgoing edge with relation "subsume"
    def is_leaf(node):
        out_edges = kb_graph.out_edges(node)
        return all(
            kb_graph[source][neighbor].get("relation") != "subsume"
            for source, neighbor in out_edges
        )

    retrieved_regulations = []

    case_embedding = emb_model.encode(case_content)
    case_embedding = F.normalize(case_embedding, p=2, dim=-1).detach()

    root_node = "HIPAA"

    queue = deque([root_node])
    visited = set()

    # start BFS
    while queue:
        num_node = len(queue)

        similarity_scores = []

        # perform BFS by layer
        for _ in range(num_node):
            node = queue.popleft()
            visited.add(node)

            for source, neighbor in kb_graph.out_edges(node):
                if (
                    neighbor in visited
                    or kb_graph[source][neighbor].get("relation") != "subsume"
                ):
                    continue

                neighbor_content = kb_graph.nodes[neighbor].get("text", None)
                if neighbor_content is None:
                    continue

                neighbor_content_embedding = emb_model.encode(neighbor_content)
                neighbor_content_embedding = F.normalize(
                    neighbor_content_embedding, p=2, dim=-1
                ).detach()
                score = torch.cosine_similarity(
                    case_embedding, neighbor_content_embedding, dim=-1
                ).item()
                similarity_scores.append((score, neighbor))

        similarity_scores.sort(reverse=True)

        # each time we retrieve k regulations
        for i in range(min(k, len(similarity_scores))):
            if is_leaf(similarity_scores[i][1]):
                retrieved_regulations.append(similarity_scores[i])
            else:
                # only add the node to the queue if it is not a leaf and it is in the top k results
                queue.append(similarity_scores[i][1])

    # it stores tuple of (score, regulation)
    return sorted(retrieved_regulations, reverse=True)[:k]


def evaluate_each_case_recall(retrieved_regulations, ground_truths):
    recall = 0.0
    for gt in ground_truths:
        for reg in retrieved_regulations:
            if gt in reg:
                recall += 1
                break

    recall /= len(ground_truths)

    return recall


def direct_evaluate(args, emb_model):
    similarity_scores = direct_search(args, emb_model)

    print(np.argsort(similarity_scores, axis=-1))

    # find the top-k regulations for each case
    top_k_similarity_idxs = np.argsort(similarity_scores, axis=-1)[
        :, -args.top_k :
    ]  # top-k regulations
    top_k_similarity_idxs = torch.flip(
        top_k_similarity_idxs, dims=(1,)
    )  # reverse the order, so the largest confidence is first
    print(top_k_similarity_idxs.shape)

    print(top_k_similarity_idxs)

    KB = load_from_disk(HF_KBs_path)
    CASES = load_from_disk(HF_cases_path)

    kb = KB[args.dataset_name]
    cases = CASES[args.dataset_name]

    regulation_ids = np.array(kb["regulation_id"])

    recall_list = []

    for i, k_idxs in enumerate(top_k_similarity_idxs):
        top_k_regulations = regulation_ids[k_idxs]

        ground_truths = cases["followed_articles"][i] + cases["violated_articles"][i]

        if not ground_truths:
            continue

        logger.info(f"similarity scores : {similarity_scores[i, k_idxs]}")
        print(f"similarity scores : {similarity_scores[i, k_idxs]}")

        recall = evaluate_each_case_recall(top_k_regulations, ground_truths)

        recall_list.append(recall)

        print(f"recall for case {i + 1} = {recall}")

        logger.info(cases["purpose"][i])
        logger.info(f"top k regulations for case {i + 1} = {top_k_regulations}")
        logger.info(f"ground truth for case {i + 1} = {ground_truths}")
        logger.info(f"recall for case {i + 1} = {recall}")
        logger.info("=" * 50)
        logger.info("\n")

    logger.info(f"average recall = {np.mean(recall_list)}")


def hierarchical_evaluate(args, emb_model):
    KB = load_from_disk(HF_KBs_path)
    CASES = load_from_disk(HF_cases_path)

    kb = KB[args.dataset_name]
    cases = CASES[args.dataset_name]

    kb_graph = nx.read_graphml(
        f"checklist_data/{args.dataset_name}/{args.dataset_name}.graphml"
    )

    recall_list = []

    for i, case in enumerate(cases):
        case_content = case["case_content"]
        ground_truths = case["followed_articles"] + case["violated_articles"]

        if not ground_truths:
            continue

        retrieved_regulations = hierarchical_search(
            emb_model, kb_graph, case_content, k=args.top_k
        )

        confidences = [score for score, _ in retrieved_regulations]
        regulations = [reg for _, reg in retrieved_regulations]

        print(f"similarity scores : {confidences}")
        logger.info(f"similarity scores : {confidences}")

        recall = evaluate_each_case_recall(regulations, ground_truths)

        recall_list.append(recall)

        print(f"recall for case {i + 1} = {recall}")

        logger.info(cases["purpose"][i])
        logger.info(f"top k regulations for case {i + 1} = {regulations}")
        logger.info(f"ground truth for case {i + 1} = {ground_truths}")
        logger.info(f"recall for case {i + 1} = {recall}")
        logger.info("=" * 50)
        logger.info("\n")

    logger.info(f"average recall = {np.mean(recall_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--encode_all_once",
        type=bool,
        default=True,
        help="encode each regulation directly without sentence splitting",
    )
    parser.add_argument(
        "--hierarchical_search",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to use hierarchical search using graph",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="hipaa")
    args = parser.parse_args()

    args.dataset_name = args.dataset_name.upper()
    emb_model = SentenceEmbeddingModelFactory.get_model("hf", model=args.model_path)

    path_suffix = "hierarchical" if args.hierarchical_search else "direct"

    if (
        args.encode_all_once and not args.hierarchical_search
    ):  # only encode once when in direct search
        path_suffix += "_once"
    if "checkpoint" in args.model_path:
        path_suffix += "_mymodel"
    if "tsdae" in args.model_path:
        path_suffix += "_tsdae"

    path_suffix += f"_{args.top_k}"

    args.path_suffix = path_suffix

    print(
        f"path to save = logs/sentence_embedding/{args.dataset_name}/regulation_rag_acc_{args.dataset_name.lower()}_{path_suffix}.log"
    )

    logging.basicConfig(
        filename=f"logs/sentence_embedding/{args.dataset_name}/regulation_rag_acc_{args.dataset_name.lower()}_{path_suffix}.log",
        filemode="w",
        level=logging.INFO,
    )

    if args.hierarchical_search:
        print(f"Starting hierarchical evaluation for {args.dataset_name}")
        hierarchical_evaluate(args, emb_model)
    else:
        print(f"Starting direct evaluation for {args.dataset_name}")
        direct_evaluate(args, emb_model)
