import copy
import json
import logging

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from get_emb_sim import get_emb_sim
from llama import Llama3P2
from utils import convert_json, get_available_gpu_idx

logging.basicConfig(
    filename="logs/data_augmentation.log", filemode="w", level=logging.INFO
)


def augment_data(llm, definitions):
    """
    augment the data by leveraging LLM from the definiton dictionary
    """

    sys_prompt = "You are a legal expert that understand and analyzes legal definitions very well."

    cot_user_prompt = """
    You will be given a legal terminology with its corresponding definitions. Your task is to provide multiple daily context key words that are related to the given terminology. Daily context key words are the words that are used in the daily context
    
    Let's complete it step by step:
    1. Understand the definition: Read carefully the definition of the given terminology.
    2. Check for key points: Focus on important details of the terminology like the kind of information involved, who is handling it, and how it's being shared or used.
    3. Think clearly about the daily context: Think about the daily context in which the given terminology is being used.
    4. Identify the key words: Identify the key words that are related to the given terminology.
    5. Revisit the key words: prevent the duplicate key words and revisit the daily context key words.
    
    Terminology:
    {term}
    
    Definition:
    {definition}
    
    Your output must be in JSON format, with an outer keys of "context", followed by a list of no more than 30 number of daily context key words.
    """

    term_map = {}

    for term, definition in definitions.items():
        target_prompt = cot_user_prompt.format(term=term, definition=definition)

        try:
            response = llm(sys_prompt, target_prompt)
            logging.info(response)
        except Exception as e:
            logging.info(e)
            continue

        res_dict = convert_json(response)

        term_map[term] = list(set(res_dict["context"]))

        print("finished: ", term)

    return term_map


def filter_data(model, term_map, threshold=0.6):
    """
    filter the data by removing the keywords that are similar to the terminology
    """
    filtered_term_map = {}

    for term, keywords in term_map.items():
        keywords = set(keywords)

        for word in copy.deepcopy(keywords):
            sim = get_emb_sim(model, term, word)

            if sim > threshold:
                keywords.remove(word)

        filtered_term_map[term] = list(keywords)

    return filtered_term_map


def to_csv(term_map, file_name: str = "data/keywords.csv"):
    """
    Turn the filtered data into a csv file, for later training
    """
    labels_map = {term: i for i, term in enumerate(term_map.keys())}
    dataset_dict = {"query": [], "positive": [], "label": []}
    for term, keywords in term_map.items():
        for word in keywords:
            dataset_dict["query"].append(term)
            dataset_dict["positive"].append(word)
            dataset_dict["label"].append(labels_map[term])

    df = pd.DataFrame(dataset_dict)
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # Augment the data
    # llm = Llama3P2()

    # with open("checklist_data/HIPAA//HIPAA_defs.json") as f:
    #     HIPAA_DEF = json.load(f)

    # term_map = augment_data(llm, HIPAA_DEF)

    # with open("data/keywords.json", "w") as f:
    #     json.dump(term_map, f, indent=4)

    ##############################################################################################################

    # Filter the data
    # with open("data/keywords.json", "r") as f:
    #     term_map = json.load(f)

    # available_gpu_idx = get_available_gpu_idx()
    # if available_gpu_idx is None:
    #     raise ValueError("No available GPU found!")

    # available_cuda = f"cuda:{available_gpu_idx}"
    # print(f"Using GPU: {available_cuda}")

    # device = torch.device(available_cuda)
    # model = SentenceTransformer(
    #     "sentence-transformers/all-mpnet-base-v2", device=device
    # )

    # filtered_term_map = filter_data(model, term_map)

    # with open("data/filtered_keywords.json", "w") as f:
    #     json.dump(filtered_term_map, f, indent=4)

    ##############################################################################################################

    # To csv

    with open("data/filtered_keywords.json", "r") as f:
        term_map = json.load(f)

    to_csv(term_map)
