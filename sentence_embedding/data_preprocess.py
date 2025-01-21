import ast
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_from_disk
from model_factory import SentenceEmbeddingModelFactory
from pooler import pool_embeddings
from sentence_splitter import SentenceSpliiter, SentenceSplitterConfig

from config import HF_cases_path, HF_KBs_path


def split_cases():
    dataset = load_from_disk(HF_cases_path)
    hipaa_cases = dataset["HIPAA"]

    sent_config = SentenceSplitterConfig(
        columns=["case_content"],
        model_name="sat-12l",
        verbose=True,
        sentence_threshold=0.2,
    )

    splitter = SentenceSpliiter(sent_config)

    print("finished initializing splitter")
    # print(type(hipaa_cases.data))
    splitted_tables = splitter(pa.Table.from_pandas(hipaa_cases.data.to_pandas()))

    print(splitted_tables["case_content_sentences"][0])

    pq.write_table(splitted_tables, "splitted_hipaa_cases.parquet")


def split_kb():
    dataset = load_from_disk(HF_KBs_path)
    hipaa_cases = dataset["HIPAA"]

    sent_config = SentenceSplitterConfig(
        columns=["regulation_content"],
        model_name="sat-12l",
        verbose=True,
        sentence_threshold=0.2,
    )

    splitter = SentenceSpliiter(sent_config)
    splitted_tables = splitter(pa.Table.from_pandas(hipaa_cases.data.to_pandas()))
    splitted_tables = splitted_tables.app

    print(splitted_tables["regulation_content_sentences"][0])

    pq.write_table(splitted_tables, "splitted_hipaa_kb.parquet")


def explore_csv():
    splitted_tables = pq.read_table("splitted_hipaa_cases.parquet")
    idx = 0
    specific_case_content = splitted_tables["case_content"][idx]

    specific_case_sentences = splitted_tables["case_content_sentences"][idx]
    specific_case_sentences = pa.array(specific_case_sentences)
    # specific_case_sentences = ast.literal_eval(specific_case_sentences)

    print(specific_case_content)
    print("=================")
    print(specific_case_sentences)
    print(type(specific_case_sentences))
    print(specific_case_sentences.to_numpy(zero_copy_only=False).astype(str).shape)


def get_sentence_emb_for_all_cases():
    splitted_tables = pq.read_table("splitted_hipaa_cases.parquet")
    case_sentences = splitted_tables["case_content_sentences"]

    emb_model = SentenceEmbeddingModelFactory.get_model("hf")

    embeddings = []

    for i, sentences in enumerate(case_sentences):

        sentences_array = pa.array(sentences).tolist()
        each_case_embeddings = emb_model.encode(sentences_array)
        each_case_embeddings_after_pooling = pool_embeddings(
            each_case_embeddings, pooling_type="mean"
        )
        embeddings.append(each_case_embeddings_after_pooling)

        print(each_case_embeddings_after_pooling.shape)

    print(len(embeddings))


def playground():
    text = '"164.502(a)(2)(i)"'
    gts = ["164.502(a)", "164.502"]

    for gt in gts:
        print(gt in text)


if __name__ == "__main__":
    playground()
