import os

DATA_DIR = "/home/data/hlibt/checklist2_WIP/"

HF_cases_path = os.path.join(DATA_DIR, "HF_cache", "cases")
HF_KBs_path = os.path.join(DATA_DIR, "HF_cache", "KBs")

PROMPT_GET_KEY_WORD = """
You are a lawyer. You will be given a regulation from {privacy_rule_name} and your task is to extract legal terminologies from the given regulation.

Let's think step by step:
1. Understand the regulation: Read carefully the regulation.
2. Think clearly about the objective of the regulation.
3. Identify the key words: Identify the key words that are related to the objective of the regulation.
4. Revisit the key words: prevent the duplicate key words and revisit the key words.

Your output must be in JSON format, with an outer keys of "context" followed by a list. The JSON format is as follows:
{
    "context": [
        "keyword1",
        "keyword2",
        "keyword3"
    ]
}

You should not output anything else than the JSON format.

Regulation:
{regulation}

Assistant:
"""
