import json

from llama3p2 import Llama3P2
from utils import convert_json
import logging

logging.basicConfig(filename='log/data_augmentation.log', filemode="w", level=logging.INFO)


def augment_data():
    with open("../HIPAA_defs.json") as f:
        HIPAA_DEF = json.load(f)

    llm = Llama3P2()

    sys_prompt = "You are a legal expert that answers question as simple as possible."
    cot_user_prompt = """
    You will be given a legal terminology with its corresponding definitions. Your task is to provide multiple daily context key words that are related to the given terminology.
    
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
    
    for term, definition in HIPAA_DEF.items():
        target_prompt = cot_user_prompt.format(
            term=term, definition=definition
        )
        
        try:
            response = llm(sys_prompt, target_prompt)
            logging.info(response)
        except Exception as e:
            logging.info(e)
            continue
        
        res_dict = convert_json(response)
        
        term_map[term] = list(set(res_dict["context"]))
        
        with open("data/keywords.json", "w") as f:
            json.dump(term_map, f, indent=4)
        
        print("finished: ", term)


if __name__ == "__main__":
    augment_data()
