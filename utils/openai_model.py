import argparse
import json
import os
import random
import time

import openai
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# attribute_path = os.path.join(BASE_DIR,'kgs', 'prompt_attribute_list.json')
# role_path = os.path.join(BASE_DIR,'kgs', 'prompt_role_kg_list.json')

attribute_merge_path = os.path.join(
    BASE_DIR, "ACLU050224", "prompt_list_aclu_cases.json"
)
# Define the OpenAI API key
api_key = "sk-Ew0BDi68GhHetFWXt5CT5Wb3UrbFstFXmv6CYUmD0ZffbEhk"
logging_path = "logs_attribute_kg_new.txt"


def logging(msg):
    with open(logging_path, "a") as f:
        f.write(msg + "\n")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(data, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


"""
response = self.chatgpt.chat.completions.create(
    model=self.gpt_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": data}
    ],
    max_tokens=max_new_tokens,
    temperature=temperature
)
"""


class OpenAI_Model:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.oneabc.org/v1")

    def compeletion(self, model: str, messages: list, max_retries: int, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                # Call the OpenAI ChatCompletion API
                response = self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                msg = response.choices[0].message.content
                assert isinstance(msg, str), "The retruned response is not a string."
                return msg  # Return the response if successful
            except Exception as e:
                # Catch all other exceptions
                print(f"Unexpected error: {e}. Retrying in 5 seconds...")
                retries += 1
                time.sleep(5)

        return ""  # Return an empty string if max_retries is exceeded

    def compeletion_for_str(
        self, model: str, messages: str, max_retries: int, **kwargs
    ):
        retries = 0
        ### convert message to chat template list
        msg_list = []
        msg_list.append({"role": "system", "content": "You are a helpful assistant."})
        msg_list.append({"role": "user", "content": messages})
        while retries < max_retries:
            try:
                # Call the OpenAI ChatCompletion API
                response = self.client.chat.completions.create(
                    model=model, messages=msg_list, **kwargs
                )
                msg = response.choices[0].message.content
                assert isinstance(msg, str), "The retruned response is not a string."
                return msg  # Return the response if successful

            except Exception as e:
                # Catch all other exceptions
                print(f"Unexpected error: {e}. Retrying in 5 seconds...")
                retries += 1
                time.sleep(5)

        return ""  # Return an empty string if max_retries is exceeded


def parepare_response(model: OpenAI_Model, data_list: list, save_path: str, args):
    ret = []
    # new_save_path = save_path.replace('.json', f'_new.json')
    # with open(save_path, 'r', encoding='utf-8') as f:
    #     saved_data = json.load(f)
    # len_saved_data = len(saved_data)
    # print(f"len_saved_data: {len_saved_data}")
    # logging(f"len_saved_data: {len_saved_data}")

    # save_path = new_save_path
    for i, msg in tqdm(enumerate(data_list)):
        # if i < len_saved_data:
        #    continue
        temp = {}
        # compeletion_for_str
        # response = model.compeletion(args.model, msg, args.max_retry, temperature = args.temperature, max_tokens = args.max_tokens)
        response = model.compeletion_for_str(
            args.model,
            msg,
            args.max_retry,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if response == "":
            logging(f"--- Failed to get response for index {i}")
            continue
        temp["input"] = msg
        temp["response"] = response
        ret.append(temp)
        logging(f"Response for index {i}: {ret}")
        if i % 10 == 0:
            save_to_json(ret, save_path)
    save_to_json(ret, save_path)


def main(args):

    model_api = OpenAI_Model(api_key=args.api_key)
    # attribute_list = load_json(args.attribute_path)
    role_list = load_json(args.role_path)
    # attribute_save_path = args.attribute_path.replace('.json', f'_response_{args.model}.json')
    role_save_path = args.role_path.replace(".json", f"_response_{args.model}.json")
    # parepare_response(model_api, attribute_list, attribute_save_path, args)
    parepare_response(model_api, role_list, role_save_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api_key", type=str, default=api_key)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retry", type=int, default=3)
    # parser.add_argument("--attribute_path", type=str, default=attribute_path)
    # parser.add_argument("--role_path", type=str, default=role_path)
    parser.add_argument("--role_path", type=str, default=attribute_merge_path)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()
    main(args)
