import json
import re

import nvidia_smi
import torch


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


def get_available_gpu_idx():
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
                    return i
            except RuntimeError:
                # If allocation fails, move to the next GPU
                continue

    # If no available GPU is found
    return None
