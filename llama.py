import torch
from transformers import pipeline

from utils import get_available_gpu_idx

available_gpu_idx = get_available_gpu_idx()
if available_gpu_idx is None:
    raise ValueError("No available GPU found!")

available_cuda = f"cuda:{available_gpu_idx}"
print(f"Using GPU: {available_cuda}")


class Llama3P2:
    def __init__(self):
        self.device = torch.device(
            available_cuda if torch.cuda.is_available() else "cpu"
        )
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device=self.device,
        )

    def generate(
        self, sys_prompt: str, user_prompt: str, max_new_token=1500, temperature=0.6
    ) -> str:
        """
        Generate text based on the prompt and optional image input.
        """
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_token,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            eos_token_id=terminators,
            top_p=0.9,
        )

        result = outputs[0]["generated_text"][-1]["content"]

        return result

    def __call__(
        self, sys_prompt: str, user_prompt: str, max_new_token=1500, temperature=0.6
    ):
        return self.generate(sys_prompt, user_prompt, max_new_token, temperature)


if __name__ == "__main__":
    llm = Llama3P2()

    sys_prompt = "You are a helpful assistant."
    user_prompt = "What is contextual integrity theory?"

    res = llm(sys_prompt, user_prompt)

    print(res)
