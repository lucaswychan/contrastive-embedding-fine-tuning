import os
from dataclasses import dataclass, field
from typing import Optional
import logging
from datetime import datetime

import torch
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import AutoModelForCausalLM, HfArgumentParser, set_seed

from loss import ContrastiveLoss
from trainer import ContrastiveSTTrainer
from utils import get_available_gpu_idx

CURRENT_TIME = datetime.now().strftime("%b-%d_%H-%M")

logger = logging.getLogger("training")
logging.basicConfig(
    filename="logs/training/{}.log".format(CURRENT_TIME), filemode="w", level=logging.INFO
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The training data file (.txt or .csv)."}
    )

@dataclass
class ContrastiveSTTrainingArguments(SentenceTransformerTrainingArguments):
    pass


def main():
    # available_gpu_idx = get_available_gpu_idx()
    # if available_gpu_idx is None:
    #     raise ValueError("No available GPU found!")

    # available_cuda = f"cuda:{available_gpu_idx}"
    # print(f"Using GPU: {available_cuda}")
    
    # device = torch.device(available_cuda)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, ContrastiveSTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = SentenceTransformer(model_args.model_name_or_path)
    
    set_seed(training_args.seed)
    
    print(training_args)

    # training_args = SentenceTransformerTrainingArguments(
    #     output_dir="output",
    #     overwrite_output_dir=True,
    #     num_train_epochs=3,
    #     per_device_train_batch_size=16,
    #     save_steps=1000,
    #     save_total_limit=2,
    #     remove_unused_columns=False,
    #     learning_rate=2e-5,
    #     adam_beta1=0.9,
    #     adam_beta2=0.999,
    #     adam_epsilon=1e-6,
    #     weight_decay=0.01,
    #     warmup_steps=100,
    #     logging_dir="log",
    #     logging_steps=10,
    # )

    train_dataset = load_dataset("csv", data_files=data_args.train_file)
    
    print(train_dataset)

    # loss = ContrastiveLoss(device=device)

    # trainer = ContrastiveSTTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     loss=loss,
    # )
    
    # # Train
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
        
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model(training_args.output_dir)
    
    # output_train_file = os.path.join(training_args.output_dir, "train_results_{}.txt".format(CURRENT_TIME))
    
    # if trainer.is_world_process_zero():
    #     with open(output_train_file, "w") as writer:
    #         logger.info("***** Train results *****")
            
    #         for key, value in sorted(train_result.metrics.items()):
    #             logger.info(f"  {key} = {value}")
    #             writer.write(f"{key} = {value}\n")
        
    #     trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state_{}.json".format(CURRENT_TIME)))
    

if __name__ == "__main__":
    main()
