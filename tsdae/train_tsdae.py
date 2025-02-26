import os
import sys

# For me to import utils module from outside
sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

# set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from denoise_dataset import DenoisingAutoEncoderDataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer, datasets,
                                   evaluation, losses, models, util)
from sentence_transformers.training_args import \
    SentenceTransformerTrainingArguments
from transformers import AutoModelForCausalLM, HfArgumentParser, set_seed

from config import HF_KBs_path

CURRENT_TIME = datetime.now().strftime("%b-%d_%H-%M")

logger = logging.getLogger("training")
logging.basicConfig(
    filename="logs/training/{}.log".format(CURRENT_TIME),
    filemode="w",
    level=logging.INFO,
)

available_cuda = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES']}"
print(f"Using GPU: {available_cuda}")

device = torch.device(available_cuda)


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
    use_labels: bool = field(
        default=False,
        metadata={"help": "Whether to use labels or not in constrastive loss."},
    )

    use_role_graph_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to use role graph data or not in constrastive loss."
        },
    )

    @property
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # _ = self._setup_devices
        # I set to one manullay
        self._n_gpu = 1
        return self._n_gpu

    def __post_init__(self):
        super().__post_init__()
        print("Using GPU in Training Argument: {}".format(self.device))


def main():

    # get the arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ContrastiveSTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # get the model for training
    model = SentenceTransformer(model_args.model_name_or_path, device=device)

    set_seed(training_args.seed)

    # load the dataset
    KB = load_from_disk(HF_KBs_path)
    kb_contents = KB["HIPAA"]["regulation_content"]
    train_dataset = DenoisingAutoEncoderDataset(kb_contents)
    train_dataset = DatasetDict({"train": train_dataset})

    # define the loss
    loss = losses.DenoisingAutoEncoderLoss(
        model=model, decoder_name_or_path="bert-base-uncased", tie_encoder_decoder=False
    )

    # set the data train path if using role graph data together
    if training_args.use_role_graph_data:
        data_args.train_file = "data/keywords_with_rolekg.csv"

    # define the file path
    label_file_suffix = "no_labels" if not training_args.use_labels else "labels"
    rkg_file_suffix = "no_rkg" if not training_args.use_role_graph_data else "rkg"
    training_args.logging_dir = os.path.join(
        training_args.logging_dir,
        f"{CURRENT_TIME}_{label_file_suffix}_{rkg_file_suffix}",
    )
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"{CURRENT_TIME}_{label_file_suffix}_{rkg_file_suffix}",
    )

    print(f"Logging directory: {training_args.logging_dir}")
    print(f"Output directory: {training_args.output_dir}")

    # define the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print("===========Starting training===========")

    # start training
    train_result = trainer.train()

    print("===========Training done===========")

    # save training results
    output_train_file = os.path.join(
        training_args.output_dir, "train_results_{}.txt".format(CURRENT_TIME)
    )

    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")

            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

            logger.info(f" use labels = {training_args.use_labels}")
            writer.write(f"use labels = {training_args.use_labels}\n")

        trainer.state.save_to_json(
            os.path.join(
                training_args.output_dir, "trainer_state_{}.json".format(CURRENT_TIME)
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
