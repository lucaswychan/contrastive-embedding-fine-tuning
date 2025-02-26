import os
import sys

# For me to import utils module from outside
sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

from datasets import load_dataset, load_from_disk
from sentence_transformers import (LoggingHandler, SentenceTransformer,
                                   datasets, evaluation, losses, models, util)
from torch.utils.data import DataLoader

device = "cuda:1"

# Define your sentence transformer model using CLS pooling
model_name = "sentence-transformers/all-mpnet-base-v2"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), "cls"
)
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device=device
)

from config import HF_KBs_path

# Define a list with sentences (1k - 100k sentences)
KB = load_from_disk(HF_KBs_path)
train_sentences = KB["HIPAA"]["regulation_content"]

# Create the special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch your data
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(
    model, decoder_name_or_path="bert-base-uncased", tie_encoder_decoder=False
)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    weight_decay=0,
    scheduler="constantlr",
    optimizer_params={"lr": 3e-5},
    show_progress_bar=True,
)

model.save("output/tsdae-model")
