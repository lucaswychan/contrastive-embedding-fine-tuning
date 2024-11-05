---
base_model: sentence-transformers/all-mpnet-base-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1613
- loss:ContrastiveLoss
widget:
- source_sentence: Correctional institution
  sentences:
  - crimes
  - office
  - utilization review
- source_sentence: Individual
  sentences:
  - Provider
  - Occupational therapy
  - login
- source_sentence: Treatment
  sentences:
  - information
  - Medical
  - Transition of care
- source_sentence: Required by law
  sentences:
  - grand jury
  - authority
  - identification
- source_sentence: Health care operations
  sentences:
  - business planning
  - government office
  - healthcare industry
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) on the train dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision f1b1b820e405bb8644f5e8d9a3b98f9c9e0a3c58 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - train
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Health care operations',
    'business planning',
    'healthcare industry',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### train

* Dataset: train
* Size: 1,613 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                          | positive                                                                        | label                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:--------|:-------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | type    | string                                                                         | string                                                                          | int                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | details | <ul><li>min: 3 tokens</li><li>mean: 4.1 tokens</li><li>max: 6 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 3.58 tokens</li><li>max: 9 tokens</li></ul> | <ul><li>0: ~1.40%</li><li>1: ~1.40%</li><li>2: ~2.70%</li><li>3: ~2.80%</li><li>4: ~2.70%</li><li>5: ~3.30%</li><li>6: ~1.70%</li><li>7: ~3.30%</li><li>8: ~3.60%</li><li>9: ~3.00%</li><li>10: ~3.00%</li><li>11: ~2.80%</li><li>12: ~4.00%</li><li>13: ~3.40%</li><li>14: ~2.10%</li><li>15: ~4.20%</li><li>16: ~3.20%</li><li>17: ~3.50%</li><li>18: ~4.30%</li><li>19: ~3.40%</li><li>20: ~2.70%</li><li>21: ~3.50%</li><li>22: ~3.30%</li><li>23: ~3.20%</li><li>24: ~2.80%</li><li>25: ~2.90%</li><li>26: ~2.20%</li><li>27: ~2.90%</li><li>28: ~3.70%</li><li>29: ~1.60%</li><li>30: ~3.40%</li><li>31: ~2.70%</li><li>32: ~2.10%</li><li>33: ~3.20%</li></ul> |
* Samples:
  | query                                     | positive               | label          |
  |:------------------------------------------|:-----------------------|:---------------|
  | <code>Implementation specification</code> | <code>standard</code>  | <code>0</code> |
  | <code>Implementation specification</code> | <code>procedure</code> | <code>0</code> |
  | <code>Implementation specification</code> | <code>contract</code>  | <code>0</code> |
* Loss: <code>loss.ContrastiveLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `overwrite_output_dir`: True
- `num_train_epochs`: 1.0
- `seed`: 1016
- `bf16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: True
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1.0
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 1016
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: True
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `use_labels`: False

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0495 | 10   | 2.1809        |
| 0.0990 | 20   | 1.8733        |
| 0.1485 | 30   | 1.9293        |
| 0.1980 | 40   | 2.2601        |
| 0.2475 | 50   | 2.0987        |
| 0.2970 | 60   | 1.8648        |
| 0.3465 | 70   | 2.1615        |
| 0.3960 | 80   | 2.2609        |
| 0.4455 | 90   | 2.0701        |
| 0.4950 | 100  | 2.2495        |
| 0.5446 | 110  | 2.0508        |
| 0.5941 | 120  | 1.9003        |
| 0.6436 | 130  | 2.0644        |
| 0.6931 | 140  | 1.9925        |
| 0.7426 | 150  | 1.7894        |
| 0.7921 | 160  | 2.0347        |
| 0.8416 | 170  | 2.2247        |
| 0.8911 | 180  | 2.1084        |
| 0.9406 | 190  | 2.0234        |
| 0.9901 | 200  | 2.158         |


### Framework Versions
- Python: 3.10.4
- Sentence Transformers: 3.1.1
- Transformers: 4.45.2
- PyTorch: 2.4.1+cu121
- Accelerate: 1.0.0
- Datasets: 3.1.0
- Tokenizers: 0.20.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->