import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Union

# For me to import utils module from outside
sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import torch
from stopes.utils.arrow_utils import apply_on_nested_array
from wtpsplit import SaT, indices_to_sentences

from utils import get_available_gpu_idx


def remove_unicode(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001f700-\U0001f77f"  # Alchemical Symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes Extended
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U0001f6c0-\U0001f6cf"  # Miscellaneous Symbols and Pictographs (part)
        "\U0001f6d0-\U0001f6d5"  # Miscellaneous Symbols and Pictographs (part)
        "\U0001f6f0-\U0001f6fa"  # Miscellaneous Symbols and Pictographs (part)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def resplit(text: str, max_length: int, sep: str) -> List[str]:
    words = text.split(sep)
    result = []
    current_piece = ""

    for i, word in enumerate(words[:-1]):
        # Append separator back to each word except the last
        word += sep
        if len(current_piece) + len(word) <= max_length:
            current_piece += word
        else:
            if current_piece:
                result.append(current_piece)
            current_piece = word

    # Handle the last word separately to avoid adding an extra separator
    last_word = words[-1]
    if len(current_piece) + len(last_word) <= max_length:
        current_piece += last_word
    else:
        if current_piece:
            result.append(current_piece)
        current_piece = last_word

    if current_piece:
        result.append(current_piece)

    return result


@dataclass
class SentenceSplitterConfig:
    columns: List[str]
    model_name: str = "sat-12l"
    sentence_suffix: str = "_sentences"
    sentence_threshold: float = 0.01
    max_sentence_len: int = 256
    min_text_length: int = 10
    min_unique_chars: int = 0
    fallback_separators: List[str] = field(
        default_factory=lambda: [
            "...",
            "\n",
            "!",
            "?",
            ";",
            ":",
            ".",
            ",",
            "\t",
            " ",
        ]
    )
    device: str = "cuda"
    remove_whitespace_before_inference: bool = False
    batch_size: int = 256
    block_size: int = 256
    # ad 1.: text is sliced into partially overlapping chunks by moving forward by a `stride` parameter (think conv1d).
    stride: int = 256  # no overlapping chunks for sentence splitted by SaT
    outer_batch_size: int = 1024
    verbose: bool = False
    pad_last_batch: bool = False


class SentenceSpliiter:
    def __init__(self, config: SentenceSplitterConfig):
        self.config = config

        self.model = SaT(model_name_or_model=config.model_name)
        device = config.device

        if "cuda" in config.device:
            self.model.half()
            device_idx = get_available_gpu_idx()
            device = f"cuda:{device_idx}" if device_idx is not None else "cpu"

        print(f"Using device: {device}")

        self.model.eval().to(device)

    @torch.no_grad()
    def _resplit_long_sentences(self, split_target: pa.Array) -> pa.Array:
        mask = pc.greater_equal(
            pc.utf8_length(split_target), self.config.max_sentence_len
        )
        texts_to_resplit = split_target.filter(mask).to_pandas().to_list()

        resplit_sentences = []
        for text, probs in zip(
            texts_to_resplit,
            self.model.predict_proba(
                texts_to_resplit,
                stride=self.config.stride,
                block_size=self.config.block_size,
                batch_size=self.config.batch_size,
                pad_last_batch=self.config.pad_last_batch,
                remove_whitespace_before_inference=self.config.remove_whitespace_before_inference,
                outer_batch_size=self.config.outer_batch_size,
                verbose=self.config.verbose,
            ),
        ):
            nb_split = round(len(probs) / self.config.max_sentence_len) + 1
            sentence_threshold = np.partition(probs, -nb_split)[-nb_split]
            sentences = indices_to_sentences(
                text,
                np.where(probs >= sentence_threshold)[0],
                strip_whitespace=False,
            )
            resplit_sentences.append(sentences)

        # if not, hard resplit with some separators
        def _resplit(raw_sentences):
            for separator in self.config.fallback_separators:
                raw_sentences = [
                    subchunk.strip()
                    for sent in raw_sentences
                    for subchunk in resplit(
                        sent, max_length=self.config.max_sentence_len, sep=separator
                    )
                ]
            return raw_sentences

        np_mask = mask.to_pandas().to_numpy()
        full_text = split_target.to_pandas().to_list()

        output_sentences = []
        j = 0
        for i, text in enumerate(full_text):
            if np_mask[i]:
                output_sentences.append(_resplit(resplit_sentences[j]))
                j += 1
            else:
                output_sentences.append([text])

        return pa.array(output_sentences, type=pa.list_(pa.string()))

    def resplit_long_sentences(self, col: pa.Array) -> pa.Array:
        list_col = apply_on_nested_array(self._resplit_long_sentences, col)
        reflatten_col = pl.from_arrow(list_col).list.eval(pl.element().explode())  # type: ignore
        # remove single char repeated
        if self.config.min_unique_chars > 0:
            reflatten_col = reflatten_col.list.eval(
                pl.when(
                    pl.element().str.split("").list.n_unique()
                    > self.config.min_unique_chars
                )
                .then(pl.element())
                .drop_nulls()
            )
        return reflatten_col.to_arrow().cast(pa.list_(pa.string()))

    @torch.no_grad()
    def basic_split_on_single_column(
        self,
        split_target: Union[pa.Array, pa.ChunkedArray],
    ) -> Union[pa.Array, pa.ChunkedArray]:
        if not (
            pa.types.is_large_string(split_target.type)
            or pa.types.is_string(split_target.type)
        ):
            raise ValueError("Column must be of type string")

        texts = split_target.to_pandas().to_list()
        texts = list(map(remove_emojis, texts))

        # only split long texts
        long_texts = [t for t in texts if len(t) > self.config.min_text_length]
        # the short texts remain unchanged and will be added back later
        keep_texts = [
            (idx, t)
            for idx, t in enumerate(texts)
            if len(t) <= self.config.min_text_length
        ]

        outputs = self.model.split(
            long_texts,
            threshold=self.config.sentence_threshold,
            stride=self.config.stride,
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            pad_last_batch=self.config.pad_last_batch,
            remove_whitespace_before_inference=self.config.remove_whitespace_before_inference,
            outer_batch_size=self.config.outer_batch_size,
            verbose=self.config.verbose,
        )
        sentences = []
        for row in outputs:
            sentences.append([s.strip() for s in row if s.strip()])

        # add back short texts
        for idx, text in keep_texts:
            sentences.insert(idx, text)

        return pa.array(sentences, type=pa.list_(pa.string()))

    def __call__(self, table: pa.Table) -> pa.Table:
        for column in self.config.columns:
            sentence_array = self.basic_split_on_single_column(table[column])

            sentence_array = self.resplit_long_sentences(sentence_array)

            print(len(sentence_array[0]))
            print(len(sentence_array[1]))
            print(len(sentence_array[2]))

            table = table.append_column(
                f"{column}{self.config.sentence_suffix}", sentence_array
            )

        return table
