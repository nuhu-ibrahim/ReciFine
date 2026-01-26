from __future__ import annotations

import logging
from typing import List

from tqdm import tqdm

from .kaner import InputExample, InputFeatures

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer,
    cls_token_at_end: bool = False,
    cls_token: str = "[CLS]",
    cls_token_segment_id: int = 1,
    sep_token: str = "[SEP]",
    sep_token_extra: bool = False,
    pad_on_left: bool = False,
    pad_token: int = 0,
    pad_token_segment_id: int = 0,
    pad_token_label_id: int = -1,
    sequence_a_segment_id: int = 0,
    mask_padding_with_zero: bool = True,
) -> List[InputFeatures]:
    label_map = {label: i for i, label in enumerate(label_list)}

    features: List[InputFeatures] = []
    for (ex_index, example) in tqdm(list(enumerate(examples))):
        if ex_index % 10000 == 0:
            logger.debug("Writing example %d of %d", ex_index, len(examples))

        tokens: List[str] = []
        label_ids: List[int] = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        while len(label_ids) > max_seq_length:
            label_ids.pop(-1)
        while len(label_ids) < max_seq_length:
            label_ids.append(pad_token_label_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.debug("*** Example ***")
            logger.debug("guid: %s", example.guid)
            logger.debug("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.debug("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.debug("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.debug("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.debug("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                counts=example.counts,
            )
        )

    return features
