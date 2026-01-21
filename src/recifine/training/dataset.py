from __future__ import annotations

import logging
import os
import torch
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss

from recifine.data import convert_examples_to_features
from recifine.data import read_examples_from_file

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode: str):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    cached_features_file = os.path.join(
        args.data_dir,
        ("cached_{}_{}_{}_" + args.knowledge_type).format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        examples = read_examples_from_file(args.data_dir, mode, args.knowledge_type)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
