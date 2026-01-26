from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import jsonlines
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from recifine.training.dataset import load_and_cache_examples
from recifine.data.kaner import read_examples_from_file
from recifine.data.transform import normalise_traditional_prediction, normalise_knowledge_prediction

logger = logging.getLogger(__name__)


def predict_without_evaluate(
    args,
    model,
    tokenizer,
    labels: List[str],
    pad_token_label_id: int,
    mode: str = "dev",
    prefix: str = "",
) -> str:
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    output_file_path = os.path.join(args.output_dir, args.model_type + "_" + args.knowledge_type + "_test_predictions.jsonl")
    label_map = {i: label for i, label in enumerate(labels)}

    examples = read_examples_from_file(args.data_dir, "test", args.knowledge_type)

    example_idx = 0
    with jsonlines.open(output_file_path, "w") as writer:
        for batch in tqdm(eval_dataloader, ascii=True, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2] if args.model_type in ["recipebert", "xlnet"] else None,
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                logits = outputs[1]


            preds = torch.argmax(logits, axis=2).detach().cpu().numpy()
            label_ids = inputs["labels"].detach().cpu().numpy()

            for i in range(preds.shape[0]):
                example = examples[example_idx]
                entity_type = example.orig_row.get("entity_type")

                pred_labels = [
                    label_map[preds[i][j]]
                    for j in range(len(preds[i]))
                    if label_ids[i][j] != pad_token_label_id
                ]

                if args.knowledge_type == "traditional":
                    answers = normalise_traditional_prediction(example.words, pred_labels)
                else:
                    answers = normalise_knowledge_prediction(example.words, pred_labels, entity_type)

                writer.write(
                    {
                        "qid": example.orig_row.get("qid", str(example_idx)) if example.orig_row else str(example_idx),
                        "text": example.orig_row.get("text", "") if example.orig_row else "",
                        "input_words": example.words,
                        "predicted_label": pred_labels,
                        "answer": answers,
                    }
                )
                example_idx += 1

    logger.info(f"Saved streamed predictions to: {output_file_path}")
    return output_file_path
