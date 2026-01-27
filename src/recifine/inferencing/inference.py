from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from recifine.data.labels import get_labels_by_knowledge_type
from recifine.models.hf_wrappers import load_model_tokenizer

from recifine.utils.require_params import _require
from recifine.config import apply_layered_yaml_to_args
from recifine.utils.seed import set_seed

from recifine.data.kaner import read_example_from_data

from recifine.training.dataset import load_and_cache_examples_from_list
from recifine.data.transform import normalise_traditional_prediction, normalise_knowledge_prediction

from tqdm import tqdm, trange

from numpy import random

logger = logging.getLogger(__name__)


def _index_entity_groups(entity_groups: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for eg in entity_groups:
        eg_name = eg.get("name", "")
        if eg_name:
            idx[eg_name] = eg
    return idx


class ReciFineNER:
    def __init__(
        self,
        *,
        dataset: str = "recifinegold",
        model: str = "recipebert",
        knowledge_type: str = "question",
        task_formulation: str = "knowledge_guided",
        model_name_or_path: Optional[str] = None,
        config: Optional[List[str] | str] = None,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        per_gpu_inf_batch_size: int = 256,
        seed: int = 42,
        no_cuda: bool = False
    ) -> None:
        args = argparse.Namespace(
            dataset=dataset,
            model=model,
            knowledge_type=knowledge_type,
            task_formulation=task_formulation,
            model_name_or_path=model_name_or_path or "",
            config=config,
            cache_dir="",
            max_seq_length=max_seq_length,
            per_gpu_inf_batch_size=per_gpu_inf_batch_size,
            seed=seed,
            no_cuda=no_cuda,
            local_rank=-1,
            labels=None,
            config_name="",
            tokenizer_name="",
            do_lower_case=None,
        )

        args = apply_layered_yaml_to_args(
            args,
            base_model_attr="model",
            dataset_attr="dataset",
            knowledge_type_attr="knowledge_type",
            extra_attr="config",
            config_type="inference",
        )

        if args.entity_groups:
            _require(args, ["model_name_or_path",])
        else:
            _require(args, ["model_name_or_path", "labels",])

        self.args = args

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name_or_path: Optional[str] = None,
        dataset: str = "recifinegold",
        model: str = "recipebert",
        task_formulation: str = "knowledge_guided",
        knowledge_type: str = "question",
        config: Optional[List[str] | str] = None,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        seed: int = 42,
        no_cuda: bool = False,
        **kwargs: Any,
    ) -> "ReciFineNER":
        if kwargs:
            logger.warning("Ignoring unexpected kwargs in from_pretrained: %s", sorted(kwargs.keys()))
        return cls(
            dataset=dataset,
            model=model,
            task_formulation=task_formulation,
            knowledge_type=knowledge_type,
            model_name_or_path=model_name_or_path,
            config=config,
            device=device,
            max_seq_length=max_seq_length,
            seed=seed,
            no_cuda=no_cuda,
        )

    def process_text(
        self,
        text: str,
        *,
        entity_type: Optional[str] = None,
        return_tokens: bool = False,
    ):

        logging.basicConfig(level=logging.INFO)

        args = self.args

        # logging.info(args)

        # validate that text is provided
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to predict must be a non-empty string")

        kt = args.knowledge_type
        tf = args.task_formulation

        args.entity_groups: List[Dict[str, Any]] = list(args.entity_groups)
        args.entity_group_index = _index_entity_groups(args.entity_groups)

        needs_group = (kt != "traditional")
        eg = {}
        if needs_group:
            if not entity_type:
                raise ValueError(
                    "entity_type is required for knowledge-guided inference. "
                    "Example: process_text(text, entity_type='FOOD' ...)"
                )
            eg = args.entity_group_index.get(entity_type)
            if eg is None:
                examples = sorted(list(args.entity_group_index.keys()))[:20]
                raise ValueError(f"Unknown entity_type='{entity_type}'. Available (examples): {examples} ...")


        if not args.labels:
            args.labels = get_labels_by_knowledge_type(args.entity_groups, args.knowledge_type)

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.device = device
        args.n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0

        set_seed(args.seed, args.n_gpu)

        pad_token_label_id = CrossEntropyLoss().ignore_index

        _, tokenizer, model = load_model_tokenizer(
            model_type=args.model_type,
            model_name_or_path=args.model_name_or_path,
            num_labels=len(args.labels),
            config_name=args.config_name,
            tokenizer_name=args.tokenizer_name,
            do_lower_case=args.do_lower_case,
        )
        model.to(args.device)
        model.eval()

        set_seed(args.seed, args.n_gpu)

        inference_batch_size = args.per_gpu_inf_batch_size * max(1, args.n_gpu)
        label_map = {i: label for i, label in enumerate(args.labels)}

        inf_data = []
        if not args.knowledge_type == "traditional":
            # get corresponding entity group
            entity_group = args.entity_group_index.get(entity_type)

            # prepare inference data
            inf_data = [
                {
                    "qid": f"{random.randint(1000000)}", 
                    "text": f"{text}", 
                    "entity_type": f"{entity_group.get('type')}", 
                    "entity_group": f"{entity_group.get('group')}", 
                    "group_definition": f"{entity_group.get('group_definition')}", 
                    "definition": f"{entity_group.get('definition')}", 
                    "example": f"{entity_group.get('examples')}", 
                    "question": f"{entity_group.get('question')}",
                    "answer": [],
                },
            ]
        else:
            # prepare inference data
            inf_data = [
                {
                    "qid": f"{random.randint(1000000)}", 
                    "text": f"{text}", 
                    "traditional": ["O"] * len(text.split(" ")),
                },
            ]

        examples = read_example_from_data(inf_data, args.knowledge_type)
        eval_dataset = load_and_cache_examples_from_list(args, tokenizer, args.labels, pad_token_label_id, inf_data, "inf")

        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=inference_batch_size)

        logger.info("***** Running inference *****")
        logger.debug("  Num examples = %d", len(eval_dataset))
        logger.debug("  Batch size = %d", inference_batch_size)

        example_idx = 0
        predictions = []
        for batch in tqdm(eval_dataloader, ascii=True, desc="Inferencing"):
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

                predictions.append({"answer": answers, "words_list": text.split(" "), "pred_labels": pred_labels[-len(text.split(" ")):],})

        return predictions
