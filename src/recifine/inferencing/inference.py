from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from recifine.data.labels import get_labels_by_knowledge_type
from recifine.models.hf_wrappers import load_model_tokenizer_fast

from recifine.utils.require_params import _require
from recifine.config import apply_layered_yaml_to_args
from recifine.utils.seed import set_seed
from recifine.data.kaner import build_model_inference_input

logger = logging.getLogger(__name__)


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str


def _bio_to_spans(text: str, offsets: List[Tuple[int, int]], labels: List[str]) -> List[Span]:
    spans: List[Span] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_type: Optional[str] = None

    def _flush():
        nonlocal cur_start, cur_end, cur_type
        if cur_start is not None and cur_end is not None and cur_type:
            spans.append(Span(start=cur_start, end=cur_end, label=cur_type, text=text[cur_start:cur_end]))
        cur_start, cur_end, cur_type = None, None, None

    for (s, e), lab in zip(offsets, labels):
        if s == e:
            continue

        if lab == "O" or lab is None:
            _flush()
            continue

        if lab.startswith("B-"):
            _flush()
            cur_type = lab[2:]
            cur_start, cur_end = s, e
        elif lab.startswith("I-"):
            t = lab[2:]
            if cur_type == t and cur_start is not None:
                cur_end = e
            else:
                _flush()
                cur_type = t
                cur_start, cur_end = s, e
        else:
            _flush()

    _flush()
    return spans


def _index_entity_groups(entity_groups: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for eg in entity_groups:
        eg_name = eg.get("name", "")
        if eg_name:
            idx[eg_name] = eg
    return idx


class RecifineNER:
    def __init__(
        self,
        *,
        dataset: str = "recifinegold",
        model: str = "bert_base_uncased",
        knowledge_type: str = "question",
        task_formulation: str = "knowledge_guided",
        model_name_or_path: Optional[str] = None,
        config: Optional[List[str] | str] = None,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        seed: int = 42,
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
            seed=seed,
            no_cuda=False,
            local_rank=-1,
            labels="",
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
            config_type="prediction",
        )

        if args.entity_groups:
            _require(args, ["model_name_or_path",])
        else:
            _require(args, ["model_name_or_path", "labels",])

        # Setup CUDA / distributed (exact semantics)
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1

        args.device = device

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
            args.local_rank,
            device,
            args.n_gpu,
            bool(args.local_rank != -1)
        )

        self.args = args
        self.max_seq_length = args.max_seq_length
        self.device = args.device
        self.n_gpu = args.n_gpu

        set_seed(args.seed, args.n_gpu)

        self.entity_groups: List[Dict[str, Any]] = list(args.entity_groups)
        self.entity_group_index = _index_entity_groups(self.entity_groups)

        logger.info("Loading model/tokenizer from: %s", args.model_name_or_path)

        labels = []
        if labels:
            labels = args.labels
        else:    
            labels = get_labels_by_knowledge_type(self.entity_groups, args.knowledge_type)

        num_labels = len(labels)

        _, self.tokenizer, self.model = load_model_tokenizer_fast(
            model_type=args.model_type,
            model_name_or_path=args.model_name_or_path,
            num_labels=num_labels,
            config_name=args.config_name,
            tokenizer_name=args.tokenizer_name,
            do_lower_case=args.do_lower_case,
        )

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

        # Build id2label map
        cfg = self.model.config

        self.id2label: Dict[int, str] = (
            {int(k): v for k, v in cfg.id2label.items()}
            if getattr(cfg, "id2label", None)
            else {}
        )

        if not self.id2label:
            raise ValueError(
                "Loaded model has no id2label. Ensure it is a token classification checkpoint."
            )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name_or_path: str,
        dataset: str = "recifinegold",
        model: str = "bert",
        task_formulation: str = "knowledge_guided",
        knowledge_type: str = "question",
        config: Optional[List[str] | str] = None,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        seed: int = 42,
        no_cuda: bool = False,
        **kwargs: Any,
    ) -> "RecifineNER":
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
    ) -> List[Span] | Dict[str, Any]:

        # validate that text is provided
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to predict must be a non-empty string")

        kt = self.args.knowledge_type
        tf = self.args.task_formulation

        needs_group = (kt != "traditional")
        eg = {}
        if needs_group:
            if not entity_type:
                raise ValueError(
                    "entity_type is required for knowledge-guided inference. "
                    "Example: process_text(text, entity_type='FOOD' ...)"
                )
            eg = self.entity_group_index.get(entity_type)
            if eg is None:
                examples = sorted(list(self.entity_group_index.keys()))[:20]
                raise ValueError(f"Unknown entity_type='{entity_type}'. Available (examples): {examples} ...")

        model_input, knowledge = build_model_inference_input(text, eg, kt)

        enc = self.tokenizer(
            model_input,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            pred_ids = out.logits.argmax(dim=-1)[0].tolist()

        labels = [self.id2label.get(i, "O") for i in pred_ids]

        shift = len(knowledge) + 1 if knowledge else 0

        spans_in_model_input = _bio_to_spans(model_input, offsets, labels)

        spans: List[Span] = []
        for s in spans_in_model_input:
            if s.end <= shift:
                continue

            new_start = max(0, s.start - shift)
            new_end = max(0, s.end - shift)

            new_start = min(new_start, len(text))
            new_end = min(new_end, len(text))
            if new_end <= new_start:
                continue

            spans.append(Span(start=new_start, end=new_end, label=s.label, text=text[new_start:new_end]))

        if not return_tokens:
            return spans

        toks = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        return {
            "model_input": model_input,
            "tokens": toks,
            "offsets": offsets,
            "bio_tags": labels,
            "spans": spans,
        }
