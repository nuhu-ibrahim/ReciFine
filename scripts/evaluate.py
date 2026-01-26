from __future__ import annotations

import argparse
import logging
import torch
from torch.nn import CrossEntropyLoss
import os

from recifine.config import apply_layered_yaml_to_args
from recifine.data.labels import get_labels
from recifine.models.hf_wrappers import load_model_tokenizer
from recifine.training.eval import evaluate
from recifine.utils.seed import set_seed
from recifine.utils.require_params import _require
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default=None,
                   help="Base data directory containing train/val/test splits.")

    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory containing trained model and for saving outputs.")

    p.add_argument("--model_type", type=str, default=None,
                   choices=["recipebert", "reciperoberta"], help="Model architecture key.")

    p.add_argument("--model_name_or_path", type=str, default=None,
                   help="HF model name or local checkpoint path.")

    p.add_argument("--dataset_to_evaluate", nargs="+", default=None,
                   help="Optional dataset subfolders to evaluate.")

    p.add_argument("--labels", type=str, default="",
                   help="""Label file or directory; inferred if empty. If the trask is traditional, 
                           you should provide the directory of the train/val/test dataset so it is inferred.""")

    p.add_argument("--tokenizer_name", type=str, default="",
                   help="Tokenizer name/path; defaults to model_name_or_path.")

    p.add_argument("--config_name", type=str, default="",
                   help="Model config name/path; defaults to model_name_or_path.")

    p.add_argument("--cache_dir", type=str, default="",
                   help="Cache directory for models and features.")

    p.add_argument("--max_seq_length", type=int, default=128,
                   help="Maximum input sequence length.")

    p.add_argument("--do_eval", action="store_true",
                   help="Run evaluation on validation set.")

    p.add_argument("--do_predict", action="store_true",
                   help="Run prediction on test set.")

    p.add_argument("--per_gpu_eval_batch_size", type=int, default=48,
                   help="Evaluation batch size per GPU/CPU.")

    p.add_argument("--eval_all_checkpoints", action="store_true",
                   help="Evaluate all checkpoints under output_dir.")

    p.add_argument("--no_cuda", action="store_true",
                   help="Disable CUDA.")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    p.add_argument("--fp16", action="store_true",
                   help="Use mixed precision.")

    p.add_argument("--fp16_opt_level", type=str, default="O1", 
                   help="Apex AMP level (if fp16).")

    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for distributed evaluation.")

    p.add_argument("--task_formulation", type=str, default="knowledge_guided",
                   choices=["traditional", "knowledge_guided"],
                   help="Whether to use traditional (BIO-n) or knowledge-guided (KA BIO) formulation.")

    p.add_argument("--knowledge_type", type=str, default="question",
                   choices=["question", "entity_type", "definition", "example", "all"],
                   help="Knowledge type config key under configs/knowledge_type_config/<knowledge-type>.")

    p.add_argument("--dataset", type=str, default="",
                   help="Dataset config key under configs/datasets/<dataset>")

    p.add_argument("--model", type=str, default="",
                   help="Base model config key under configs/base_config/<config>")

    p.add_argument("--overwrite_cache", action="store_true",
                   help="Whether to overwrite cache.")

    p.add_argument("--do_lower_case", action="store_true", 
                   help="Whether to lowercase text before tokenization.")

    return p


def main():
    args = build_parser().parse_args()

    # Merge YAML into args
    args = apply_layered_yaml_to_args(
        args,
        base_model_attr="model",
        dataset_attr="dataset",
        knowledge_type_attr="knowledge_type",
        extra_attr="config",
        config_type="evaluation",
    )

    # Knowledge type is traditional whenever the task formulation is traditional
    if args.task_formulation == "traditional":
        args.knowledge_type = "traditional"

        # Validate AFTER merge for traditional tasks
        _require(args, ["data_dir", "model_type", "model_name_or_path", "output_dir", "labels_trad"])
    else:
        # Validate AFTER merge for knowledge augmented tasks
        _require(args, ["data_dir", "model_type", "model_name_or_path", "output_dir"])

    if not (os.path.exists(args.output_dir) and os.listdir(args.output_dir)):
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed, args.n_gpu)

    labels = get_labels(args.labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    _, tokenizer, model = load_model_tokenizer(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        num_labels=len(labels),
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        do_lower_case=args.do_lower_case,
    )
    model.to(args.device)
    logger.debug("Evaluation parameters %s", args)

    evaluate_data_dirs = []
    if args.dataset_to_evaluate:
        evaluate_data_dirs = args.dataset_to_evaluate

    base_data_dir = args.data_dir
    for data_dir in tqdm(evaluate_data_dirs, desc="Data_Directory:"):
        dataset_test = data_dir

        data_dir=os.path.join(base_data_dir, data_dir)
        args.data_dir = data_dir

        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")

if __name__ == "__main__":
    main()
