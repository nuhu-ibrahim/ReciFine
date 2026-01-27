from __future__ import annotations

import argparse
import logging

from recifine.inferencing.inference import ReciFineNER

import json

logger = logging.getLogger(__name__)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--text", type=str, required=True,
                   help="Raw input text to run inference on.")

    p.add_argument("--dataset", type=str, default="recifinegold",
                   help="Dataset configuration key (e.g., recifinegold).")

    p.add_argument("--model", type=str, default="recipebert",
                   help="Base model configuration key under configs/base_config.")

    p.add_argument("--task_formulation",
                   choices=["traditional", "knowledge_guided"], default="knowledge_guided",
                   help="Whether to use traditional (BIO-n) or knowledge-guided (KA BIO) formulation.")

    p.add_argument("--knowledge_type",
                   choices=["question", "entity_type", "definition", "example", "all"], default="question",
                   help="Knowledge type used when task_formulation is knowledge_guided.")

    p.add_argument("--entity_type", type=str, default=None,
                   help="Specific entity type to query in knowledge-guided mode.")

    p.add_argument("--model_name_or_path", type=str, default=None,
                   help="HF model name or local checkpoint path.")

    p.add_argument("--device", type=str, default=None,
                   help="Device to run on (e.g., cuda, cpu, or cuda:0).")

    p.add_argument("--no_cuda", action="store_true",
                   help="Disable CUDA.")

    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for distributed evaluation.")

    p.add_argument("--return_tokens", action="store_true",
                   help="If set, return token-level predictions in addition to spans.")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")

    p.add_argument("--per_gpu_inf_batch_size", type=int, default=48,
                   help="Inference batch size per GPU/CPU.")

    p.add_argument("--do_lower_case", action="store_true", default=None,
                   help="Whether to lowercase text before tokenization.")

    p.add_argument("--labels", default=None, type=str,
                   help="Label file or directory; inferred if empty.")

    p.add_argument("--config_name", default="", type=str,
                   help="Model config name/path; defaults to model_name_or_path.")

    p.add_argument("--tokenizer_name", default="", type=str,
                   help="Tokenizer name/path; defaults to model_name_or_path.")

    p.add_argument("--cache_dir", default=None, type=str,
                   help="Cache directory for models and features.")

    p.add_argument("--max_seq_length", type=int, default=256,
                   help="Maximum input sequence length.")

    p.add_argument("--model_type", type=str, default=None,
                   choices=["recipebert", "reciperoberta"], help="Model architecture key.")

    return p


def main():
    logging.basicConfig(level=logging.INFO)

    args = build_parser().parse_args()

    ner = ReciFineNER(
        dataset=args.dataset,
        model=args.model,
        task_formulation=args.task_formulation,
        knowledge_type=args.knowledge_type,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
    )
    
    out = ner.process_text(args.text, entity_type=args.entity_type, return_tokens=args.return_tokens)

    logger.info(out)

if __name__ == "__main__":
    main()
