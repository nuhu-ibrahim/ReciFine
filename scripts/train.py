from __future__ import annotations

import argparse
import glob
import logging
import os
import torch
from torch.nn import CrossEntropyLoss

from recifine.config import apply_layered_yaml_to_args
from recifine.data.labels import get_labels
from recifine.models.hf_wrappers import load_model_tokenizer
from recifine.training.train import train
from recifine.training.eval import evaluate
from recifine.utils.seed import set_seed
from recifine.utils.require_params import _require
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--do_train", action="store_true", default=None,
                help="Whether to run training.")

    p.add_argument("--evaluate_during_training", action="store_true", default=None,
                help="Whether to run evaluation while training.")

    p.add_argument("--per_gpu_train_batch_size", default=12, type=int,
                help="Training batch size per GPU/CPU.")

    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                help="Number of updates steps to accumulate before backward pass.")

    p.add_argument("--learning_rate", default=5e-5, type=float,
                help="Initial learning rate for optimizer.")

    p.add_argument("--weight_decay", default=0.0, type=float,
                help="Weight decay for optimizer.")

    p.add_argument("--adam_epsilon", default=1e-8, type=float,
                help="Epsilon value for Adam optimizer.")

    p.add_argument("--max_grad_norm", default=1.0, type=float,
                help="Maximum gradient norm for clipping.")

    p.add_argument("--num_train_epochs", default=30.0, type=float,
                help="Total number of training epochs.")

    p.add_argument("--max_steps", default=-1, type=int,
                help="If > 0, overrides num_train_epochs with total training steps.")

    p.add_argument("--warmup_steps", default=0, type=int,
                help="Number of warmup steps for learning rate scheduler.")

    p.add_argument("--logging_steps", type=int, default=5000,
                help="Log training metrics every X steps.")

    p.add_argument("--save_steps", type=int, default=5000,
                help="Save model checkpoint every X steps.")

    p.add_argument("--overwrite_output_dir", action="store_true", default=None,
                help="Overwrite the content of the output directory.")

    p.add_argument("--server_ip", type=str, default="",
                help="IP address for distributed training server.")

    p.add_argument("--server_port", type=str, default="",
                help="Port for distributed training server.")

    p.add_argument("--data_dir", type=str, default=None,
                help="Base data directory containing train/val/test splits.")

    p.add_argument("--output_dir", type=str, default=None,
                help="Directory to save trained model.")

    p.add_argument("--model_type", type=str, default=None,
                choices=["recipebert", "reciperoberta"], help="Model architecture key.")

    p.add_argument("--model_name_or_path", type=str, default=None,
                help="HF model name or local checkpoint path.")

    p.add_argument("--labels", type=str, default=None,
                help="""Label file or directory; inferred if empty. If the trask is traditional, 
                       you should provide the directory of the train/val/test dataset so it is inferred.""")

    p.add_argument("--tokenizer_name", type=str, default="",
                help="Tokenizer name/path; defaults to model_name_or_path.")

    p.add_argument("--config_name", type=str, default="",
                help="Model config name/path; defaults to model_name_or_path.")

    p.add_argument("--cache_dir", type=str, default=None,
                help="Cache directory for models and features.")

    p.add_argument("--max_seq_length", type=int, default=128,
                help="Maximum input sequence length.")

    p.add_argument("--do_eval", action="store_true",
                help="Run evaluation on validation set.")

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


def main(*, caller: str | None = None):
    args = build_parser().parse_args()

    if caller == "finetune":
        _require(args, ["model_name_or_path"])
        config_type = "finetuning"
    else:
        config_type = "training"

    # Merge YAML into args
    args = apply_layered_yaml_to_args(
        args,
        base_model_attr="model",
        dataset_attr="dataset",
        knowledge_type_attr="knowledge_type",
        extra_attr="config",
        config_type=config_type,
    )

    # Knowledge type is traditional whenever the task formulation is traditional
    if args.task_formulation == "traditional":
        args.knowledge_type = "traditional"

        # Validate AFTER merge for traditional tasks
        _require(args, ["data_dir", "model_type", "model_name_or_path", "output_dir", "labels_trad"])
    else:
        # Validate AFTER merge for knowledge augmented tasks
        _require(args, ["data_dir", "model_type", "model_name_or_path", "output_dir"])

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

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
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args.seed, args.n_gpu)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    _, tokenizer, model = load_model_tokenizer(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        num_labels=num_labels,
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
        do_lower_case=args.do_lower_case,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    logger.debug("Training parameters %s", args)

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", args.output_dir)

            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        from transformers import WEIGHTS_NAME
        from recifine.models.hf_wrappers import resolve_model_classes

        config_class, model_class, tokenizer_class = resolve_model_classes(args.model_type)

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
            if global_step:
                result = {f"{global_step}_{k}": v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            for key in sorted(results.keys()):
                writer.write(f"{key} = {results[key]}\n")

if __name__ == "__main__":
    main()
