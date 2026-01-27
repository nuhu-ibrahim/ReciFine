from __future__ import annotations

import logging

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


def get_repo_root() -> Path:
    here = Path(__file__).resolve()

    package_root = here.parent
    if (package_root / "configs").exists():
        return package_root

    for parent in [here] + list(here.parents):
        candidate = parent / "src" / "recifine"
        if (candidate / "configs").exists():
            return candidate

    return here.parents[2]


def resolve_path_maybe_relative(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def load_yaml_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping (dict-like).")
    return data


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_layered_config(
    *,
    repo_root: Path,
    base_model: Optional[str] = None,
    dataset: Optional[str] = None,
    knowledge_type: Optional[str] = None,
    extra_configs: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Path]]:

    loaded_files: List[Path] = []
    cfg: Dict[str, Any] = {}

    def _load_if_exists(rel_or_abs: Optional[str | Path]) -> None:
        nonlocal cfg
        if not rel_or_abs:
            return
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.exists():
            cfg = deep_merge_dicts(cfg, load_yaml_config(p))
            loaded_files.append(p)

    if base_model:
        _load_if_exists(f"configs/base_config/{base_model}.yaml")

    if dataset:
        _load_if_exists(f"configs/datasets/{dataset}/paper.yaml")

    if knowledge_type:
        _load_if_exists(f"configs/knowledge_type_config/{knowledge_type}.yaml")

    if extra_configs:
        for c in extra_configs:
            _load_if_exists(c)

    return cfg, loaded_files


def merge_config_into_namespace(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    merged = copy.copy(args)

    for k, v in (cfg or {}).items():
        if not hasattr(merged, k):
            setattr(merged, k, v)
            continue

        current = getattr(merged, k)
        if current is None:
            setattr(merged, k, v)
        elif isinstance(current, str) and current.strip() == "":
            setattr(merged, k, v)

    return merged

def get_present_arg_keys(args: argparse.Namespace) -> set[str]:
    return {
        key
        for key, value in vars(args).items()
        if value is not None
    }

def apply_layered_yaml_to_args(
    args: argparse.Namespace,
    *,
    base_model_attr: str = "model",
    dataset_attr: str = "dataset",
    knowledge_type_attr: str = "knowledge_type",
    extra_attr: str = "config",
    config_type: str = "training",
    task_formulation_attr: str = "task_formulation"
) -> argparse.Namespace:

    logging.basicConfig(level=logging.INFO)

    repo_root = get_repo_root()

    model_registry_attr = "model_registry"

    base_model = getattr(args, base_model_attr, None)
    dataset = getattr(args, dataset_attr, None)
    knowledge_type = getattr(args, knowledge_type_attr, None)
    extra = getattr(args, extra_attr, None)
    task_formulation = getattr(args, task_formulation_attr, None)
    model_name_or_path = getattr(args, "model_name_or_path", None)
    output_dir = getattr(args, "output_dir", None)

    # get keys that are present in args
    present_keys = get_present_arg_keys(args)

    extra_list: Optional[List[str]]
    if extra is None:
        extra_list = None
    elif isinstance(extra, list):
        extra_list = extra
    else:
        extra_list = [str(extra)]

    cfg, loaded_files = load_layered_config(
        repo_root=repo_root,
        base_model=base_model,
        dataset=dataset,
        knowledge_type=knowledge_type,
        extra_configs=extra_list,
    )

    args = merge_config_into_namespace(args, cfg)

    if not hasattr(args, "repo_root"):
        args.repo_root = str(repo_root)
    args._loaded_config_files = [str(p) for p in loaded_files]

    # condition that works with training/finetuning/evaluation and prediction
    if config_type == "training" or config_type == "finetuning" or config_type == "evaluation" or config_type == "prediction":

        variant_suffix = "_trad" if knowledge_type == "traditional" or task_formulation == "traditional" else "_ka"

        # Base keys you want to expose to the rest of the code/only set these keys when they are not passed from params
        variant_driven_keys = set((
            "data_dir",
            "output_dir",
            "labels",
            "dataset_to_evaluate",
            "dataset_to_predict",
        )) - present_keys

        for base_key in variant_driven_keys:
            variant_key = f"{base_key}{variant_suffix}"
            if hasattr(args, variant_key):
                val = getattr(args, variant_key)
                # Only overwrite if the variant value is meaningfully set
                if val is not None and (not isinstance(val, str) or val.strip()):
                    setattr(args, base_key, val)


        if model_name_or_path in (None, "") and (config_type == "finetuning" or config_type == "evaluation" or config_type == "prediction"):

            if task_formulation == "traditional":
                args.knowledge_type = "traditional"

            # when model name or path is not provided in cli and not configured in yaml, we use default in memory
            if getattr(args, model_registry_attr, None) in (None, "") or args.model not in args.model_registry or args.knowledge_type not in args.model_registry[args.model]:
                # set it to default in file
                args.model_name_or_path = os.path.join(args.output_dir, args.knowledge_type)
            else:
                args.model_name_or_path = args.model_registry[args.model][args.knowledge_type]["hf_repo"]

                # add model name or path in path key to ensure the path doesn't get resolved when we are using hf_repo
                present_keys.add("model_name_or_path")

        # Add the knowledge type to output directory if config output directory is being used
        if output_dir in (None, ""):
            args.output_dir = os.path.join(args.output_dir, args.knowledge_type)


        path_keys = set(("data_dir", "output_dir", "labels", "cache_dir", "config_name", "tokenizer_name"))
        if not config_type == "training":
            path_keys.add("model_name_or_path")

        path_keys = path_keys - present_keys
        for path_key in path_keys:
            if hasattr(args, path_key):
                val = getattr(args, path_key)
                if isinstance(val, str) and val.strip():
                    setattr(args, path_key, str(resolve_path_maybe_relative(val, repo_root)))

    # condition that works with dataset preprocessing
    elif config_type == "preprocessing":
        
        for path_key in set(("original_dataset", "data_dir_ka", "dataset_to_evaluate_ka", "data_dir_trad", "dataset_to_evaluate_trad")) - present_keys:
            if hasattr(args, path_key):
                val = getattr(args, path_key)
                if isinstance(val, str) and val.strip():
                    setattr(args, path_key, str(resolve_path_maybe_relative(val, repo_root)))

    # condition that works with inferencing wrapper
    elif config_type == "inference":
        if model_name_or_path in (None, ""):
            if getattr(args, model_registry_attr, None) in (None, ""):
                raise ValueError(f"model_name_or_path is empty and model_registry is not configured in {dataset} config.")
            elif args.model not in args.model_registry:
                raise ValueError(f"model_registry has no entry for model='{args.model}'. Available: {list(args.model_registry.keys())}")
        
            if task_formulation == "traditional":
                args.knowledge_type = "traditional"

            if args.knowledge_type not in args.model_registry[args.model]:
                raise ValueError(
                    f"model_registry[{args.model}] has no entry for knowledge_type='{args.knowledge_type}'. "
                    f"Available: {list(args.model_registry[args.model].keys())}"
                )

            args.model_name_or_path = args.model_registry[args.model][args.knowledge_type]["hf_repo"]

        # resolve path if provided by user  # check if it works if user provides hugging face path
        else:
            if task_formulation == "traditional":
                args.knowledge_type = "traditional"

            for path_key in set(("model_name_or_path",)) - present_keys:
                if hasattr(args, path_key):
                    val = getattr(args, path_key)
                    if isinstance(val, str) and val.strip():
                        setattr(args, path_key, str(resolve_path_maybe_relative(val, repo_root)))

    return args
