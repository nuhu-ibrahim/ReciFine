from __future__ import annotations

import argparse
import logging

from recifine.config import apply_layered_yaml_to_args
from recifine.utils.require_params import _require
from recifine.preprocess import PREPROCESSORS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="", help="Dataset folder name under configs/datasets/<dataset>/")
    p.add_argument("--seed", type=int, default=42, help="Split seed.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing data ouputs.")
    return p

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = build_parser().parse_args()

    # Merge YAML into args
    args = apply_layered_yaml_to_args(
        args,
        base_model_attr="model",
        dataset_attr="dataset",
        knowledge_type_attr="knowledge_type",
        extra_attr="config",
        config_type="preprocessing"
    )

    # Validate AFTER merge
    _require(args, ["dataset_name", "original_dataset", "data_dir_ka", 
        "dataset_to_evaluate_ka", "data_dir_trad", "dataset_to_evaluate_trad", "entity_groups"])

    ds = str(args.dataset_name).lower().strip()
    ds_ka = f"{ds}_ka"
    ds_trad = f"{ds}_trad"

    missing = [
        name for name in (ds_ka, ds_trad)
        if name not in PREPROCESSORS
    ]

    if missing:
        raise ValueError(
            f"No preprocessor registered for dataset variant(s): {missing}. "
            f"Available: {sorted(PREPROCESSORS.keys())}"
        )

    logger.info("Running preprocessing for dataset='%s'", ds)
    PREPROCESSORS[ds_ka](args, seed=args.seed, overwrite=args.overwrite)
    PREPROCESSORS[ds_trad](args, seed=args.seed, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
