from __future__ import annotations

from typing import Tuple, Type

from torch.optim import AdamW

from transformers import (
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    AutoTokenizer,
)

from .bert_recipe_token import BertRecipeTokenClassifier
from .roberta_recipe_token import RobertaForTokenClassification


MODEL_CLASSES = {
    "recipebert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "reciperoberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}


def resolve_model_classes(model_type: str):
    model_type = model_type.lower()
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model_type={model_type}. Options={list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_type]


def load_model_tokenizer(
    model_type: str,
    model_name_or_path: str,
    num_labels: int,
    config_name: str = "",
    tokenizer_name: str = "",
    do_lower_case: bool = False,
) -> Tuple[object, object, object]:
    config_class, model_class, tokenizer_class = resolve_model_classes(model_type)

    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        do_lower_case=do_lower_case,
    )
    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    return config, tokenizer, model


def load_model_tokenizer_fast(
    model_type: str,
    model_name_or_path: str,
    num_labels: int,
    config_name: str = "",
    tokenizer_name: str = "",
    do_lower_case: bool = False,
) -> Tuple[object, object, object]:

    config_class, model_class, _tokenizer_class = resolve_model_classes(model_type)

    config = config_class.from_pretrained(
        config_name if config_name else model_name_or_path,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name_or_path,
        use_fast=True,                 # <-- key line
        do_lower_case=bool(do_lower_case),
    )

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )

    return config, tokenizer, model