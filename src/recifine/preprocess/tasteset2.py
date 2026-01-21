from __future__ import annotations

import ast
import csv
import json
import logging
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _find_line_by_span(recipe_text: str, span: str) -> tuple[int | None, str | None]:
    lines = recipe_text.split("\n")
    char_count = 0

    span_list = ast.literal_eval(span)
    for i, line in enumerate(lines, start=1):
        line_start = char_count
        line_end = char_count + len(line)

        if line_start <= span_list[0][0] < line_end:
            return i, line

        char_count += len(line) + 1

    return None, None


def _get_entity_group(entity_type: str, config: Dict[str, Any]) -> Dict[str, Any] | None:
    for group in config["entity_groups"]:
        if entity_type in group["tags"]:
            return group
    return None


def _split_sentences_with_entity_balance(
    sentences: List[Dict[str, Any]],
    split_ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    random.seed(seed)

    entity_to_sentences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sentences:
        entity_to_sentences[s["entity_type"]].append(s)

    train_texts: set[str] = set()
    val_texts: set[str] = set()
    test_texts: set[str] = set()

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for entity, entity_sents in entity_to_sentences.items():
        train_size = int(len(entity_sents) * split_ratios.get("train", 0.8))
        val_size = int(len(entity_sents) * split_ratios.get("val", 0.1))
        test_size = int(len(entity_sents) * split_ratios.get("test", 0.1))

        logger.debug(
            "Type=%s Total=%d Train=%d Val=%d Test=%d",
            entity, len(entity_sents), train_size, val_size, test_size,
        )

        train_local: List[str] = []
        val_local: List[str] = []
        test_local: List[str] = []

        for sent in entity_sents:
            sentence_hashable = sent["text"]  

            if (
                (len(train_local) < train_size and sentence_hashable not in val_texts and sentence_hashable not in test_texts)
                or (sentence_hashable in train_texts)
            ):
                splits["train"].append(sent)
                train_texts.add(sentence_hashable)
                train_local.append(sentence_hashable)

            elif (
                (len(val_local) < val_size and sentence_hashable not in train_texts and sentence_hashable not in test_texts)
                or (sentence_hashable in val_texts)
            ):
                splits["val"].append(sent)
                val_texts.add(sentence_hashable)
                val_local.append(sentence_hashable)

            elif (
                (len(test_local) < test_size and sentence_hashable not in train_texts and sentence_hashable not in val_texts)
                or (sentence_hashable in test_texts)
            ):
                splits["test"].append(sent)
                test_texts.add(sentence_hashable)
                test_local.append(sentence_hashable)

            else:
                logger.debug(
                    "Unassigned sentence (Type=%s). ValCond1=%s ValCond2=%s TestCond1=%s TestCond2=%s Sentence=%s",
                    entity,
                    (len(val_local) < val_size and sentence_hashable not in train_texts and sentence_hashable not in test_texts),
                    (sentence_hashable in val_texts),
                    (len(test_local) < test_size and sentence_hashable not in train_texts and sentence_hashable not in val_texts),
                    (sentence_hashable in test_texts),
                    sent,
                )

        logger.debug(
            "Type=%s Assigned Train=%d Val=%d Test=%d",
            entity, len(train_local), len(val_local), len(test_local),
        )

    for k in splits:
        random.shuffle(splits[k])

    logger.debug(
        "Total sentences=%d Split sizes: train=%d val=%d test=%d",
        len(sentences), len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    return splits


def _ensure_writable(path: str, overwrite: bool) -> None:
    if overwrite:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")


def _write_jsonl_rows(rows: List[Dict[str, Any]], out_path: str, overwrite: bool) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _ensure_writable(out_path, overwrite)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl_group_by_qid_fallback(input_path: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            grouped[str(obj["qid"])].append(obj)
    return grouped


def _build_bio_tags(text: str, annotations: list[dict[str, Any]]) -> list[str]:
    tokens = text.split()
    tags = ["O"] * len(tokens)

    for ann in annotations:
        entity = str(ann["entity_type"]).upper()
        for answer in ann.get("answer", []):
            answer_tokens = str(answer).split()
            for i in range(len(tokens) - len(answer_tokens) + 1):
                if tokens[i : i + len(answer_tokens)] == answer_tokens:
                    if tags[i] == "O":
                        tags[i] = f"B-{entity}"
                        for j in range(1, len(answer_tokens)):
                            if tags[i + j] == "O":
                                tags[i + j] = f"I-{entity}"
    return tags


def _merge_jsons_by_qid_fallback(input_path: str, output_file: str, overwrite: bool = False) -> None:
    grouped_data = _load_jsonl_group_by_qid_fallback(input_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if (not overwrite) and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_file} (use --overwrite)")

    with open(output_file, "w", encoding="utf-8") as f:
        for qid, entries in grouped_data.items():
            text = entries[0]["text"]
            tags = _build_bio_tags(text, entries)
            merged = {"qid": qid, "text": text, "traditional": tags}
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")


def preprocess_tasteset2_ka(args, seed: int = 42, overwrite: bool = False) -> None:
    input_csv = args.original_dataset
    # config = args.config
    out_root = args.data_dir_ka
    test_folders = getattr(args, "dataset_to_evaluate_ka", None) or []
    entity_groups = args.entity_groups

    if isinstance(input_csv, (list, tuple)):
        if len(input_csv) != 1:
            raise ValueError(f"TasteSet2 expects a single CSV path, got {len(input_csv)} paths: {input_csv}")
        input_csv = input_csv[0]

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"CSV input not found: {input_csv}")

    if not test_folders:
        raise ValueError("dataset_to_evaluate_ka is required (to define test output folders).")

    split_ratios = getattr(args, "split_ratios", None)

    os.makedirs(out_root, exist_ok=True)

    train_path = os.path.join(out_root, "train.jsonl")
    val_path = os.path.join(out_root, "val.jsonl")
    test_paths = [os.path.join(out_root, folder, "test.jsonl") for folder in test_folders]

    _ensure_writable(train_path, overwrite)
    _ensure_writable(val_path, overwrite)
    for tp in test_paths:
        _ensure_writable(tp, overwrite)

    rows: List[Dict[str, Any]] = []

    with open(input_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)

        recipe_id = 1
        for row in reader:
            if len(row) < 2:
                continue

            recipe_text, annotations = row[0], row[1]
            entities = json.loads(annotations)

            for entity in entities:
                entity_type = entity["type"]
                entity_text = entity["entity"]
                entity_span = entity["span"]

                qid, recipe_line = _find_line_by_span(recipe_text, entity_span)
                entity_group = _get_entity_group(entity_type, {"entity_groups": entity_groups})

                if entity_group:
                    rows.append(
                        {
                            "qid": str(recipe_id) + "_" + str(qid),
                            "text": recipe_line,
                            "entity_type": entity_group["type"],
                            "entity_group": entity_group["group"],
                            "group_definition": entity_group["group_definition"],
                            "definition": entity_group["definition"],
                            "example": entity_group["examples"],
                            "question": entity_group["question"],
                            "answer": [entity_text],
                        }
                    )
                else:
                    logger.debug("Unknown entity_type=%s (no matching group).", entity_type)

            recipe_id += 1

    splits = _split_sentences_with_entity_balance(rows, split_ratios=split_ratios, seed=seed)

    _write_jsonl_rows(splits["train"], train_path, overwrite=overwrite)
    _write_jsonl_rows(splits["val"], val_path, overwrite=overwrite)
    for tp in test_paths:
        _write_jsonl_rows(splits["test"], tp, overwrite=overwrite)

    logger.info("Wrote Knowledge Augmented: %s", train_path)
    logger.info("Wrote Knowledge Augmented: %s", val_path)
    for tp in test_paths:
        logger.info("Wrote Knowledge Augmented: %s", tp)


def preprocess_tasteset2_trad(args, seed: int = 42, overwrite: bool = False) -> None:
    in_root = getattr(args, "data_dir_ka", None)
    out_root = getattr(args, "data_dir_trad", None)
    if not in_root or not out_root:
        raise ValueError("preprocess_tasteset2_trad requires args.data_dir_ka and args.data_dir_trad")

    in_train = os.path.join(in_root, "train.jsonl")
    in_val = os.path.join(in_root, "val.jsonl")

    if not os.path.isfile(in_train):
        raise FileNotFoundError(f"Missing KA train.jsonl: {in_train}")
    if not os.path.isfile(in_val):
        raise FileNotFoundError(f"Missing KA val.jsonl: {in_val}")

    ka_test_folders = getattr(args, "dataset_to_evaluate_ka", None) or []
    if not ka_test_folders:
        raise ValueError("dataset_to_evaluate_ka is required for TRAD test generation.")

    canonical_ka_test_folder = ka_test_folders[0]
    in_test = os.path.join(in_root, canonical_ka_test_folder, "test.jsonl")
    if not os.path.isfile(in_test):
        raise FileNotFoundError(f"Missing KA test.jsonl at: {in_test}")

    trad_test_folders = getattr(args, "dataset_to_evaluate_trad", None) or []
    if not trad_test_folders:
        raise ValueError("dataset_to_evaluate_trad is required (needs at least one TRAD test folder).")

    os.makedirs(out_root, exist_ok=True)

    out_train = os.path.join(out_root, "train.jsonl")
    out_val = os.path.join(out_root, "val.jsonl")

    _merge_jsons_by_qid_fallback(in_train, out_train, overwrite=overwrite)
    _merge_jsons_by_qid_fallback(in_val, out_val, overwrite=overwrite)

    grouped_data = _load_jsonl_group_by_qid_fallback(in_test)
    merged_jsons: list[dict[str, Any]] = []
    for qid, entries in grouped_data.items():
        text = entries[0]["text"]
        tags = _build_bio_tags(text, entries)
        merged_jsons.append({"qid": qid, "text": text, "traditional": tags})

    logger.info("Wrote Traditional: %s", out_train)
    logger.info("Wrote Traditional: %s", out_val)

    for folder in trad_test_folders:
        out_test = os.path.join(out_root, folder, "test.jsonl")
        os.makedirs(os.path.dirname(out_test), exist_ok=True)

        if (not overwrite) and os.path.exists(out_test) and os.path.getsize(out_test) > 0:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_test} (use --overwrite)")

        with open(out_test, "w", encoding="utf-8") as f:
            for item in merged_jsons:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info("Wrote Traditional: %s", out_test)
