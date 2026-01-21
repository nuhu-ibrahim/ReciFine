from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
Sentence = List[Tuple[str, str]]


def _read_conll_sentences(path: str) -> List[Sentence]:
    sents: List[Sentence] = []
    cur: Sentence = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                tok, tag = parts[0].strip(), parts[1].strip()
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"Bad line (expected TOKEN<TAB>TAG): {raw!r}")
                tok, tag = parts[0].strip(), parts[1].strip()

            cur.append((tok, tag))

    if cur:
        sents.append(cur)
    return sents


def _identify_entities(sentence: Sentence) -> set[str]:
    ents = set()
    for _, tag in sentence:
        if tag.startswith("B-"):
            ents.add(tag[2:])
    return ents


def _extract_entities(sentence: Sentence, wanted_tags: List[str]) -> tuple[List[str], str]:
    wanted = set(wanted_tags)
    entities: List[str] = []
    current: List[str] = []
    text_tokens: List[str] = []

    for tok, tag in sentence:
        text_tokens.append(tok)

        if tag.startswith("B-") and tag[2:] in wanted:
            if current:
                entities.append(" ".join(current))
            current = [tok]
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

    return entities, " ".join(text_tokens)


def _split_sentences(sentences: List[Sentence], split_ratios: Dict[str, float], seed: int) -> Dict[str, List[Sentence]]:
    random.seed(seed)
    split_ratios = split_ratios

    ent2sents: Dict[str, List[Sentence]] = defaultdict(list)
    for sent in sentences:
        for ent in _identify_entities(sent):
            ent2sents[ent].append(sent)

    used = set()
    splits: Dict[str, List[Sentence]] = {"train": [], "val": [], "test": []}

    for ent, ent_sents in ent2sents.items():
        random.shuffle(ent_sents)
        train_sz = int(len(ent_sents) * split_ratios.get("train", 0.8))
        val_sz = int(len(ent_sents) * split_ratios.get("val", 0.1))

        for sent in ent_sents:
            key = tuple(tuple(x) for x in sent)
            if key in used:
                continue

            if len(splits["train"]) < train_sz:
                splits["train"].append(sent)
            elif len(splits["val"]) < val_sz:
                splits["val"].append(sent)
            else:
                splits["test"].append(sent)

            used.add(key)

    for sent in sentences:
        key = tuple(tuple(x) for x in sent)
        if key not in used:
            splits["train"].append(sent)
            used.add(key)

    for k in splits:
        random.shuffle(splits[k])

    return splits


def _ensure_writable(path: str, overwrite: bool) -> None:
    if overwrite:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")


def _write_ka_jsonl(sentences: List[Sentence], entity_groups: List[Dict], out_path: str, qid_start: int) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    qid = qid_start

    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for eg in entity_groups:
                answers, text = _extract_entities(sent, eg["tags"])
                if not answers:
                    continue

                row = {
                    "qid": str(qid),
                    "text": text,
                    "entity_type": eg["type"],
                    "entity_group": eg["group"],
                    "group_definition": eg["group_definition"],
                    "definition": eg["definition"],
                    "example": eg["examples"],
                    "question": eg["question"],
                    "answer": answers,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qid += 1

    return qid


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
            merged_json = {"qid": qid, "text": text, "traditional": tags}
            f.write(json.dumps(merged_json, ensure_ascii=False) + "\n")


def preprocess_finer_ka(args, seed: int = 42, overwrite: bool = False) -> None:
    input_path = args.original_dataset
    out_root = args.data_dir_ka
    test_folders = args.dataset_to_evaluate_ka
    entity_groups = args.entity_groups
    split_ratios = getattr(args, "split_ratios", None) or {"train": 0.8, "val": 0.1, "test": 0.1}

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"FINER input not found: {input_path}")

    os.makedirs(out_root, exist_ok=True)

    train_path = os.path.join(out_root, "train.jsonl")
    val_path = os.path.join(out_root, "val.jsonl")
    test_paths = [os.path.join(out_root, folder, "test.jsonl") for folder in test_folders]

    _ensure_writable(train_path, overwrite)
    _ensure_writable(val_path, overwrite)
    for tp in test_paths:
        _ensure_writable(tp, overwrite)

    logger.info("Reading FINER: %s", input_path)
    sentences = _read_conll_sentences(input_path)
    logger.info("Loaded %d sentences", len(sentences))

    splits = _split_sentences(sentences, split_ratios, seed=seed)
    logger.info("Sentence splits: train=%d val=%d test=%d", len(splits["train"]), len(splits["val"]), len(splits["test"]))

    qid = 1
    qid = _write_ka_jsonl(splits["train"], entity_groups, train_path, qid_start=qid)
    qid = _write_ka_jsonl(splits["val"], entity_groups, val_path, qid_start=qid)
    for tp in test_paths:
        _ = _write_ka_jsonl(splits["test"], entity_groups, tp, qid_start=qid)

    logger.info("Wrote Knowledge Augmented: %s", train_path)
    logger.info("Wrote Knowledge Augmented: %s", val_path)
    for tp in test_paths:
        logger.info("Wrote Knowledge Augmented: %s", tp)


def preprocess_finer_trad(args, seed: int = 42, overwrite: bool = False) -> None:
    in_root = getattr(args, "data_dir_ka", None)
    out_root = getattr(args, "data_dir_trad", None)
    if not in_root or not out_root:
        raise ValueError("preprocess_finer_trad requires args.data_dir_ka and args.data_dir_trad")

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

    trad_test_folders = getattr(args, "dataset_to_evaluate_trad", None)

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
