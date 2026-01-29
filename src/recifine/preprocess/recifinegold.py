from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
Sentence = List[Tuple[str, str]]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def build_label_map_from_entity_groups(entity_groups: List[Dict[str, Any]]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for eg in entity_groups:
        tags = eg.get("tags", [])
        if not tags:
            continue
        short_tag = str(tags[0])

        doccano_labels = eg.get("doccano_labels", []) or eg.get("labels", [])  # allow either key
        if not doccano_labels:
            continue

        for lbl in doccano_labels:
            lbl = str(lbl)
            if lbl in label_map and label_map[lbl] != short_tag:
                raise ValueError(
                    f"Duplicate doccano label {lbl!r} mapped to multiple tags: "
                    f"{label_map[lbl]!r} and {short_tag!r}"
                )
            label_map[lbl] = short_tag

    if not label_map:
        raise ValueError("No label mappings found. Add doccano_labels to entity_groups in config.")
    return label_map


def tokenize_text(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

def _ensure_writable(path: str, overwrite: bool) -> None:
    if overwrite:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")


def _build_char_tag_array(text: str, labels: List[Any], label_map: Dict[str, str]) -> List[str]:
    token_labels = ["O"] * len(text)

    for ann in labels:
        if isinstance(ann, dict):
            start = ann.get("start")
            end = ann.get("end")
            raw_label = ann.get("tag")

        elif isinstance(ann, (list, tuple)) and len(ann) == 3:
            start, end, raw_label = ann

        else:
            continue

        tag = label_map.get(str(raw_label), "O")
        if tag == "O":
            continue

        start = max(0, min(int(start), len(text)))
        end = max(0, min(int(end), len(text)))

        for i in range(start, end):
            token_labels[i] = tag

    return token_labels


def _suffix_bio_for_token_at_charpos(char_tags: List[str], idx: int) -> str:
    if idx < 0 or idx >= len(char_tags):
        return "O"

    label = char_tags[idx]
    if label == "O":
        return "O"

    prev_label = char_tags[idx - 1] if idx - 1 >= 0 else "O"
    return f"{label}-B" if prev_label != label else f"{label}-I"


def _jsonl_entry_to_sentences_suffix_bio(
    text: str,
    labels: List[List[Any]],
    label_map: Dict[str, str],
) -> List[Sentence]:
    tokens = tokenize_text(text)
    char_tags = _build_char_tag_array(text, labels, label_map)

    sentences: List[Sentence] = []
    current: Sentence = []

    current_pos = 0
    for tok in tokens:
        idx = text.find(tok, current_pos)
        if idx == -1:
            continue
        current_pos = idx + len(tok)

        tag = _suffix_bio_for_token_at_charpos(char_tags, idx)
        current.append((tok, tag))

        if tok in [".", "!", "?"]:
            if current:
                sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences


def _read_original_jsonl_sentences(jsonl_path: str, label_map: Dict[str, str]) -> List[Sentence]:
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Input recifinegold jsonl not found: {jsonl_path}")

    all_sentences: List[Sentence] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d in %s", line_num, jsonl_path)
                continue

            text = entry.get("text", "")
            labels = entry.get("label", []) or []
            if not isinstance(text, str) or not text:
                continue

            all_sentences.extend(_jsonl_entry_to_sentences_suffix_bio(text, labels, label_map))

    return all_sentences


def _identify_entities_suffix_bio(sentence: Sentence) -> set[str]:
    return {tag[:-2] for _, tag in sentence if tag.endswith("-B")}


def _extract_entities_suffix_bio(sentence: Sentence, wanted_tags: List[str]) -> tuple[List[str], str]:
    wanted = set(wanted_tags)
    entities: List[str] = []
    current: List[str] = []
    text_tokens: List[str] = []

    for tok, tag in sentence:
        text_tokens.append(tok)

        if tag.endswith("-B") and tag[:-2] in wanted:
            if current:
                entities.append(" ".join(current))
            current = [tok]
        elif tag.endswith("-I") and current:
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

    return entities, " ".join(text_tokens)


def _split_sentences_entity_balanced_suffix_bio(
    sentences: List[Sentence],
    split_ratios: Dict[str, float] | None,
    seed: int,
) -> Dict[str, List[Sentence]]:
    random.seed(seed)
    if not split_ratios:
        split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    ent2sents: Dict[str, List[Sentence]] = defaultdict(list)
    for sent in sentences:
        for ent in _identify_entities_suffix_bio(sent):
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


def _write_erfg_ka_jsonl(
    sentences: List[Sentence],
    entity_groups: List[Dict[str, Any]],
    out_path: str,
    qid_start: int,
    overwrite: bool,
) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _ensure_writable(out_path, overwrite)

    qid = qid_start
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for eg in entity_groups:
                answers, text = _extract_entities_suffix_bio(sent, eg["tags"])
                if not answers:
                    continue

                row = {
                    "qid": str(qid),
                    "text": text,
                    "entity_type": eg["type"],
                    "entity_group": eg["group"],
                    "group_definition": eg["group_definition"],
                    "definition": eg.get("definition"),
                    "example": eg["examples"],
                    "question": eg["question"],
                    "answer": answers,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qid += 1

    return qid


def preprocess_recifinegold_ka(args, seed: int = 42, overwrite: bool = False) -> None:
    in_jsonl = args.original_dataset
    out_root = args.data_dir_ka
    test_folders = getattr(args, "dataset_to_evaluate_ka", None) or []
    entity_groups = args.entity_groups
    split_ratios = getattr(args, "split_ratios", None)

    if not isinstance(in_jsonl, str) or not in_jsonl.endswith(".jsonl"):
        raise ValueError(f"args.original_dataset must be a .jsonl file path, got: {in_jsonl!r}")
    if not os.path.isfile(in_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {in_jsonl}")
    if not test_folders:
        raise ValueError("dataset_to_evaluate_ka is required (to define test output folders).")

    label_map = build_label_map_from_entity_groups(entity_groups)

    os.makedirs(out_root, exist_ok=True)

    train_path = os.path.join(out_root, "train.jsonl")
    val_path = os.path.join(out_root, "val.jsonl")
    test_paths = [os.path.join(out_root, folder, "test.jsonl") for folder in test_folders]

    _ensure_writable(train_path, overwrite)
    _ensure_writable(val_path, overwrite)
    for tp in test_paths:
        _ensure_writable(tp, overwrite)

    logger.info("Reading RecifineGold JSONL from: %s", in_jsonl)
    sentences = _read_original_jsonl_sentences(in_jsonl, label_map=label_map)
    logger.info("Loaded %d sentences", len(sentences))

    splits = _split_sentences_entity_balanced_suffix_bio(sentences, split_ratios=split_ratios, seed=seed)
    logger.info(
        "Sentence splits: train=%d val=%d test=%d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    qid = 1
    qid = _write_erfg_ka_jsonl(splits["train"], entity_groups, train_path, qid_start=qid, overwrite=overwrite)
    qid = _write_erfg_ka_jsonl(splits["val"], entity_groups, val_path, qid_start=qid, overwrite=overwrite)
    for tp in test_paths:
        _ = _write_erfg_ka_jsonl(splits["test"], entity_groups, tp, qid_start=qid, overwrite=overwrite)

    logger.info("Wrote Knowledge Augmented: %s", train_path)
    logger.info("Wrote Knowledge Augmented: %s", val_path)
    for tp in test_paths:
        logger.info("Wrote Knowledge Augmented: %s", tp)


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


def preprocess_recifinegold_trad(args, seed: int = 42, overwrite: bool = False) -> None:
    in_root = getattr(args, "data_dir_ka", None)
    out_root = getattr(args, "data_dir_trad", None)
    if not in_root or not out_root:
        raise ValueError("preprocess_recifinegold_trad requires args.data_dir_ka and args.data_dir_trad")

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
