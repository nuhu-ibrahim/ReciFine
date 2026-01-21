from __future__ import annotations

import json
import logging
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
Sentence = List[Tuple[str, str]]


_TOKEN_PATTERN = re.compile(r"\d+/\d+-\w+|\d+/\d+|\w+(?:-\w+)*|\w+'\w+|\w+|[.,!?;:()\"/-]")
_EOS = {".", "!", "?"}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text)


def _is_end_of_sentence(tok: str) -> bool:
    return tok in _EOS


def _parse_bioc_xml_to_sentences(xml_path: str) -> List[Sentence]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows: List[Tuple[str, str]] = []
    first_document = True

    for document in root.findall("document"):
        if not first_document:
            rows.append(("", ""))
        first_document = False

        full_text_el = document.find("./infon[@key='full_text']")
        if full_text_el is None or full_text_el.text is None:
            continue

        full_text = full_text_el.text.strip()
        words = _tokenize(full_text)
        word_tags = ["O"] * len(words)

        char_positions: List[int] = []
        word_index_map: List[int] = []
        current_pos = 0
        clean_word_index = 0

        for w in words:
            char_positions.append(current_pos)
            word_index_map.append(clean_word_index)
            current_pos += len(w) + 1

            if not re.match(r"^[-/]+$", w):
                clean_word_index += 1

        for annotation in document.findall("annotation"):
            text_el = annotation.find("text")
            loc_el = annotation.find("location")
            if text_el is None or text_el.text is None or loc_el is None:
                continue

            ann_text = text_el.text.strip()

            sem_el = annotation.find("./infon[@key='semantic_tags']")
            _semantic_tags = sem_el.text.strip() if (sem_el is not None and sem_el.text is not None) else ""

            word_offset = int(loc_el.get("offset")) - 1
            char_length = int(loc_el.get("length"))

            if word_offset >= len(word_index_map):
                continue

            mapped_word_offset = word_index_map[word_offset]
            start_char = char_positions[mapped_word_offset]
            end_char = start_char + char_length

            selected: List[int] = []
            for i, pos in enumerate(char_positions):
                if start_char <= pos < end_char:
                    selected.append(i)

            extracted = words[selected[0] : selected[-1] + 1] if selected else []
            if " ".join(extracted) == ann_text:
                for i in selected:
                    word_tags[i] = "FOOD"

        for i, (w, tag) in enumerate(zip(words, word_tags)):
            if re.match(r"\d+/\d+-\w+", w):
                rows.append((w, tag))
            elif re.match(r"\d+/\d+", w):
                a, b = w.split("/")
                rows.append((a, tag))
                rows.append(("/", tag))
                rows.append((b, tag))
            elif "-" in w and len(w) > 1:
                parts = w.split("-")
                for part in parts[:-1]:
                    rows.append((part, tag))
                    rows.append(("-", tag))
                rows.append((parts[-1], tag))
            else:
                rows.append((w, tag))

            if _is_end_of_sentence(w) and (i + 1) < len(words):
                rows.append(("", ""))

    sents: List[Sentence] = []
    cur: Sentence = []
    for tok, tag in rows:
        if not tok and not tag:
            if cur:
                sents.append(cur)
                cur = []
            continue
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

    exclude_tokens = {"-LRB-", "-RRB-"}

    for tok, tag in sentence:
        if tok not in exclude_tokens:
            text_tokens.append(tok)

        if tag in wanted:
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

    return entities, " ".join(text_tokens)


def _split_sentences_simple(sentences: List[Sentence], split_ratios: Dict[str, float], seed: int) -> Dict[str, List[Sentence]]:
    random.seed(seed)

    train_sz = int(len(sentences) * split_ratios.get("train", 0.8))
    val_sz = int(len(sentences) * split_ratios.get("val", 0.1))

    splits: Dict[str, List[Sentence]] = {"train": [], "val": [], "test": []}
    for sent in sentences:
        if len(splits["train"]) < train_sz:
            splits["train"].append(sent)
        elif len(splits["val"]) < val_sz:
            splits["val"].append(sent)
        else:
            splits["test"].append(sent)

    for k in splits:
        random.shuffle(splits[k])

    return splits


def _ensure_writable(path: str, overwrite: bool) -> None:
    if overwrite:
        return
    if os.path.exists(path) and os.path.getsize(path) > 0:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --overwrite)")


def _write_ka_jsonl(sentences: List[Sentence], entity_groups: List[Dict[str, Any]], out_path: str, qid_start: int) -> int:
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
                    "definition": eg.get("definition"),
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


def preprocess_foodbase_ka(args, seed: int = 42, overwrite: bool = False) -> None:
    input_path = args.original_dataset
    out_root = args.data_dir_ka
    test_folders = args.dataset_to_evaluate_ka
    entity_groups = args.entity_groups
    split_ratios = getattr(args, "split_ratios", None)

    if isinstance(input_path, (list, tuple)):
        if len(input_path) != 1:
            raise ValueError(f"FoodBase expects a single XML path, got {len(input_path)} paths: {input_path}")
        input_path = input_path[0]

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"FoodBase XML input not found: {input_path}")

    if not test_folders:
        raise ValueError("dataset_to_evaluate_ka is required (to define test output folders).")

    os.makedirs(out_root, exist_ok=True)

    train_path = os.path.join(out_root, "train.jsonl")
    val_path = os.path.join(out_root, "val.jsonl")
    test_paths = [os.path.join(out_root, folder, "test.jsonl") for folder in test_folders]

    _ensure_writable(train_path, overwrite)
    _ensure_writable(val_path, overwrite)
    for tp in test_paths:
        _ensure_writable(tp, overwrite)

    logger.info("Reading Foodbase XML: %s", input_path)
    sentences = _parse_bioc_xml_to_sentences(input_path)
    logger.info("Loaded %d sentences from Foodbase XML", len(sentences))

    splits = _split_sentences_simple(sentences, split_ratios, seed=seed)
    logger.info(
        "Sentence splits: train=%d val=%d test=%d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    qid = 1
    qid = _write_ka_jsonl(splits["train"], entity_groups, train_path, qid_start=qid)
    qid = _write_ka_jsonl(splits["val"], entity_groups, val_path, qid_start=qid)
    for tp in test_paths:
        _ = _write_ka_jsonl(splits["test"], entity_groups, tp, qid_start=qid)

    logger.info("Wrote Knowledge Augmented: %s", train_path)
    logger.info("Wrote Knowledge Augmented: %s", val_path)
    for tp in test_paths:
        logger.info("Wrote Knowledge Augmented: %s", tp)


def preprocess_foodbase_trad(args, seed: int = 42, overwrite: bool = False) -> None:
    in_root = getattr(args, "data_dir_ka", None)
    out_root = getattr(args, "data_dir_trad", None)
    if not in_root or not out_root:
        raise ValueError("preprocess_foodbase_trad requires args.data_dir_ka and args.data_dir_trad")

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
