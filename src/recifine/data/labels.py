from __future__ import annotations

import json
import os
from typing import List, Optional


def get_labels(path: Optional[str]) -> List[str]:
    if path:
        if os.path.isdir(path):
            unique_labels = set()
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.endswith(".jsonl"):
                        file_path = os.path.join(root, filename)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                obj = json.loads(line)
                                merge_tags = obj.get("traditional", [])
                                for tag in merge_tags:
                                    if tag != "O":
                                        unique_labels.add(tag)
            return ["O"] + sorted(unique_labels)
        else:
            with open(path, "r", encoding="utf-8") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
    else:
        return ["O", "I-ANS", "B-ANS"]

def get_labels_by_knowledge_type(entity_groups: List[Dict[str, Any]], knowledge_type: str) -> List[str]:
    if knowledge_type != "traditional":
        return ["O", "I-ANS", "B-ANS"]

    unique_types = {
        eg["type"].upper()
        for eg in entity_groups
        if "type" in eg
    }

    labels = ["O"]
    for t in sorted(unique_types):
        labels += [f"B-{t}", f"I-{t}"]

    return labels


def make_label_map(labels: List[str]):
    """Helper: label -> id and id -> label maps."""
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for i, lab in enumerate(labels)}
    return label2id, id2label
