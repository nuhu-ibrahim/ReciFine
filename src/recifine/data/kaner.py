from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: List[str]
    counts: Optional[int] = None
    orig_row: Optional[Dict[str, Any]] = None


@dataclass
class InputFeatures:
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]
    label_ids: List[int]
    counts: Optional[int] = None


class DataProcessor:
    def get_train_examples(self, data_dir: str):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir: str):
        raise NotImplementedError()

    def get_inf_examples(self, data: Dict[str, Any]):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_jsonlines(cls, input_file: str) -> List[Dict[str, Any]]:
        """Reads a jsonlines with KANER format."""
        with jsonlines.open(input_file) as nerf:
            nerdata = [line for line in nerf]
        return nerdata


def get_matched(answer_tokens: List[List[str]], ind: int, words_tokens: List[str]) -> Tuple[bool, int]:
    matched = False
    curr_match = False
    curr_ans_len = -1
    for ans_i in range(len(answer_tokens)):
        curr_match = False
        curr_ans_len = len(answer_tokens[ans_i])
        for token_A, token_B in zip(answer_tokens[ans_i], words_tokens[ind:ind + curr_ans_len]):
            if token_A == token_B:
                curr_match = True
            else:
                curr_match = False
                break
        if curr_match:
            matched = True
            break
    return curr_match, curr_ans_len


def generate_bio(text: str, answers: List[str]) -> List[str]:
    words_tokens = text.lower().split(" ")
    answer_tokens = [answer.lower().split(" ") for answer in answers]
    tagged_tokens: List[str] = []
    ind = 0

    while len(tagged_tokens) <= len(words_tokens):
        matched, len_ans = get_matched(answer_tokens, ind, words_tokens)
        if matched:
            tagged_tokens.append("B-ANS")
            num_of_is = ["I-ANS"] * (len_ans - 1)
            tagged_tokens.extend(num_of_is)
            ind += len_ans
        else:
            ind += 1
            tagged_tokens.append("O")

    tagged_tokens.pop(-1)

    # post-process to replace anything before "::" with "O"
    cutoff = words_tokens.index("::") + 1
    tagged_tokens[:cutoff] = ["O"] * cutoff

    if len(words_tokens) != len(tagged_tokens):
        print(text, words_tokens, tagged_tokens, len(words_tokens) - len(tagged_tokens))

    return tagged_tokens


class KANerProcessor(DataProcessor):
    def get_train_examples(self, data_dir: str, knowledge_type: str = "question") -> List[InputExample]:
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "train.jsonl")), "train", knowledge_type
        )

    def get_dev_examples(self, data_dir: str, knowledge_type: str = "question") -> List[InputExample]:
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "val.jsonl")), "dev", knowledge_type
        )

    def get_test_examples(self, data_dir: str, knowledge_type: str = "question") -> List[InputExample]:
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "test.jsonl")), "test", knowledge_type
        )

    def get_inf_examples(self, data: List[Dict[str, Any]], knowledge_type: str = "question") -> List[InputExample]:
        return self._create_examples(
            data, "test", knowledge_type
        )

    def get_labels(self) -> List[str]:
        return ["B-ANS", "I-ANS", "O"]

    def _create_examples(self, lines: List[Dict[str, Any]], set_type: str, knowledge_type: str = "question") -> List[InputExample]:
        examples: List[InputExample] = []
        for _, js in tqdm(list(enumerate(lines)), desc="Reading data"):
            guid = js["qid"]
            txknowledge_type = ""

            if knowledge_type == "all":
                exmpls = [tup[0] for tup in js["example"]]
                extxt = " ".join(exmpls)
                txknowledge_type = js["question"] + " : " + js["definition"] + " : " + extxt
            elif knowledge_type == "what_type_q":
                txknowledge_type = "what " + js["entity_type"] + " ? "
            elif knowledge_type == "what_type":
                txknowledge_type = "what " + js["entity_type"]
            elif knowledge_type == "what":
                txknowledge_type = "what "
            else:
                txknowledge_type = js[knowledge_type]

            if "definiton" in knowledge_type:
                txknowledge_type = js["entity_type"] + " : " + js[knowledge_type]

            if knowledge_type == "example":
                exmpls = [tup[0] for tup in txknowledge_type]
                extxt = " ".join(exmpls)
                txknowledge_type = js["entity_type"] + " : " + extxt

            if knowledge_type == "traditional":
                words = js["text"]
                label = js["traditional"]
                words_list = words.split(" ")

                answer_len = sum(1 for lab in label if lab != "O")
                counts = 1 if answer_len > 0 else 0

                examples.append(InputExample(guid=guid, words=words_list, labels=label, counts=counts, orig_row=js))
            else:
                words = txknowledge_type + " :: " + js["text"]
                label = generate_bio(words, js["answer"])
                words_list = words.split(" ")

                answer_len = len(js["answer"])
                counts = 1 if answer_len > 0 else 0

                examples.append(InputExample(guid=guid, words=words_list, labels=label, counts=counts, orig_row=js))

        return examples


def read_examples_from_file(data_dir: str, mode: str, knowledge_type: str = "question") -> List[InputExample]:
    processor = KANerProcessor()
    mode_examples = {
        "train": processor.get_train_examples,
        "dev": processor.get_dev_examples,
        "test": processor.get_test_examples,
    }
    return mode_examples[mode](data_dir, knowledge_type)


def read_example_from_data(data: List[Dict[str, Any]], knowledge_type: str = "question") -> List[InputExample]:
    processor = KANerProcessor()

    return processor.get_inf_examples(data, knowledge_type)
