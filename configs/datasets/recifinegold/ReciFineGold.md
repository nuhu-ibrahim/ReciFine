# Recifine Gold Dataset

## Overview
ReciFineGold is a finely annotated recipe dataset designed to support research in **controllable recipe generation, structured recipe understanding, and information extraction from procedural cooking text**. The dataset contains step-by-step cooking instructions drawn from 500 real-world recipes, each **manually annotated** with span-level semantic labels that capture the key components of culinary actions, ingredients, tools, and states.

Each recipe in the dataset consists of:

- A unique recipe identifier (`id`)  
- A natural language cooking procedure (`text`)  
- A list of character-level annotations (`label`), where each annotation specifies:  
  - the start and end offsets of a text span, and  
  - its corresponding semantic entity type  
---

## Entity Types
The dataset defines the following semantic entity types that were adopted from the [Recipe Case Study Paper](https://www.dl.soc.i.kyoto-u.ac.jp/~tajima/papers/hmdata18yamakatawww.pdf), each representing a different aspects of the cooking process:

| Tag | Name | Definition |
|-----|------|------------|
| **F** | Food | Edible items; includes both raw ingredients and intermediate products |
| **T** | Tool | Cooking tools such as *knives, bowls and pans* |
| **D** | Duration | Time durations used in cooking (e.g., *20 minutes*) |
| **Q** | Quantity | Quantities associated with ingredients |
| **Ac** | Action by chef | Verbs denoting deliberate actions by the cook (e.g., *bring* in “Bring the mixture to a boil.”) |
| **Ac2** | Discontinuous Ac | Non-contiguous parts of compound chef actions (e.g., *to a boil* in “Bring the mixture to a boil.”) |
| **Af** | Action by food | Verbs where food is the agent (e.g., *melt, boil*) |
| **At** | Action by tool | Verbs pertaining to a tool's action (e.g., *grind, beat*) |
| **Sf** | Food state | Descriptions of food's physical state (e.g., *chopped, soft*) |
| **St** | Tool state | Descriptions of tool state or readiness (e.g., *preheated*, *greased*, *covered*) |

**Table 1: Entity types in the English Recipe Flow Graph corpus**

---
## Source and Download
### Download link:
The dataset is available for download under CC BY-NC 4.0 on HuggingFace through the link below.

[https://huggingface.co/datasets/Atnafu/Afri-MCQA](https://huggingface.co/datasets/Atnafu/Afri-MCQA)

### Download steps
1. Download the file `recifinegold.jsonl` from the link above.

---

## Expected Directory Structure

Place the `recifinegold.jsonl` file as follows:

```
configs/datasets/recifinegold/
└── data/
    └── recifinegold.jsonl
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset recifinegold
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/recifinegold/
├── data/
    └── recifinegold.jsonl
├── data_ka
    ├── train.jsonl
    ├── val.jsonl
    └── test
        └── test.jsonl
├── data_trad
    ├── train.jsonl
    ├── val.jsonl
    └── test
        └── test.jsonl
    └── paper.yaml
```

Notes:
* **data_ka** folder contains the processed data for knowledge augmented training/finetuning/prediction or evaluation using any of the 5 knowlege context types.
* **data_trad** folder contains the processed data for traditional training/finetuning/prediction or evaluation.

---

## Citation
If you use the RecifineGold dataset or the knowledge-augmented and entity-type specific architecture, please cite our papers:

```
@inproceedings{
  title   = {Knowledge Augmentation Enhances Token Classification for Recipe Understanding},
  author  = {Ibrahim, Nuhu and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year    = {2026}
}

@inproceedings{
  title   = {ReciFine: Finely Annotated Recipe Dataset for Controllable Recipe Generation},
  author  = {Ibrahim, Nuhu and Ravikumar, Rishi and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year    = {2026}
}
```
