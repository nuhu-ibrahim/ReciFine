# AR (All Recipe) Dataset

## Overview

AR is a recipe ingredient named-entity recognition (NER) dataset derived from AllRecipes, annotated with food-related entities such as **ingredient names, quantities, units, states, sizes, and temperature entity types**. The original dataset is provided in tab-separated format and is converted in this project into a knowledge-guided and traditional BIO-n JSONL format suitable for training, validation, and prediction.

---

## Source and Download

### Download link:
  [https://github.com/cosylabiiit/recipe-knowledge-mining](https://github.com/cosylabiiit/recipe-knowledge-mining)

### Download steps

1. Download the file `ar_complete_dataset.tsv` from the link above.

---

## Raw Data Format

The file `ar_complete_dataset.tsv` follows the standard CoNLL BIO format:

* One token per line with each line containing:

  ```
  TOKEN<TAB>TAG
  ```
* Tags use the BIO scheme (e.g., `NAME`, `NAME`, `O`).
* Sentences are separated by blank lines.

---

## Expected Directory Structure

Place the raw dataset as follows:

```
configs/datasets/ar/
└── data/
    └── ar_complete_dataset.tsv
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset ar
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/ar/
├── data
    └── ar_complete_dataset.tsv
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

If you use the AR dataset, please cite the original dataset authors.

```
@inproceedings{diwan2020named,
  title={A Named Entity Based Approach to Model Recipes},
  author={Diwan, Nirav and Batra, Devansh and Bagler, Ganesh},
  booktitle={2020 IEEE 36th International conference on data engineering workshops (ICDEW)},
  pages={88--93},
  year={2020},
  organization={IEEE}
}
```
