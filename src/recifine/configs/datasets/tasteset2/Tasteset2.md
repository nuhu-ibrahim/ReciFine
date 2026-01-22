# Tasteset2 Dataset

## Overview

TASTEset2 is a recipe named-entity recognition (NER) dataset designed for fine-grained extraction of culinary entities such as **food items, quantities, units, processes, physical qualities, taste attributes**, and more from ingredient lists.

---

## Source and Download

### Download link:
  [https://github.com/taisti/TASTEset-2.0/tree/main/data](https://github.com/taisti/TASTEset-2.0/tree/main/data)

### Download steps

1. Download the file `TASTEset.csv` from the link above.

---

## Expected Directory Structure

Place the raw dataset as follows:

```
configs/datasets/tasteset2/
└── data/
    └── TASTEset.csv
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset tasteset2
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/tasteset2/
├── data
    └── TASTEset.csv
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
If you use the Tasteset2 dataset, please cite the original dataset authors.

```
@article{wroblewska2022tasteset,
  title={TASTEset--Recipe Dataset and Food Entities Recognition Benchmark},
  author={Wr{\'o}blewska, Ania and Kaliska, Agnieszka and Paw{\l}owski, Maciej and Wi{\'s}niewski, Dawid and Sosnowski, Witold and {\L}awrynowicz, Agnieszka},
  journal={arXiv preprint arXiv:2204.07775},
  year={2022}
}
```