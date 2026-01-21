# FINER Dataset

## Overview

FINER is a recipe named-entity recognition (NER) dataset annotated with food-related entities such as **ingredients, quantities, units, products, and states**. The original dataset is released in CoNLL (BIO) format and is converted in this project into a **knowledge-guided and traditional BIO-n JSONL format** suitable for training, validation, and prediction.

---

## Source and Download

### Download link:
  [https://figshare.com/articles/dataset/Food_Ingredient_Named-Entity_Data_Construction_using_Semi-supervised_Multi-model_Prediction_Technique/20222361?file=36144501](https://figshare.com/articles/dataset/Food_Ingredient_Named-Entity_Data_Construction_using_Semi-supervised_Multi-model_Prediction_Technique/20222361?file=36144501)

### Download steps

1. Download the file `finer.rar` from the link above.
2. Extract the archive.
3. The extracted file of interest is:

   ```
   finer.conll
   ```

---

## Raw Data Format

The file `finer.conll` follows the standard CoNLL BIO format:

* One token per line with each line containing:

  ```
  TOKEN<TAB>TAG
  ```
* Tags use the BIO scheme (e.g., `B-ING`, `I-ING`, `O`).
* Sentences are separated by blank lines.

---

## Expected Directory Structure

Place the raw dataset as follows:

```
configs/datasets/finer/
└── data/
    └── finer.conll
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset finer
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/finer/
├── data
    └── finer.conll
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

If you use the FINER dataset, please cite the original dataset authors.

```
@inproceedings{komariah2023smpt,
  title={SMPT: A Semi-supervised Multi-model Prediction Technique for Food Ingredient Named Entity Recognition (FINER) Dataset Construction},
  author={Komariah, Kokoy Siti and Purnomo, Ariana Tulus and Satriawan, Ardianto and Hasanuddin, Muhammad Ogin and Setianingsih, Casi and Sin, Bong-Kee},
  booktitle={Informatics},
  volume={10},
  number={1},
  pages={10},
  year={2023},
  organization={MDPI}
}
```