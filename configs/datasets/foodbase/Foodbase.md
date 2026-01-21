# Foodbase Dataset

## Overview
FoodBase is an annotated food entity corpus built from Allrecipes data. It includes food entity annotations mapped to semantic categories from the Hansard tagset, providing both a manually curated gold standard corpus.

---

## Source and Download

### Download link:
  [http://cs.ijs.si/repository/FoodBase/foodbase.zip](http://cs.ijs.si/repository/FoodBase/foodbase.zip)

### Download steps

1. Download the file `foodbase.zip` from the link above.
2. Extract the content.
3. The extracted file of interest is:

   ```
   FoodBase_curated.xml
   ```

---

## Raw Data Format
The file `FoodBase_curated.xml` is an XML-structured recipe corpus with span-based annotations:

- Each `<document>` contains a recipe with metadata and full text in `<infon key="full_text">`.  
- Food entities are marked as separate `<annotation>` elements.  
- Each annotation specifies a character **offset** and **length**, the exact entity text.

---

## Expected Directory Structure

Place the raw dataset as follows:

```
configs/datasets/foodbase/
└── data/
    └── FoodBase_curated.xml
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset foodbase
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/foodbase/
├── data
    └── FoodBase_curated.xml
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

If you use the Foodbase dataset, please cite the original dataset authors.

```
@article{popovski2019foodbase,
  title={FoodBase Corpus: A New Resource of Annotated Food Entities},
  author={Popovski, Gorjan and Seljak, Barbara Korou{\v{s}}i{\'c} and Eftimov, Tome},
  journal={Database},
  volume={2019},
  pages={baz121},
  year={2019},
  publisher={Oxford University Press}
}
```