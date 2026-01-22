# English Recipe Flowgraph Dataset

## Overview
English Recipe Flowgraph is an annotated procedural recipe corpus that captures the semantic structure and execution flow of English cooking instructions. Each recipe is marked with recipe-specific named entities **(e.g., ingredients, actions, tools, quantities, etc)** and a flow graph that represents how actions interact with those entities and the sequencing of steps toward the final dish.

---

## Source and Download
### Download link:
  [https://sites.google.com/view/yy-lab/resource/english-recipe-flowgraph](https://sites.google.com/view/yy-lab/resource/english-recipe-flowgraph)

### Download steps
1. Follow the steps on the website to get the annotated dataset.
2. Collect all the *.list* files in the **r-100** and **r-200** folders.

---

## Expected Directory Structure

Place all the .list files as follows:

```
configs/datasets/englishflowgraph/
└── data/
    └── list/
        ├── recipe1.list
        ├── recipe2.list
               ...
        └── recipen.list
```

---
## Preprocessing the Dataset
Execute the command:

```bash
python scripts/preprocess.py \
  --dataset englishflowgraph
```

---

## Preprocessing Output Structure
After preprocessing, the dataset follows the standard repository layout:

```
configs/datasets/englishflowgraph/
├── data/
    └── list/
        ├── recipe1.list
        ├── recipe2.list
               ...
        └── recipen.list
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
If you use the English Recipe Flowgraph dataset, please cite the original dataset authors.

```
@inproceedings{yamakata2020english,
  title={English Recipe Flow Graph Corpus},
  author={Yamakata, Yoko and Mori, Shinsuke and Carroll, John A},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={5187--5194},
  year={2020}
}
```