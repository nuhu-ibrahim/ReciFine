# Evaluating Using ReciFine Library
The ReciFine library implements the knowledge-augmented and entity-type specific token classification approach 
introduced in:

```
@inproceedings{
  title   = {Knowledge Augmentation Enhances Token Classification for Recipe Understanding},
  author  = {Ibrahim, Nuhu and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year    = {2026}
}
```

Unlike traditional BIO-n NER, where all entity types are predicted jointly from raw text, this approach reformulates
NER as an entity-type specific task. The model is guided to identify instances of one entity-type at a time by prepending
a curated knowledge prefix to the input text. This library also supports the traditional BIO-n NER.

---

## Installation

### Option A: Install from source (recommended)

```bash
git clone https://github.com/nuhu-ibrahim/ReciFine
cd ReciFine
pip install -e .
```

### Option B: Install from a GitHub release tag (example)

```bash
pip install https://github.com/nuhu-ibrahim/ReciFine/archive/refs/tags/V1.zip
```

---

## Evaluation scripts

All evaluation is done via:

```bash
python3 scripts/evaluate.py ...
```

Evaluation uses the layered configuration system:
- `--dataset` loads configuration from the path **configs/datasets/\<dataset>/paper.yaml**
- `--model` loads appropriate models from the path **configs/base_config/\<model>.yaml**
- `--task_formulation` tells whether the model to evaluated was trained using ``knowledge_augmented`` or 
  ``traditional``
- `--knowledge_type` loads knowledge type from the path **configs/knowledge_type_config/\<knowledge_type>.yaml** when
  evaluation is **knowledge-augmented**
- `--model_name_or_path` the trained model **HF_MODEL_ID** or **LOCAL_CHECKPOINT_DIR** to be used for the evaluation.
- `--dataset_to_evaluate` the folder or list of folders containing the .jsonl file(s) to be evaluated.

If `--model_name_or_path` or `dataset_to_evaluate` are not provided, the paths configured in the dataset config are 
used.

## Example 1: Traditional evalution (BIO-n tagging)

Traditional evaluation evaluates the BIO tags for all the entity types at once directly from a raw text without 
using any curated knowledge.

```bash
python scripts/evaluate.py \
  --dataset recifinegold \
  --model bert_base_uncased \
  --task_formulation traditional
```

---

## Example 2: Knowledge-augmented Evaluation

Knowledge-guided evaluation prepends a **curated knowledge prefix** (e.g., question, definition, examples, combined or
entity type name) to the input and evaluates the model's ability to extract entities for a single entity type at a 
time.

```bash
python scripts/evaluate.py \
  --dataset recifinegold \
  --model bert_base_uncased \
  --task_formulation knowledge_guided \
  --knowledge_type question
```

---

## Supported Datasets
The ReciFine Library supports evaluating on **8 datasets**. For each dataset, please follow the dataset-specific setup
instructions below to obtain the dataset and preprocess them into the required formats for both knowledge-augmented and
traditional evaluation.

- [ReciFineGold.md](configs/datasets/recifinegold/ReciFineGold.md)
- [EnglishRecipeFlowGraph.md](configs/datasets/englishflowgraph/EnglishFlowGraph.md)
- [FINER.md](configs/datasets/finer/Finer.md)
- [FoodBase.md](configs/datasets/foodbase/Foodbase.md)
- [TASTEset-1.md](configs/datasets/tasteset1/Tasteset1.md)
- [TASTEset-2.md](configs/datasets/tasteset2/Tasteset2.md)
- [AR.md](configs/datasets/ar/AR.md)
- [GK.md](configs/datasets/gk/Gk.md)

---

## Important parameters

### Core configuration
- `--dataset`: Dataset key (e.g., `ar`, `englishflowgraph`, `finer`, `foodbase`, `gk`, `recifinegold`, `tasteset1`,
  and `tasteset2`)

- `--model`: Base model config key (e.g., `bert_base_cased.yaml`, `bert_base_uncased`, `roberta_base` or
  `roberta-large`).

- `--task_formulation`: Either;
    - `traditional`: (BIO-n tagging directly and at a time over the recipe text), or
    - `knowledge_guided`: (knowledge prefix + entity-specific extraction).

- `--knowledge_type`: One of `question`, `entity_type`, `definition`, `example`, `all`. This is for
  `knowledge_guided` evaluation only.

### Input / output
- `--dataset_to_evaluate`: The directory(s) containing `.jsonl` files to be evaluated.

- `--model_name_or_path`: The path of the trained model to be evaluated.

- `--output_dir`: Where evaluation outputs are written. ReciFine automatically appends the knowledge
  type to this directory (e.g., `.../output_ka/question`).

- `--labels`: Label source used to infer the label set. In traditional mode, this typically points to the traditional
  data directory.

---

## Next steps
- For training: see [TRAINING.md](Training.md)
- For inference: see [README.md](README.md)
- For fine-tuning: see [FINETUNING.md](FINETUNING.md)
- For prediction: see [PREDICTION.md](PREDICTION.md)
