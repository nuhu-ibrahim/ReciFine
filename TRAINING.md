# Training Using ReciFine Library
The ReciFine library implements the knowledge-augmented and entity type-specific token classification approach introduced in:

```
@inproceedings{
  title   = {Knowledge Augmentation Enhances Token Classification for Recipe Understanding},
  author  = {Ibrahim, Nuhu and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year    = {2026}
}
```

Unlike traditional BIO-n NER, where all entity types are predicted jointly from raw text, this approach reformulates 
NER as an entity type-specific task. The model is guided to identify instances of one entity type at a time by prepending 
a curated knowledge prefix to the input text. This library also supports the traditional BIO-n NER.

---

## Installation

### Install from source

```bash
git clone https://github.com/nuhu-ibrahim/ReciFine
cd ReciFine
pip install -e .
```

<!-- ### Option B: Install from a GitHub release tag (example)

```bash
pip install https://github.com/nuhu-ibrahim/ReciFine/archive/refs/tags/V1.zip
``` -->

---

## Training scripts

All training is done via:

```bash
python3 scripts/train.py ...
```

Training uses the layered configuration system:
- `--dataset` loads configuration from the path **configs/datasets/\<dataset>/paper.yaml**
- `--model` loads appropriate models **recipebert** or **reciperoberta**
- `--task_formulation` tells whether the training is ``knowledge_augmented`` or ``traditional``
- `--knowledge_type` loads knowledge type from the path **configs/knowledge_type_config/\<knowledge_type>.yaml** when 
  training is **knowledge-augmented**
---

## Example 1: Traditional training (BIO-n tagging)

Traditional training predicts the BIO tags for all the entity types at once directly from a raw text without using any 
curated knowledge.

```bash
python scripts/train.py \
  --dataset recifinegold \
  --model recipebert \
  --task_formulation traditional
```

---

## Example 2: Knowledge-augmented Training

Knowledge-guided training prepends a *curated knowledge prefix* (e.g., question, definition, examples, combined or 
entity type name) to the input and trains the model to extract entities for a single entity type at a time.

```bash
python scripts/train.py \
  --dataset recifinegold \
  --model recipebert \
  --task_formulation knowledge_guided \
  --knowledge_type question
```

---

## Supported Datasets
The ReciFine Library supports training on **8 datasets**. For each dataset, please follow the dataset-specific setup
instructions below to obtain the dataset and preprocess them into the required formats for both knowledge-augmented and
traditional training. 

- [ReciFineGold.md](src/recifine/configs/datasets/recifinegold/ReciFineGold.md)
- [EnglishRecipeFlowGraph.md](src/recifine/configs/datasets/englishflowgraph/EnglishFlowGraph.md)
- [FINER.md](src/recifine/configs/datasets/finer/Finer.md)
- [FoodBase.md](src/recifine/configs/datasets/foodbase/Foodbase.md)
- [TASTEset-1.md](src/recifine/configs/datasets/tasteset1/Tasteset1.md)
- [TASTEset-2.md](src/recifine/configs/datasets/tasteset2/Tasteset2.md)
- [AR.md](src/recifine/configs/datasets/ar/AR.md)
- [GK.md](src/recifine/configs/datasets/gk/Gk.md)

---

## Important parameters

### Core configuration
- `--dataset`: Dataset key (e.g., `ar`, `englishflowgraph`, `finer`, `foodbase`, `gk`, `recifinegold`, `tasteset1`, 
  and `tasteset2`)
- `--model`: Base model config key (i.e., `recipebert` or `reciperoberta`).

- `--task_formulation`: Either;
    - `traditional`: (BIO-n tagging directly and at a time over the recipe text), or
    - `knowledge_guided`: (knowledge prefix + entity type-specific extraction).

- `--knowledge_type`: One of `question`, `entity_type`, `definition`, `example`, `all`. This is for 
  `knowledge_guided` training only.

### Input / output
- `--data_dir`: The directory containing `train.jsonl`, `val.jsonl`, and the `test/` folder(s).  
  Usually set automatically from the dataset config.

- `--output_dir`: Where checkpoints and evaluation outputs are written. ReciFine automatically appends the knowledge 
  type to this directory (e.g., `.../output_ka/question`).

- `--labels`: Label source used to infer the label set. In traditional mode, this typically points to the traditional 
  data directory.

### Training hyperparameters
- `--per_gpu_train_batch_size` & `--per_gpu_eval_batch_size`: Batch sizes for train/eval.

- `--num_train_epochs`: Number of epochs.

- `--learning_rate`, `--warmup_steps` & `--weight_decay`: Optimisation parameters.

- `--seed`: Random seed (default: `42`).

---

## Outputs

After training, the model and tokenizer are saved under:

- `<output_dir>/<knowledge_type>/` for knowledge-guided training, or
- `<output_dir>/traditional/` for traditional training

Evaluation results (if `--do_eval`) are written to `eval_results.txt` in the output directory.

---

## Notes on training other datasets

To train any of the other supported datasets, follow its dataset setup guide first (links above). After preprocessing, the same `scripts/train.py` command works when you update `--dataset` and (optionally) `--model`, `--task_formulation`, and `--knowledge_type`.

---

## Next steps
- For evaluation: see [EVALUATION.md](EVALUATION.md)
- For inference: see [README.md](README.md)
- For fine-tuning: see [FINETUNING.md](FINETUNING.md)
- For prediction: see [PREDICTION.md](PREDICTION.md)
