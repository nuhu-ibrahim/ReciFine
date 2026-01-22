# Fine-tuning Using Recifine Library
The ReciFine library implements the knowledge-augmented and entity-specific token classification approach introduced in:

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
cd Recifine
pip install -e .
```

### Option B: Install from a GitHub release tag (example)

```bash
pip install https://github.com/nuhu-ibrahim/ReciFine/archive/refs/tags/V1.zip
```

---

## Finetuning scripts

All finetuning is done via:

```bash
python3 scripts/finetune.py ...
```

Fine-tuning uses the layered configuration system:
- `--dataset` loads configuration from the path **configs/datasets/\<dataset>/paper.yaml**
- `--model` loads appropriate models **recipebert** or **reciperoberta**
- `--model_name_or_path`the existing model **HF_MODEL_ID** or **LOCAL_CHECKPOINT_DIR**
- `--task_formulation` tells whether the fine-tuning is ``knowledge_augmented`` or ``traditional``
- `--knowledge_type` loads knowledge type from the path **configs/knowledge_type_config/\<knowledge_type>.yaml** when
  fine-tuning is **knowledge-augmented**
---

## Example 1: Traditional finetuning (BIO-n tagging)

Traditional finetuning predicts the BIO tags for all the entity types at once directly from a raw text without using any
curated knowledge.

```bash
python scripts/finetune.py \
  --dataset recifinegold \
  --model recipebert \
  --task_formulation traditional
```

---

## Example 2: Knowledge-augmented Fine-tuing

Knowledge-guided fine-tuning prepends a *curated knowledge prefix* (e.g., question, definition, examples, combined or
entity type name) to the input and fine-tunes the model to extract entities for a single entity type at a time.

```bash
python scripts/finetune.py \
  --dataset recifinegold \
  --model recipebert \
  --task_formulation knowledge_guided \
  --knowledge_type question \
  --model_name_or_path <HF_MODEL_ID_OR_LOCAL_CHECKPOINT_DIR>
```

---

## Important parameters

### Core configuration
- `--dataset`: Dataset key (e.g., `ar`, `englishflowgraph`, `finer`, `foodbase`, `gk`, `recifinegold`, `tasteset1`,
  and `tasteset2`)
- `--model`: Base model config key (i.e., `recipebert`, or `reciperoberta`).

- `--task_formulation`: Either;
    - `traditional`: (BIO-n tagging directly and at a time over the recipe text), or
    - `knowledge_guided`: (knowledge prefix + entity-specific extraction).

- `--knowledge_type`: One of `question`, `entity_type`, `definition`, `example`, `all`. This is for
  `knowledge_guided` fine-tuning only.

### Input / output
- `--data_dir`: The directory containing `train.jsonl`, `val.jsonl`, and the `test/` folder(s).  
  Usually set automatically from the dataset config.

- `--output_dir`: Where checkpoints and evaluation outputs are written. ReciFine automatically appends the knowledge
  type to this directory (e.g., `.../output_ka/question`).

- `--labels`: Label source used to infer the label set. In traditional mode, this typically points to the traditional
  data directory.

- `--model_name_or_path` the existing model **HF_MODEL_ID** or **LOCAL_CHECKPOINT_DIR**

### Fine-tuning hyperparameters
- `--per_gpu_train_batch_size` & `--per_gpu_eval_batch_size`: Batch sizes for train/eval.

- `--num_train_epochs`: Number of epochs.

- `--learning_rate`, `--warmup_steps` & `--weight_decay`: Optimisation parameters.

- `--seed`: Random seed (default: `42`).

---

## Outputs

After fine-tuning, the model and tokenizer are saved under:

- `<output_dir>/<knowledge_type>/` for knowledge-guided fine-tuning, or
- `<output_dir>/traditional/` for traditional fine-tuning

Evaluation results (if `--do_eval`) are written to `eval_results.txt` in the output directory.

---

## Next steps
- For inference: see [README.md](README.md)
- For training: see [TRAINING.md](TRAINING.md)
- For evaluation: see [EVALUATION.md](EVALUATION.md)
- For prediction: see [PREDICTION.md](PREDICTION.md)
