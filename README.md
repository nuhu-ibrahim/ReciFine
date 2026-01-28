# ReciFine

ReciFine is both **a research framework for recipe-focused Named Entity Recognition (NER)** and **a provider of gold-standard food NER datasets**, including **the largest and richest collection of annotated recipes released to date**. The project supports **traditional** and **knowledge-augmented** NER formulations and is designed to facilitate reproducible research across preprocessing, modelling, training, evaluation, and inference.

It provides:
- **ReciFineLibrary**, a unified framework for training, fine-tuning, evaluating, predicting, and inferencing named 
  entities in recipe ingredient lists and instructions.
- **ReciFineGold**, a manually annotated gold-standard dataset of 500 real-world recipes with span-level semantic labels that capture the key components.
- **ReciFine**, the largest and richest annotated recipe dataset to date, produced by scaling annotation to over 2.2 million recipes.
- **ReciFineGen**, the first human-annotated evaluation dataset of automatically generated recipes, designed to support benchmarking for recipe generation and adaptation tasks.

ReciFine introduces **knowledge-augmented and entity type-specific token classification**, enabling models to reason explicitly about fine-grained recipe entities.

---
## Notebook 
Here is a Google Colab Notebook that contains written implementation for quick entity extraction with ReciFine Library: [ReciFine Colab Notebook](https://colab.research.google.com/drive/1CatH2YOhnOWf-VglprEgxONWRkrXRBXo?usp=sharing).

---

## Datasets

### Hugging Face Releases

We release both our gold-standard and large-scale datasets on Hugging Face:

- **ReciFine (Large annotated dataset of over 2.2 million recipes)**  
  [ReciFine Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifine)

- **ReciFineGold (Gold Standard, 500 recipes)**  
    [ReciFineGold Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifinegold)

- **ReciFineGen (Gold Standard)**  
  [ReciFineGen Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifinegen)

- Trained **RecipeBERT** and **RecipeRoBERTa** models for each of the 2 training formulations (traditional and knowledge-augmented) and the 5 knowledge-augmented types.

ReciFine constitutes the **largest and most semantically rich collection of annotated recipe entities currently available**.

Detailed descriptions of the datasets, annotation schema, and entity definitions are provided in [DATASETS.md](DATASETS.md)

All the datasets and the trained models are available in a [HugginFace Collection](https://huggingface.co/collections/nuhuibrahim/recifine)

---

## Quick Start: Inference with `ReciFineNER`

ReciFine provides a lightweight inference wrapper for recipe NER. 

### Code Sample 1 (Knowledge-guided inference)
Below is a code sample showing how to extract QUANTITY entities from a text using the recipebert model trained on the ReciFineGold dataset and using the knowledge-augmented and entity type-specific formulation with **QUESTION** knowledge-type.
```python
pip install https://github.com/nuhu-ibrahim/ReciFine/archive/refs/tags/V1.zip

from recifine.inferencing.inference import ReciFineNER

ner = ReciFineNER.from_pretrained(
    model="recipebert",
    task_formulation="knowledge_guided",
    knowledge_type="question"
)

text = "Add 2 cups of chopped onions and fry until golden ."

prediction = ner.process_text(
    text,
    entity_type="QUANTITY"
)

print(prediction)
```

#### Output
```
{
    'answer': {'QUANTITY': ['2 cups']}, 
    'words_list': ['Add', '2', 'cups', 'of', 'chopped', 'onions', 'and', 'fry', 'until', 'golden', '.'], 
    'pred_labels': ['O', 'B-ANS', 'I-ANS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
}

```


### Code Sample 2 (Traditional inference)
Below shows code sample for extracting entities from text using the recipebert model trained on ReciFineGold dataset using the traditional (BIO-n) approach.
```python
pip install https://github.com/nuhu-ibrahim/ReciFine/archive/refs/tags/V1.zip

from recifine.inferencing.inference import ReciFineNER

ner = ReciFineNER.from_pretrained(
    model="recipebert",
    task_formulation="traditional",
)

text = "Add 2 cups of chopped onions and fry until golden ."

prediction = ner.process_text(
    text
)

print(prediction)
```

#### Output
```
[
    {
        'answer': {
            'ACTION BY CHEF': ['Add', 'chopped', 'fry'], 
            'QUANTITY': ['2 cups'], 
            'FOOD': ['onions'], 'FOOD STATE': ['golden']
        }, 
        'words_list': ['Add', '2', 'cups', 'of', 'chopped', 'onions', 'and', 'fry', 'until', 'golden', '.'],
        'pred_labels': ['B-ACTION BY CHEF', 'B-QUANTITY', 'I-QUANTITY', 'O', 'B-ACTION BY CHEF', 'B-FOOD', 'O', 'B-ACTION BY CHEF', 'O', 'B-FOOD STATE', 'O']
    }
]
```

---

### Parameters Explained 

#### `text` (required)
**Type:** `str`  
Raw input sentence.

---

#### `dataset` (optional, default: `"recifinegold"`)
**Type:** `str`  

The list of supported dataset are in  [Supported Datasets](#supported-datasets)


---

#### `model` (optional, default: `"recipebert"`)
**Type:** `str`  

List of supported models are:
- recipebert
- reciperoberta

---

#### `task_formulation` (optional, default: `"knowledge_guided"`)
**Type:** `str`  
Allowed: `"traditional"`, `"knowledge_guided"`

- if `traditional`, model predicts BIO-n tags directly from raw text at once.
- if `knowledge_guided`, model uses a knowledge prefix + input text to predict entities belonging to a single entity 
  type.

---

#### `knowledge_type` (optional, default: `"question"`)
**Type:** `str`  
Allowed: `"question"`, `"entity_type"`, `"definition"`, `"example"`, `"all"`

Controls which type of knowledge prefix is appended to the input text when `tast_formulation` is `knowledge_guided`

---

## Supported Datasets
- [ReciFineGold.md](src/recifine/configs/datasets/recifinegold/ReciFineGold.md)
- [EnglishRecipeFlowGraph.md](src/recifine/configs/datasets/englishflowgraph/EnglishFlowGraph.md)
- [FINER.md](src/recifine/configs/datasets/finer/Finer.md)
- [FoodBase.md](src/recifine/configs/datasets/foodbase/Foodbase.md)
- [TASTEset-1.md](src/recifine/configs/datasets/tasteset1/Tasteset1.md)
- [TASTEset-2.md](src/recifine/configs/datasets/tasteset2/Tasteset2.md)
- [AR.md](src/recifine/configs/datasets/ar/AR.md)
- [GK.md](src/recifine/configs/datasets/gk/Gk.md)

---
**IMPORTANT NOTE**: Only **ReciFineGold** (`recipebert`
and `reciperoberta` only) are supported automatically through HuggingFace. To use the other datasets and model types,
you will need to train the models and provide the absolute link to the model weights through the 
`model_name_or_path` parameter.

---
## Hardware Requirements

**ReciFineNER** has a low hardware requirement. For fast inference speed, a GPU should be used, but this is not a 
strict requirement.

---
## Using the ReciFine Library
ReciFine can be used to train, fine-tune, evaluate, and run inference on all eight benchmark datasets considered in
this work, and it is fully extensible to new datasets that follow the same configuration and data format.

For detailed, step-by-step guidance, please refer to the following documentation:

- [TRAINING.md](TRAINING.md)
- [FINETUNING.md](FINETUNING.md)
- [EVALUATION.md](EVALUATION.md)
- [PREDICTION.md](PREDICTION.md)

---

## Research Papers

### Knowledge-Augmented and Entity Type-Specific Token Classification
The knowledge-augmented and entity type-specific token classification model architecture is described in the paper 
**Knowledge Augmentation Enhances Token Classification for Recipe Understanding**.

```bibtex
@inproceedings{
  title     = {Knowledge Augmentation Enhances Token Classification for Recipe Understanding},
  author    = {Ibrahim, Nuhu and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year      = {2026}
}
```

### ReciFine Datasets and Controllable Recipe Generation
The ReciFine, ReciFineGold and ReciFineGen datasets are described in the paper
**"ReciFine: Finely Annotated Recipe Dataset for Controllable Recipe Generation"**.

```bibtex
@inproceedings{
  title   = {ReciFine: Finely Annotated Recipe Dataset for Controllable Recipe Generation},
  author  = {Ibrahim, Nuhu and Ravikumar, Rishi and Stevens, Robert and Batista-Navarro, Riza},
  booktitle = {EACL},
  year    = {2026}
}
```

## Results

---

## Security
See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

---

## License
This library is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [LICENSE.md](LICENSE.md)

---

## Contact us
If you have questions please open a Github issue or send us an emails.