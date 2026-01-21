# Overview of Released Datasets

- **ReciFine**: Large, automatically annotated dataset of over **2.2 million recipes**  
  [ReciFine Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifine)

- **ReciFineGold**: Manually annotated gold-standard dataset of **500 recipes**  
  [ReciFineGold Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifinegold)

- **ReciFineGen**: Gold-standard evaluation dataset for **recipe generation**  
  [ReciFineGen Dataset on HugginFace](https://huggingface.co/datasets/nuhuibrahim/recifinegen)

---

## ReciFineGold: Gold-Standard Dataset

**ReciFineGold** consists of **500 manually annotated real-world recipes**, annotated by expert annotators following the [Recipe Case Study Paper](https://www.dl.soc.i.kyoto-u.ac.jp/~tajima/papers/hmdata18yamakatawww.pdf) guidelines.

Key properties:
- Token-level annotations for all 10 ERFG entity types
- High inter-annotator agreement (F1 = **91.53%**)
- Designed as a gold-standard benchmark for training and evaluation
- Used to validate model generalisation and annotation quality at scale

---

## ReciFine: Large Silver-standard Dataset

**ReciFine** is a large, silver-standard dataset derived from the [RecipeNLG](https://aclanthology.org/2020.inlg-1.4/) corpus and enriched with fine-grained 
semantic annotations.

### Scale
- **2.2+ million recipes**
- **97+ million entity mentions**
- Token-level annotations across all recipe instructions

### Entity Distribution
| **Entity Type**   | **Frequency** | **Top Entities** |
|-------------------|--------------:|:---|
| Food (F)          |    30,199,222 | *salt, water, sugar, butter, flour* |
| Tool (T)          |    10,384,889 | *bowl, oven, pan, saucepan* |
| Duration (D)      |     3,700,982 | *10 min, 5 min, 30 min* |
| Quantity (Q)      |     4,476,287 | *remaining, all, 2, half, 1* |
| Chef Action (Ac)  |    30,854,542 | *add, bake, mix, stir, cook* |
| Discont. Ac (Ac2) |     2,199,530 | *together, to taste, to a boil* |
| Food Action (Af)  |     3,023,358 | *cool, stand, set, combined* |
| Tool Action (At)  |       705,504 | *comes out, stand, set* |
| Food State (Sf)   |     7,331,595 | *hot, tender, smooth, browned* |
| Tool State (St)   |     5,022,448 | *large, medium, small, 350°* |

**Table 1** Frequencies of entities for each ReciFine entity
type, along with their most frequent entities.

---

## ReciFineGen: Recipe Generation Evaluation Dataset

**ReciFineGen** is a gold-standard dataset designed to evaluate **controllable recipe generation** using human judgements.


### Evaluation Protocol
- **200 randomly selected test recipes** were used.
- Each recipe was **generated 10 times** by a model to account for stochasticity in generation using 5 different
  extracted prompt context type.

### Human Evaluation

Human evaluation was conducted to assess the quality of automatically generated recipes that automatic metrics cannot fully capture, such as realism, creativity, and usability.

### Annotation Setup
- Recipes were rated on a **5-point Likert scale**:
    - **1** = Very poor
    - **5** = Excellent
- Five evaluation dimensions were considered:
    - **Readability**: Fluency and clarity of the recipe text.
    - **Accuracy**: Correctness of ingredients, tools, and actions.
    - **Feasibility**: Whether the recipe can realistically be executed.
    - **Creativity**: Novelty and originality of the recipe.
    - **Overall Quality**: Holistic assessment by the annotator.

- 5 annotators with native-level English proficiency independently evaluated the outputs.

## ReciFine & ReciFineGold Annotation Schema

The **ReciFine** and **ReciFineGold** follow the [The Case Study Paper](https://www.dl.soc.i.kyoto-u.ac.jp/~tajima/papers/hmdata18yamakatawww.pdf) annotation scheme and include **10 fine-grained entity types** annotated at token level within recipe instructions.

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

**Table 2:** Entity types in the English Recipe Flow Graph corpus.

---