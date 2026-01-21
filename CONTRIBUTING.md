# Contributing to ReciFine
We welcome contributions from the research and engineering community, including bug fixes, new datasets, model extensions, and documentation improvements.

---

## Ways to Contribute

You can contribute to ReciFine in several ways:

- Reporting bugs or issues
- Proposing new features or extensions
- Improving documentation or examples
- Adding support for new datasets, entity types, or models
- Improving preprocessing, training, or inference pipelines
- Contributing experimental results, benchmarks, or annotations

---

## Reporting Bugs

If you encounter a bug:

1. Check the existing GitHub issues to ensure it has not already been reported.
2. Open a new issue and include:
   - A clear and descriptive title
   - Steps to reproduce the issue
   - Expected vs. actual behaviour
   - Environment details (OS, Python version, PyTorch/Transformers versions)
   - Relevant logs or stack traces

Please keep bug reports focused and reproducible.

---

## Proposing Features or Enhancements

Feature requests are welcome. When proposing a feature:

- Clearly describe the motivation and use case
- Explain how it fits within ReciFine’s design philosophy
- Indicate whether it affects preprocessing, modelling, training, inference, or datasets
- Where relevant, reference related academic work or benchmarks

For substantial changes, we recommend opening an issue for discussion **before** submitting a pull request.

---

## Pull Request Process

1. **Fork** the repository and create a new branch from `main`.
2. Make your changes following the project’s coding style.
3. Ensure your code:
   - Is clearly structured and readable
   - Includes docstrings and comments where appropriate
   - Does not break existing functionality
4. Run any existing tests or scripts relevant to your changes.
5. Submit a pull request with:
   - A clear description of the change
   - The motivation behind it
   - Any relevant experimental results (if applicable)

Pull requests will be reviewed with an emphasis on **correctness, clarity, and research reproducibility**.

---


## Research and Dataset Contributions

If your contribution involves datasets, annotations, or research artefacts:

- Clearly document annotation guidelines and assumptions
- Specify data provenance and preprocessing steps
- Ensure configurations are reproducible via the existing config system
- Avoid hard-coding dataset- or experiment-specific paths

We particularly welcome extensions to the food and recipe NER datasets, provided they meet the same quality and documentation standards as the existing gold annotations.

---

## Security and Vulnerability Disclosure

If you discover a security vulnerability or potential misuse risk, you may:

- Report it **privately by email**, or
- Create a **GitHub issue** if public disclosure is appropriate

For private disclosure, please email:

**nuhu.ibrahim (at) manchester.ac.uk**

We will respond as promptly as possible.

---

## License

By contributing to ReciFine, you agree that your contributions will be licensed under the same license as the project.

Please see the **[LICENSE.md](LICENSE.md)** file in this repository for full details.

---

## Questions and Discussion

For general questions, clarifications, or research discussions:

- Use GitHub issues where appropriate
- Keep discussions constructive and technically focused

We appreciate your interest in ReciFine and your contributions to advancing food and recipe-focused NLP research.
