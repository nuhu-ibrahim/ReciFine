from __future__ import annotations

def normalise_traditional_prediction(
    tokens: List[str],
    labels: List[str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    current_type: Optional[str] = None
    current_tokens: List[str] = []

    def flush():
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            text = " ".join(current_tokens)
            out.setdefault(current_type, []).append(text)
        current_type = None
        current_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "O" or not lab:
            flush()
            continue

        if lab.startswith("B-"):
            flush()
            current_type = lab[2:]
            current_tokens = [tok]
            continue

        if lab.startswith("I-"):
            typ = lab[2:]
            if current_type == typ and current_tokens:
                current_tokens.append(tok)
            else:
                flush()
            continue

        flush()

    flush()
    return out


def normalise_knowledge_prediction(
    input_words: List[str],
    predicted_labels: List[str],
    entity_type: str,
    separator: str = "::",
) -> Dict[str, List[str]]:
    try:
        start = input_words.index(separator) + 1
    except ValueError:
        return {"ANS": []}

    tokens = input_words[start:]
    labels = predicted_labels[start:]

    out: List[str] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            text = " ".join(current)
            if text:
                out.append(text)
        current = []

    for tok, lab in zip(tokens, labels):
        if lab == "B-ANS":
            flush()
            current = [tok]
        elif lab == "I-ANS":
            if current:
                current.append(tok)
            else:
                current = [tok]
        else:
            flush()

    flush()
    return {entity_type.upper(): out}