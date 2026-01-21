import argparse

def _require(args: argparse.Namespace, names: list[str]) -> None:
    missing = []
    for n in names:
        v = getattr(args, n, None)
        if v is None:
            missing.append(n)
        elif isinstance(v, str) and v.strip() == "":
            missing.append(n)
    if missing:
        loaded = getattr(args, "_loaded_config_files", [])
        msg = (
            "Missing required parameters: "
            + ", ".join(f"--{m}" for m in missing)
            + "\nLoaded config files:\n  - "
            + ("\n  - ".join(loaded) if loaded else "(none)")
            + "\n\nFix by either:\n"
              "  (1) setting them in YAML, or\n"
              "  (2) passing them on the CLI.\n"
        )
        raise SystemExit(msg)
