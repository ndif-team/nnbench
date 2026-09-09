"""Extract or check the pinned CausaLab vocabulary without importing its runtime.

Usage: python scripts/check_causalab_alignment.py /path/to/causalab [--check]
Without --check, prints a replacement snapshot for review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

SNAPSHOT = Path(__file__).resolve().parents[1] / "isb/causalab_vocabulary.json"


def extract(checkout: Path) -> dict:
    fields = {
        "causalab/protocol/schema.py": {
            "Component": "components", "FeaturizerKind": "featurizers",
            "Mechanism": "mechanisms", "DEPRECATED_COMPONENTS": "deprecated_components"},
        "causalab/protocol/engine.py": {"CAPABILITIES": "capabilities"},
    }
    result = {
        "repository": "https://github.com/goodfire-ai/causalab",
        "revision": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {},
    }
    for path, names in fields.items():
        raw = (checkout / path).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{path}"])
        if raw != committed:
            raise ValueError(f"{path} differs from HEAD; commit upstream edits before pinning")
        result["source_sha256"][path] = hashlib.sha256(raw).hexdigest()
        for node in ast.parse(raw).body:
            target = (node.target if isinstance(node, ast.AnnAssign)
                      else node.targets[0] if isinstance(node, ast.Assign) else None)
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            value = node.value
            if isinstance(value, ast.Subscript):
                value = value.slice  # Literal[...] contains only string constants.
            values = ast.literal_eval(value)
            result[names[target.id]] = values if isinstance(values, dict) else sorted(values)
        if not set(names.values()) <= result.keys():
            raise ValueError(f"upstream declarations changed in {path}; review the extractor")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = extract(args.checkout)
    if args.check:
        if actual != json.loads(SNAPSHOT.read_text()):
            raise SystemExit("CausaLab snapshot differs: review source, revision, and vocabulary")
        print(f"CausaLab alignment verified at {actual['revision']}")
    else:
        print(json.dumps(actual, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
