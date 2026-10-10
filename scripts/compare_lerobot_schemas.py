"""Report observation/action differences before mixing sim and real data."""

import argparse
import json
from pathlib import Path


def describe(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    relevant = {key: {"dtype": value.get("dtype"), "shape": value.get("shape"),
                      "names": value.get("names")}
                for key, value in features.items()
                if key == "action" or key.startswith("observation.")}
    return {"robot_type": info["robot_type"], "fps": info["fps"], "features": relevant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_dataset", type=Path)
    parser.add_argument("real_dataset", type=Path)
    args = parser.parse_args()
    sim = describe(args.sim_dataset)
    real = describe(args.real_dataset)
    differences = {}
    for key in ("robot_type", "fps"):
        if sim[key] != real[key]:
            differences[key] = {"sim": sim[key], "real": real[key]}
    for key in sorted(set(sim["features"]) | set(real["features"])):
        if sim["features"].get(key) != real["features"].get(key):
            differences[key] = {"sim": sim["features"].get(key),
                                "real": real["features"].get(key)}
    print(json.dumps({"compatible_metadata": not differences, "differences": differences}, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
