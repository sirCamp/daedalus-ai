"""Iris classification with sklearn — minimal example for Daedalus."""

import argparse
import json
import logging
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    logger.info("Loading Iris dataset")
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
    )

    logger.info(
        "Training RandomForest (n_estimators=%d, max_depth=%s, min_samples_split=%d)",
        args.n_estimators, args.max_depth, args.min_samples_split,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    logger.info("accuracy=%.4f", accuracy)
    logger.info("f1_macro=%.4f", f1)

    results = {
        "eval": {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1, 4),
        }
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
