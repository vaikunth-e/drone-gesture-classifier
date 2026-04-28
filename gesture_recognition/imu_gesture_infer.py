import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["time_us", "ax", "ay", "az", "gx", "gy", "gz"]
SIGNAL_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def count_zero_crossings_centered(x: np.ndarray) -> int:
    centered = x - np.mean(x)
    signs = np.sign(centered)
    nz = signs != 0
    if not np.any(nz):
        return 0
    signs = signs[nz]
    return int(np.sum(signs[:-1] != signs[1:]))



def extract_features(df: pd.DataFrame) -> np.ndarray:
    features: list[float] = []

    for col in SIGNAL_COLUMNS:
        x = df[col].to_numpy(dtype=np.float64)

        mean = float(np.mean(x))
        std = float(np.std(x))
        min_v = float(np.min(x))
        max_v = float(np.max(x))
        range_v = max_v - min_v
        energy = float(np.mean(x * x))
        peak_abs = float(np.max(np.abs(x)))
        rms = float(np.sqrt(energy))
        zero_crossings_centered = float(count_zero_crossings_centered(x))
        mean_abs_diff = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0

        features.extend(
            [
                mean,
                std,
                min_v,
                max_v,
                range_v,
                energy,
                peak_abs,
                rms,
                zero_crossings_centered,
                mean_abs_diff,
            ]
        )

    return np.array(features, dtype=np.float64)



def iter_target_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*.txt")):
            if p.is_file():
                yield p



def load_one_file(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return extract_features(df)



def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one IMU sample file or a folder of sample files.")
    parser.add_argument("target", help="A .txt sample file or a folder containing sample files")
    parser.add_argument("--model", default="gesture_classifier.joblib", help="Path to trained model artifact")
    parser.add_argument("--show-scores", action="store_true", help="Print raw decision scores for each class")
    parser.add_argument("--unknown-threshold", type=float, default=None,
                        help="Optional minimum best score required to accept a prediction")
    parser.add_argument("--margin-threshold", type=float, default=None,
                        help="Optional minimum gap between best and second-best score required to accept a prediction")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    class_names = list(label_encoder.classes_)

    target = Path(args.target)
    files = list(iter_target_files(target))
    if not files:
        raise ValueError(f"No .txt files found under {target}")

    for path in files:
        x = load_one_file(path).reshape(1, -1)
        pred_idx = int(pipeline.predict(x)[0])
        pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

        print(f"\nFile: {path}")

        if hasattr(pipeline, "decision_function"):
            scores = np.asarray(pipeline.decision_function(x)).reshape(-1)
            order = np.argsort(scores)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else best_idx
            best_score = float(scores[best_idx])
            second_score = float(scores[second_idx])
            margin = best_score - second_score

            accepted = True
            if args.unknown_threshold is not None and best_score < args.unknown_threshold:
                accepted = False
            if args.margin_threshold is not None and margin < args.margin_threshold:
                accepted = False

            final_label = class_names[best_idx] if accepted else "unknown"
            print(f"Predicted: {final_label}")
            print(f"Best class: {class_names[best_idx]}  score={best_score:.6f}")
            if len(order) > 1:
                print(f"Second class: {class_names[second_idx]}  score={second_score:.6f}")
                print(f"Margin: {margin:.6f}")

            if args.show_scores:
                print("Scores:")
                for idx in order:
                    print(f"  {class_names[int(idx)]:>10s} : {float(scores[int(idx)]): .6f}")
        else:
            print(f"Predicted: {pred_label}")


if __name__ == "__main__":
    main()
