# set for [1] second window, NOT 2 second. change by searching for "2SECOND"

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

#DATA = "./imu_samples" # 2SECOND window
DATA = "./1sec_imu_samples" # 1SECOND window
REQUIRED_COLUMNS = ["time_us", "ax", "ay", "az", "gx", "gy", "gz"]
SIGNAL_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
FEATURE_NAMES_PER_SIGNAL = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "energy",
    "peak_abs",
    "rms",
    "zero_crossings_centered",
    "mean_abs_diff",
]


def infer_label(path: Path, root: Path) -> str:
    """Infer class label from parent folder first, then filename prefix."""
    rel = path.relative_to(root)

    if len(rel.parts) >= 2:
        folder_label = rel.parts[0].strip().lower()
        if folder_label:
            return folder_label

    stem = path.stem.lower()
    if "_" in stem:
        prefix = stem.split("_")[0].strip()
        if prefix:
            return prefix

    return stem



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



def feature_name_list() -> list[str]:
    names: list[str] = []
    for signal in SIGNAL_COLUMNS:
        for feat in FEATURE_NAMES_PER_SIGNAL:
            names.append(f"{signal}_{feat}")
    return names



def iter_sample_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path



def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X: list[np.ndarray] = []
    y: list[str] = []
    paths: list[str] = []

    for path in iter_sample_files(root):
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        label = infer_label(path, root)
        X.append(extract_features(df))
        y.append(label)
        paths.append(str(path))

    if not X:
        raise ValueError(f"No .txt training files found under {root}")

    return np.vstack(X), np.array(y), paths



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear SVM gesture classifier from IMU CSV windows.")
    parser.add_argument("--data-dir", default=DATA, help="Root folder containing gesture sample files")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of data reserved for testing")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0, help="Regularization parameter for LinearSVC")
    parser.add_argument("--model-out", default="gesture_classifier.joblib", help="Path for saved sklearn pipeline")
    parser.add_argument("--metadata-out", default="gesture_classifier_metadata.json", help="Path for JSON metadata")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    X, y_raw, paths = load_dataset(data_root)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=args.c, dual="auto", max_iter=10000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"Loaded {len(X)} samples from {data_root}")
    print(f"Classes: {class_names}")
    print(f"Features per signal: {len(FEATURE_NAMES_PER_SIGNAL)}")
    print(f"Total features: {X.shape[1]}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_name_list(),
        "required_columns": REQUIRED_COLUMNS,
        "signal_columns": SIGNAL_COLUMNS,
    }
    joblib.dump(artifact, args.model_out)

    metadata = {
        "data_dir": str(data_root),
        "num_samples": int(len(X)),
        "num_features": int(X.shape[1]),
        "features_per_signal": FEATURE_NAMES_PER_SIGNAL,
        "classes": class_names,
        "feature_names": feature_name_list(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_paths": paths_test,
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved model to {args.model_out}")
    print(f"Saved metadata to {args.metadata_out}")


if __name__ == "__main__":
    main()
