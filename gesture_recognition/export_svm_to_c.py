# set for [1] second window, NOT 2 second. change by searching for "2SECOND"

import argparse
from pathlib import Path

import joblib
import numpy as np


def fmt_array(values: np.ndarray) -> str:
    return ", ".join(f"{float(v):.9g}f" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained sklearn LinearSVC model to a C header.")
    #parser.add_argument("--model", default="gesture_classifier.joblib") # 2SECOND
    #parser.add_argument("--out", default="main/gesture_model_params.h") # 2SECOND
    parser.add_argument("--model", default="1sec_gesture_classifier.joblib") # 1SECOND
    parser.add_argument("--out", default="main/1sec_gesture_model_params.h") # 1SECOND
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)
    weights = np.asarray(clf.coef_, dtype=np.float64)
    bias = np.asarray(clf.intercept_, dtype=np.float64)
    labels = [str(x) for x in label_encoder.classes_]

    lines: list[str] = []
    lines.append("#ifndef GESTURE_MODEL_PARAMS_H")
    lines.append("#define GESTURE_MODEL_PARAMS_H")
    lines.append("")
    lines.append(f"#define GESTURE_NUM_FEATURES {means.shape[0]}")
    lines.append(f"#define GESTURE_NUM_CLASSES {len(labels)}")
    lines.append("")
    lines.append("static const char *GESTURE_LABELS[GESTURE_NUM_CLASSES] = {")
    for label in labels:
        lines.append(f'    "{label}",')
    lines.append("};")
    lines.append("")
    lines.append(f"static const float GESTURE_FEATURE_MEAN[GESTURE_NUM_FEATURES] = {{{fmt_array(means)}}};")
    lines.append(f"static const float GESTURE_FEATURE_SCALE[GESTURE_NUM_FEATURES] = {{{fmt_array(scales)}}};")
    lines.append("")
    lines.append("static const float GESTURE_WEIGHT[GESTURE_NUM_CLASSES][GESTURE_NUM_FEATURES] = {")
    for row in weights:
        lines.append(f"    {{{fmt_array(row)}}},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const float GESTURE_BIAS[GESTURE_NUM_CLASSES] = {{{fmt_array(bias)}}};")
    lines.append("")
    lines.append("#endif")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Classes: {labels}")
    print(f"Features: {means.shape[0]}")


if __name__ == "__main__":
    main()
