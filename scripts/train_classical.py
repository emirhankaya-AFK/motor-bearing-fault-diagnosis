"""Leakage-aware leave-one-load-out benchmark for engineered features."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from bearing_fault import LABELS, extract_feature_matrix, feature_names, load_dataset


def main() -> None:
    windows, labels, loads, recording_ids = load_dataset(Path("data/raw"))
    features = extract_feature_matrix(windows)
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    folds = []
    holdout_model = None

    for test_load in sorted(np.unique(loads)):
        train_mask = loads != test_load
        test_mask = loads == test_load
        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(features[train_mask], labels[train_mask])
        predictions = model.predict(features[test_mask])
        fold = {
            "held_out_load_hp": int(test_load),
            "train_recordings": sorted(np.unique(recording_ids[train_mask]).tolist()),
            "test_recordings": sorted(np.unique(recording_ids[test_mask]).tolist()),
            "accuracy": round(float(accuracy_score(labels[test_mask], predictions)), 4),
            "macro_f1": round(float(f1_score(labels[test_mask], predictions, average="macro")), 4),
            "classification_report": classification_report(
                labels[test_mask], predictions, labels=LABELS, output_dict=True, zero_division=0
            ),
        }
        folds.append(fold)
        if test_load == 3:
            holdout_model = model
            ConfusionMatrixDisplay.from_predictions(
                labels[test_mask], predictions, labels=LABELS, normalize="true", cmap="Blues"
            )
            plt.title("3 HP holdout — normalized confusion matrix")
            plt.tight_layout()
            plt.savefig(artifact_dir / "confusion_matrix_load_3.png", dpi=170)
            plt.close()

    summary = {
        "dataset": "CWRU 12 kHz drive-end; 0.007-inch seeded faults",
        "window_size": 2048,
        "window_overlap": 0,
        "split": "leave-one-motor-load-out; source recordings never cross a fold",
        "samples": int(len(labels)),
        "folds": folds,
        "mean_macro_f1": round(float(np.mean([fold["macro_f1"] for fold in folds])), 4),
        "std_macro_f1": round(float(np.std([fold["macro_f1"] for fold in folds])), 4),
    }
    (artifact_dir / "classical_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    assert holdout_model is not None
    joblib.dump(holdout_model, artifact_dir / "extra_trees_load_3.joblib")

    importance = sorted(
        zip(feature_names(), holdout_model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    (artifact_dir / "feature_importance.json").write_text(
        json.dumps(dict(importance), indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
