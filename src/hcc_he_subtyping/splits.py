from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def patient_level_split(
    manifest: pd.DataFrame,
    case_column: str = "case_id",
    label_column: str = "subtype",
    seed: int = 123,
) -> pd.DataFrame:
    """Create one stratified 70:20:10 split without cross-validation."""
    patient_labels = manifest[[case_column, label_column]].drop_duplicates()
    conflicting = patient_labels.groupby(case_column)[label_column].nunique()
    if (conflicting > 1).any():
        raise ValueError("A case cannot have multiple subtype labels")

    train_cases, remainder = train_test_split(
        patient_labels,
        test_size=0.30,
        random_state=seed,
        stratify=patient_labels[label_column],
    )
    validation_cases, test_cases = train_test_split(
        remainder,
        test_size=1 / 3,
        random_state=seed,
        stratify=remainder[label_column],
    )
    assignment = {
        **dict.fromkeys(train_cases[case_column], "train"),
        **dict.fromkeys(validation_cases[case_column], "validation"),
        **dict.fromkeys(test_cases[case_column], "test"),
    }
    output = manifest.copy()
    output["split"] = output[case_column].map(assignment)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a patient-level 70:20:10 split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--label-column", default="subtype")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    result = patient_level_split(manifest, args.case_column, args.label_column, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

