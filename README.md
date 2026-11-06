# H&E-based molecular ecosystem subtyping in HCC

Research implementation of a pathology model that combines stain-normalized 10x H&E patches, frozen BEPH embeddings, and two CLAM-SB classifiers to infer M1, M2, and M3 molecular ecosystem states.

## Workflow

1. Read the WSI at an effective 10x magnification and segment tissue.
2. Tile tissue into 224 x 224 pixel patches.
3. Normalize patch staining with the Macenko method.
4. Extract 768-dimensional features with the frozen BEPH backbone.
5. Run two CLAM-SB classifiers: M1 vs M2&M3 and M1&M2 vs M3.
6. Fuse the two binary outputs into a three-class prediction.

| M1 vs M2&M3 | M1&M2 vs M3 | Final label |
|---|---|---|
| M1 | M1&M2 | M1 |
| M1 | M3 | M2 |
| M2&M3 | M1&M2 | M2 |
| M2&M3 | M3 | M3 |

## Installation

```bash
conda env create -f environment.yml
conda activate hcc-he-subtyping
pip install -e .
git lfs pull
```

## Inference

Place one `N x 768` BEPH feature bag per slide in `data/features/h5_files/`, then run:

```bash
hcc-he-infer --config configs/inference.yaml
```

The output contains both binary probabilities, both binary decisions, and the fused M1/M2/M3 label. Progress is printed for every slide.

## Reproduce the 70:20:10 split

Raw data are not included. Given a manifest containing `case_id`, `slide_id`, and `subtype`:

```bash
hcc-he-split \
  --manifest metadata/your_training_manifest.csv \
  --output metadata/split_70_20_10.csv \
  --seed 123
```

The split is subtype-stratified and patient-level. It creates one training, validation, and held-out test partition; this release does not use five-fold cross-validation.

To train both binary classifiers from that manifest in one command:

```bash
hcc-he-train \
  --manifest metadata/your_training_manifest.csv \
  --output-dir runs/training \
  --seed 123
```

## TCGA-LIHC validation IDs

The public slide identifiers used for TCGA-LIHC inference are provided in `metadata/tcga_lihc_slide_ids.csv`. TCGA images must be obtained from the NCI Genomic Data Commons under its applicable data-use terms.

## Repository layout

```text
configs/                 Inference and training configurations
docs/                    Model card and data specification
metadata/                Templates and public TCGA slide IDs
src/hcc_he_subtyping/    CLAM-SB, fusion, inference, and split code
tests/                   Unit and checkpoint compatibility tests
weights/                 Released model weights and checksums
```

## Weights

The two CLAM-SB checkpoints are small. The BEPH backbone is tracked with Git LFS because it exceeds GitHub's regular file limit. See `weights/README.md` for SHA-256 checksums.

## Research use and citation

This software is for research use only. Update `CITATION.cff` with the final article and repository metadata before release. No license is implied until a `LICENSE` file is added by the repository owner.
