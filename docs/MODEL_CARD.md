# Model card

## Intended use

This research model predicts HCC molecular ecosystem labels (M1, M2, or M3) from H&E whole-slide images. It is intended for retrospective research and is not a diagnostic device.

## Input

- H&E WSI processed at an effective 10x magnification.
- Tissue-containing 224 x 224 pixel patches.
- Macenko stain normalization against a study reference image.
- A frozen BEPH backbone producing one 768-dimensional embedding per patch.

## Architecture

Two single-branch CLAM models aggregate patch embeddings with gated attention:

1. M1 (class 0) versus M2&M3 (class 1).
2. M1&M2 (class 0) versus M3 (class 1).

Concordant low-class predictions yield M1, concordant high-class predictions yield M3, and discordant predictions yield M2.

## Data partitioning

Model development used one patient-level, subtype-stratified 70:20:10 training/validation/test split. Slides belonging to the same patient must remain in the same partition. No five-fold cross-validation is represented by this release.

## Limitations

Performance may be affected by scanner resolution, staining protocol, tissue preparation, tumor content, and domain shift. External validation is required before use in a new cohort.

