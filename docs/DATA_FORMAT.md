# Data format

Raw study images and private clinical annotations are not distributed.

## Feature bags

Each slide is represented by one HDF5 file named `<slide_id>.h5` with:

- `features`: float matrix of shape `N x 768`.
- `coords`: optional integer matrix of shape `N x 2`, containing level-0 patch coordinates.

Place feature bags under `data/features/h5_files/` or change `features_dir` in the inference configuration.

## Training manifest

The split utility expects `case_id`, `slide_id`, and `subtype`. The optional `feature_path` column can point to each HDF5 bag. The split is performed at case level to prevent leakage between slides from the same patient.

## TCGA-LIHC manifest

`metadata/tcga_lihc_slide_ids.csv` lists the public diagnostic slide identifiers evaluated by both released binary classifiers. It contains identifiers only and does not redistribute TCGA images.

