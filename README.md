# CS436 3D Scene Reconstruction — Week 1 Deliverable

This repository contains the Week 1 milestone for the Structure-from-Motion project: a full script and companion notebook for detecting local features on your photo set, filtering matches via Lowe's ratio test, and exporting visualization frames that document the quality of those correspondences.

## Repository layout

- `assets/` — place your captured images named `img_1`, `img_2`, ... (any standard extension).
- `src/feature_matching_pipeline.py` — reusable Python module & CLI that processes the dataset and writes out match visualizations plus statistics.
- `notebooks/week1_feature_matching.ipynb` — notebook used to showcase the weekly deliverable, importing the structured code.
- `outputs/feature_matches/` — auto-created folder containing the `.jpg` match visualizations emitted by the pipeline.

## Environment setup

1. (Optional) create and activate a virtual environment.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your image dataset to `assets/` or update the `--image-dir` argument accordingly.

## Running the feature-matching pipeline

```bash
python src/feature_matching_pipeline.py \
    --image-dir assets \
    --output-dir outputs/feature_matches \
    --detector ORB \
    --ratio 0.75
```

Key flags:
- `--detector {ORB,SIFT}` selects the feature family. ORB is fast & default; SIFT may improve matches for challenging textures (requires `opencv-contrib-python`).
- `--ratio` controls how strict Lowe's ratio test is (lower = stricter, fewer matches).
- `--pairs 1-5 2-6 ...` optionally overrides the default “consecutive pairs” mode, referencing image indices in their filenames.

Each run prints per-pair statistics and writes `*_matches.jpg` files to the output directory. The script expects filenames of the form `img_<index>.<ext>` and now sorts them by their numeric index so that `img_9` precedes `img_10`.

## Visualizing results in the notebook

Open `notebooks/week1_feature_matching.ipynb`, run all cells, and the notebook will:
1. Ensure `src/` is importable.
2. Invoke the same `run_pipeline` function that powers the CLI.
3. Preview up to three of the generated visualization images inline via Matplotlib, which you can screenshot for weekly reports.

If the notebook warns that images are missing, confirm the filenames and rerun the pipeline after adding the dataset.

## Viewing the reconstructed point cloud

- After running the two-view reconstruction cell in `notebooks/week1_feature_matching.ipynb`, open the generated PLY at `outputs/reconstruction/two_view_points.ply` in a viewer such as MeshLab or CloudCompare.
- If the cloud looks sparse or inverted, verify that your photo pair has translation (not just rotation) and try rerunning with a different pair or stricter Lowe ratio.

## Next steps

With reliable feature matching established, the next milestone will use these correspondences to estimate the Essential matrix, recover relative pose, and triangulate a sparse 3D point cloud.
