# CS436 3D Scene Reconstruction — Complete SfM Pipeline

A comprehensive Structure-from-Motion implementation spanning three weeks of progressive development: from feature matching through two-view geometry to multi-view incremental reconstruction with refinement.

## Overview

This project implements the complete SfM pipeline:
- **Week 1**: Feature detection, matching, and quality analysis
- **Week 2**: Two-view reconstruction using Essential matrix and triangulation
- **Week 3**: Multi-view incremental registration with lightweight bundle adjustment

## Repository layout

- `assets/` — image dataset; place images named `img_1`, `img_2`, ... (any standard extension)
- `src/`
  - `feature_matching_pipeline.py` — feature detection, matching, and visualization
  - `two_view_reconstruction.py` — Essential matrix estimation and two-view geometry
  - `multi_view_sfm.py` — incremental multi-view registration with refinement
- `notebooks/`
  - `week1_deliverable.ipynb` — Feature matching analysis and visualization
  - `week2_deliverable.ipynb` — Two-view reconstruction and pose verification
  - `week3_deliverable.ipynb` — Multi-view SfM with bundle adjustment
  - `weekly_tracker.ipynb` — Progress tracking across all phases
- `outputs/`
  - `feature_matches/` — match visualization images
  - `reconstruction/` — PLY point cloud files

## Environment setup

1. (Optional) create and activate a virtual environment.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your image dataset to `assets/` or update the image directory accordingly.

---

## Week 1: Feature Matching

**Goal**: Detect and match local features across image pairs; quantify match quality.

### Running the pipeline

```bash
python src/feature_matching_pipeline.py \
    --image-dir assets \
    --output-dir outputs/feature_matches \
    --detector SIFT \
    --ratio 0.75
```

**Key flags:**
- `--detector {ORB,SIFT}` — ORB is fast (default); SIFT provides higher-quality matches (requires `opencv-contrib-python`)
- `--ratio` — Lowe's ratio threshold (lower = stricter, fewer matches)
- `--pairs` — optionally specify custom image pairs (default: consecutive pairs)

**Output:**
- Per-pair statistics (match counts, inlier ratios)
- `*_matches.jpg` visualization files in output directory
- Sorted by numeric index (e.g., `img_9` before `img_10`)

### Viewing results

Run `notebooks/week1_deliverable.ipynb`:
1. Loads and displays match visualizations
2. Reports per-pair feature statistics
3. Helps assess match quality before proceeding to geometry

**Typical results:**
- 100–500+ matches per image pair (depends on scene texture)
- 60–80% inlier ratio after ratio test filtering
- Visual confirmation of correct correspondences

---

## Week 2: Two-View Reconstruction

**Goal**: Estimate camera pose between two images and reconstruct a sparse 3D point cloud.

### Core technique

1. **Feature Matching** — Establish 2D-2D correspondences via SIFT/ORB
2. **Essential Matrix Estimation** — Use RANSAC to robustly fit the Essential matrix
3. **Pose Recovery** — Decompose E into rotation R and translation t
4. **Cheirality Check** — Select the unique pose where points have positive depth in both views
5. **Triangulation** — DLT (Direct Linear Transform) to recover 3D point positions
6. **PLY Export** — Save colored sparse point cloud

### Running the pipeline

Open `notebooks/week2_deliverable.ipynb` and run all cells:
- Loads a fixed image pair (default: `img_5.jpeg` vs `img_6.jpeg`)
- Runs two-view reconstruction
- Reports pose metrics and triangulation statistics
- Visualizes the resulting point cloud

### Key outputs

- **Pose metrics**:
  - ||R^T R - I||_F — orthogonality error (should be ~1e-6)
  - det(R) — should equal +1 (proper rotation)
  - ||t|| — translation magnitude
  
- **Epipolar residuals** — mean, median, max reprojection error
  
- **Point cloud**: 
  - Typical: 200–1000 points for good image pairs
  - Colored by source image pixels (BGR → RGB)

**Output file**: `outputs/reconstruction/two_view_points.ply`

### Visualization

View the PLY file in:
- **MeshLab** (free, cross-platform)
- **CloudCompare** (open-source point cloud viewer)
- **3DViewer** (browser-based)

Look for:
- ✓ Coherent 3D structure matching scene geometry
- ✗ Sparse or inverted point clouds → try different image pair or adjust ratio threshold

---

## Week 3: Multi-View Incremental SfM with Refinement

**Goal**: Incrementally register a full image sequence and refine the 3D reconstruction.

### Pipeline overview

1. **Bootstrap** (images 0–1)
   - Detect features in both images
   - Match features via descriptor distance
   - Estimate Essential matrix (RANSAC)
   - Recover pose via cheirality check
   - Triangulate initial point cloud

2. **Incremental Registration** (images 2+)
   - For each new image:
     - Match its features against registered 3D points (3D-2D correspondences)
     - Estimate pose via PnP RANSAC (`cv2.solvePnPRansac`)
     - Refine pose with Levenberg-Marquardt (`cv2.solvePnPRefineLM`)
     - Triangulate new points against the previous registered view
     - Add to growing map

3. **Lightweight Bundle Adjustment**
   - Per-view LM refinement: minimize reprojection error for each camera
   - Multi-view re-triangulation: refine all 3D points using all observations
   - Result: reduced drift accumulation, improved point accuracy

### Running the pipeline

Open `notebooks/week3_deliverable.ipynb` and run all cells:

```python
result = run_incremental_sfm(
    asset_dir=ASSETS_DIR,
    detector='SIFT',
    ratio_thresh=0.75,
    output_path=OUTPUT_DIR / 'week3_multi_view_points.ply',
    refine=True,
    min_correspondences=12,
)
```

**Parameters:**
- `detector` — SIFT or ORB
- `ratio_thresh` — Lowe's ratio for feature matching
- `refine` — enable lightweight bundle adjustment
- `min_correspondences` — minimum 3D-2D pairs to register a view

### Results interpretation

**Example output:**
```
Total images: 14
Successfully registered: 14
Skipped/failed: 0
Total 3D points: 2771
Points re-triangulated: 1269 (46%)

Point cloud bounds:
  X: [-322.60, 742.32]
  Y: [-12.00, 882.61]
  Z: [-452.88, 2223.76]
```

**What these metrics mean:**

| Metric | Interpretation |
|--------|-----------------|
| **100% registration** | All views successfully localized; robust feature matching & pose estimation |
| **Re-triangulated ~46%** | Bundle adjustment actively refined half the points; effective drift control |
| **~2,771 points** | Healthy sparse reconstruction (dense enough for geometry, efficient for computation) |
| **Depth range larger than X/Y** | Good scene depth variation; camera is moving meaningfully through the scene |

**Quality indicators:**
- ✓ **All or nearly all views registered** — robust incremental process
- ✓ **20–50% re-triangulation** — effective refinement
- ✓ **Coherent camera trajectory** — cameras form sensible path (visible in 3D plot)
- ✓ **No "flying" or "bent" structures** — geometric consistency

### Visualizations

The notebook generates four plots:

1. **3D Point Cloud with Camera Positions**
   - Red dots = registered camera centers
   - Colored points = reconstructed scene
   - Labels = image indices

2. **Orthographic Projections** (XY, XZ, YZ)
   - Reveals geometric structure from multiple viewpoints
   - Helps identify local or global distortions

3. **Pose Inlier Distribution**
   - Green bars = successfully registered views
   - Red bars = failed/skipped views
   - Height = number of PnP inliers per image
   - Typically decreases toward end (fewer 3D point observations)

4. **Depth Distribution Histogram**
   - Shows 3D point spread along Z (depth) axis
   - Mean & median lines for reference
   - Indicates scene depth coverage

### Output file

`outputs/reconstruction/week3_multi_view_points.ply` — colored multi-view point cloud

Open in MeshLab/CloudCompare to inspect the full reconstruction.

---

## Virtual Tour Viewer

Turn the refined camera poses and sparse point cloud into an interactive tour:
- Builds a view graph where cameras connect when they share many 3D points.
- Left-click in the current image to jump to the best neighboring camera observing that region.
- Animates the move with lerped translation, slerped rotation, image cross-fade, and projected sparse points.

### Run it

```bash
python src/virtual_tour.py \
  --image-dir assets \
  --detector SIFT \
  --ratio 0.75 \
  --min-shared 15 \
  --fps 30 \
  --transition-sec 1.0
```

Controls: left-click to navigate; `q`/`Esc` to quit. Use `--max-points` to limit rendered points for speed and `--no-refine` to disable the lightweight bundle adjustment if you need faster start-up.

### Web / Three.js viewer (optional)

- Notebook: `notebooks/week4_virtual_tour.ipynb` runs SfM, exports `virtual_tour_data.json`, and writes `virtual_tour_viewer.html` (both in `outputs/reconstruction/`).
- Serve locally to view:  
  ```bash
  cd outputs/reconstruction
  python -m http.server 8000
  # open http://localhost:8000/virtual_tour_viewer.html
  ```
  Drag to orbit, scroll to zoom; cameras are white spheres and the sparse point cloud is colored.

---

## Implementation Notes

### Core Modules

**`src/feature_matching_pipeline.py`**
- `detect_and_describe()` — SIFT/ORB feature extraction
- `match_descriptors()` — Lowe's ratio test filtering
- `load_image()`, `collect_image_paths()` — I/O utilities

**`src/two_view_reconstruction.py`**
- `approximate_intrinsics()` — Camera matrix from image size
- `compute_fundamental_matrix()` — F matrix estimation
- `triangulate_dlt()` — 3D point recovery
- `save_ply()` — PLY export

**`src/multi_view_sfm.py`**
- `run_incremental_sfm()` — Main SfM pipeline
- `_pose_from_pnp()` — PnP RANSAC + LM refinement
- `_retriangulate_points()` — Multi-view point refinement
- `_register_initial_pair()` — Bootstrap from first two images

### Algorithm Details

**PnP Pose Estimation:**
- RANSAC for outlier rejection
- Levenberg-Marquardt refinement on inliers
- Minimum 12 correspondences to register a view

**Bundle Adjustment (Lightweight):**
- Per-image: minimize reprojection error via `cv2.solvePnPRefineLM`
- Global: DLT re-triangulation using all multi-view observations
- Effective drift control without full non-linear optimization

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No matches found | Check image overlap; reduce Lowe ratio; use SIFT instead of ORB |
| Sparse point cloud | Increase image overlap; use pairs with strong baseline; reduce outlier threshold |
| Inverted 3D points | Ensure images have translation (not just rotation); try different image pair |
| Registration failures | Check feature descriptor quality; increase `min_correspondences` gradually; verify camera motion |
| Bent/drifting structure | Enable `refine=True`; check for rapid camera motion between frames |

---

## References

- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed.)
- OpenCV documentation: `findEssentialMat`, `recoverPose`, `solvePnPRansac`, `triangulatePoints`
- SfM survey: Snavely et al., "Photo Tourism" (SIGGRAPH 2006)
