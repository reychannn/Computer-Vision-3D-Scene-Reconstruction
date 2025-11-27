"""Two-view reconstruction pipeline for Week 2.

This module wraps the notebook logic for estimating an Essential matrix,
recovering relative pose, triangulating inlier correspondences, and exporting
the resulting sparse point cloud to a PLY file.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.feature_matching_pipeline import collect_image_paths, create_detector


@dataclass
class ReconstructionResult:
    """Container describing the reconstructed view pair."""

    image_a: Path
    image_b: Path
    K: np.ndarray
    E: np.ndarray
    pose_label: str
    R: np.ndarray
    t: np.ndarray
    inlier_mask: np.ndarray
    pts_a_in: np.ndarray
    pts_b_in: np.ndarray
    pts_a_norm: np.ndarray
    pts_b_norm: np.ndarray
    points_3d: np.ndarray
    colors_rgb: np.ndarray
    ply_path: Path
    match_count: int
    inlier_count: int


def choose_image_pair(
    asset_dir: Path, preferred_names: Optional[Sequence[str]] = None
) -> Tuple[Path, Path]:
    """Pick the two images to reconstruct, optionally honoring a preferred pair."""

    images = collect_image_paths(asset_dir)
    if preferred_names:
        lookup = {path.name: path for path in images}
        if all(name in lookup for name in preferred_names):
            return lookup[preferred_names[0]], lookup[preferred_names[1]]
    return images[0], images[1]


def approximate_intrinsics(image: np.ndarray) -> np.ndarray:
    """Approximate K with f = image width and principal point at the center."""

    h, w = image.shape[:2]
    f = float(w)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def detect_and_match_points(
    detector_name: str, img_a: np.ndarray, img_b: np.ndarray, ratio: float
):
    """Detect keypoints, compute descriptors, and apply Lowe's ratio test."""

    detector = create_detector(detector_name)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kps_a, des_a = detector.detectAndCompute(gray_a, None)
    kps_b, des_b = detector.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None:
        raise RuntimeError("Could not compute descriptors; check image quality.")
    norm = cv2.NORM_HAMMING if des_a.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in knn if n is not None and m.distance < ratio * n.distance]
    good.sort(key=lambda m: m.distance)
    pts_a = np.float64([kps_a[m.queryIdx].pt for m in good])
    pts_b = np.float64([kps_b[m.trainIdx].pt for m in good])
    return good, pts_a, pts_b


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def triangulate_with_pose(R: np.ndarray, t: np.ndarray, K: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray):
    """Triangulate normalized points for a given pose and perform a cheirality check."""

    proj_0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    proj_1 = K @ np.hstack([R, t])
    pts1_norm = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    pts2_norm = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    hom = cv2.triangulatePoints(proj_0, proj_1, pts1_norm, pts2_norm)
    points_3d = (hom[:3] / hom[3]).T
    points_cam1 = (R @ points_3d.T + t).T
    valid_mask = (points_3d[:, 2] > 0) & (points_cam1[:, 2] > 0)
    return points_3d, valid_mask


def save_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> str:
    """Write a minimal ASCII PLY for viewing the sparse reconstruction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(points):
            if colors is not None:
                r, g, b = colors[i]
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(r)} {int(g)} {int(b)}\n"
                )
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return str(path)


def pose_sanity_checks(R: np.ndarray, t: np.ndarray):
    """Return diagnostics for rotation orthonormality and translation magnitude."""

    RtR = R.T @ R
    ortho_err = float(np.linalg.norm(RtR - np.eye(3)))
    detR = float(np.linalg.det(R))
    t_norm = float(np.linalg.norm(t))
    return {"ortho_err": ortho_err, "det": detR, "t_norm": t_norm}


def essential_residuals(E: np.ndarray, pts1_norm: np.ndarray, pts2_norm: np.ndarray):
    """Compute |x2^T E x1| residuals for normalized inlier correspondences."""

    res = []
    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1 = np.array([p1[0], p1[1], 1.0])
        x2 = np.array([p2[0], p2[1], 1.0])
        res.append(abs(x2 @ E @ x1))
    res = np.asarray(res)
    return {
        "mean": float(res.mean()),
        "median": float(np.median(res)),
        "max": float(res.max()),
    }


def run_two_view_reconstruction(
    asset_dir: Path,
    preferred_pair: Optional[Iterable[str]] = None,
    detector: str = "SIFT",
    ratio_thresh: float = 0.7,
    output_path: Optional[Path] = None,
    write_ply: bool = True,
) -> ReconstructionResult:
    """Full two-view reconstruction flow returning a ReconstructionResult.

    If write_ply is False, the returned ply_path will be None and no file is written.
    """

    img_a_path, img_b_path = choose_image_pair(asset_dir, preferred_pair)
    image_a = cv2.imread(str(img_a_path))
    image_b = cv2.imread(str(img_b_path))
    if image_a is None or image_b is None:
        raise IOError("Image files could not be loaded.")

    K = approximate_intrinsics(image_a)
    matches, pts_a, pts_b = detect_and_match_points(detector, image_a, image_b, ratio_thresh)
    if len(matches) < 8:
        raise RuntimeError("Not enough matches to estimate the Essential matrix.")

    E, ransac_mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        raise RuntimeError("cv2.findEssentialMat failed.")
    if E.ndim > 2:
        E = E[:, :, 0]

    _, _, _, ransac_mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=ransac_mask)
    inlier_mask = ransac_mask.ravel().astype(bool)
    pts_a_in = pts_a[inlier_mask]
    pts_b_in = pts_b[inlier_mask]

    R1, R2, t_unit = cv2.decomposeEssentialMat(E)
    pose_candidates = [
        ("R1,+t", R1, t_unit),
        ("R1,-t", R1, -t_unit),
        ("R2,+t", R2, t_unit),
        ("R2,-t", R2, -t_unit),
    ]

    cheirality_counts = []
    triangulated = []
    for _, R_cand, t_cand in pose_candidates:
        cloud, valid_mask = triangulate_with_pose(R_cand, t_cand, K, pts_a_in, pts_b_in)
        cheirality_counts.append(int(valid_mask.sum()))
        triangulated.append((cloud, valid_mask))

    best_idx = int(np.argmax(cheirality_counts))
    best_label, R_best, t_best = pose_candidates[best_idx]
    best_cloud, best_valid = triangulated[best_idx]
    best_points = best_cloud[best_valid]

    pixel_coords = np.round(pts_a_in[best_valid]).astype(int)
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_a.shape[1] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_a.shape[0] - 1)
    colors_rgb = image_a[pixel_coords[:, 1], pixel_coords[:, 0], ::-1]

    ply_path: Optional[Path] = None
    if write_ply:
        if output_path is None:
            output_path = asset_dir.parent / "outputs" / "reconstruction" / "two_view_points.ply"
        ply_path = Path(save_ply(output_path, best_points, colors_rgb))

    pts_a_norm = cv2.undistortPoints(pts_a_in.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts_b_norm = cv2.undistortPoints(pts_b_in.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    t_unit_best = t_best / np.linalg.norm(t_best)
    E_pose = _skew(t_unit_best.ravel()) @ R_best

    return ReconstructionResult(
        image_a=img_a_path,
        image_b=img_b_path,
        K=K,
        E=E_pose,
        pose_label=best_label,
        R=R_best,
        t=t_best,
        inlier_mask=inlier_mask,
        pts_a_in=pts_a_in,
        pts_b_in=pts_b_in,
        pts_a_norm=pts_a_norm,
        pts_b_norm=pts_b_norm,
        points_3d=best_points,
        colors_rgb=colors_rgb,
        ply_path=ply_path,
        match_count=len(matches),
        inlier_count=int(inlier_mask.sum()),
    )


def find_best_reconstruction(
    asset_dir: Path,
    detector: str = "SIFT",
    ratio_thresh: float = 0.7,
    output_path: Optional[Path] = None,
) -> tuple[ReconstructionResult, list[dict]]:
    """Evaluate all image pairs and return the densest reconstruction.

    Returns (best_result, pair_summaries). Each summary is a dict with keys:
    pair, status, points, inliers, matches, error (optional).
    """

    images = collect_image_paths(asset_dir)
    pair_summaries: list[dict] = []
    best: ReconstructionResult | None = None
    best_points = -1

    for img_a, img_b in combinations(images, 2):
        pair_name = (img_a.name, img_b.name)
        try:
            candidate = run_two_view_reconstruction(
                asset_dir=asset_dir,
                preferred_pair=pair_name,
                detector=detector,
                ratio_thresh=ratio_thresh,
                output_path=None,
                write_ply=False,
            )
            pts_count = len(candidate.points_3d)
            pair_summaries.append(
                {
                    "pair": pair_name,
                    "status": "ok",
                    "points": pts_count,
                    "inliers": candidate.inlier_count,
                    "matches": candidate.match_count,
                    "pose": candidate.pose_label,
                }
            )
            if pts_count > best_points:
                best_points = pts_count
                best = candidate
        except Exception as exc:  # noqa: BLE001
            pair_summaries.append(
                {
                    "pair": pair_name,
                    "status": "error",
                    "points": 0,
                    "inliers": 0,
                    "matches": 0,
                    "pose": None,
                    "error": str(exc),
                }
            )

    if best is None:
        raise RuntimeError("No valid reconstruction found across image pairs.")

    # Write the best PLY (and return a copy with ply_path set).
    ply_out = output_path or asset_dir.parent / "outputs" / "reconstruction" / "two_view_points.ply"
    _ = save_ply(ply_out, best.points_3d, best.colors_rgb)
    best_with_path = ReconstructionResult(**{**best.__dict__, "ply_path": Path(ply_out)})
    return best_with_path, pair_summaries
