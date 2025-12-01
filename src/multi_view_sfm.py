"""Incremental multi-view SfM with lightweight refinement.

This module reuses the feature utilities from `feature_matching_pipeline.py`
and the intrinsics/triangulation helpers from `two_view_reconstruction.py`
to build a simple multi-view reconstruction:

1) Detect SIFT/ORB features for every image in the sequence.
2) Bootstrap from the first two images with an Essential matrix + cheirality check.
3) Incrementally register the remaining views with PnP (RANSAC) using matches
   to already-triangulated points.
4) Triangulate new points against the best-matching registered neighbor.
5) Run a lightweight per-view pose refinement (`solvePnPRefineLM`) to reduce drift.
6) Export the refined sparse point cloud to a PLY file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.feature_matching_pipeline import (
    collect_image_paths,
    create_detector,
    detect_and_describe,
    load_image,
    match_descriptors,
)
from src.two_view_reconstruction import approximate_intrinsics, save_ply


@dataclass
class CameraPose:
    """Pose of a registered view."""

    R: np.ndarray  # 3x3 rotation (world -> camera)
    t: np.ndarray  # 3x1 translation (world -> camera)
    image_path: Path
    registered: bool = True


@dataclass
class FeatureSet:
    """Precomputed keypoints and descriptors for an image."""

    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class IncrementalResult:
    """Container summarizing the reconstruction."""

    K: np.ndarray
    poses: List[CameraPose]
    points_3d: np.ndarray
    colors_rgb: np.ndarray
    ply_path: Optional[Path]
    stats: dict


def _triangulate_pair(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    R_ab: np.ndarray,
    t_ab: np.ndarray,
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate points between two arbitrary camera poses.

    pts_a/pts_b are in pixel coords for cameras A (reference) and B.
    R_ab, t_ab express the transform from camera A to camera B.
    R_ref, t_ref are the absolute pose (world -> camera) for camera A.
    Returns:
        world_points: Nx3 coordinates in the global (camera-0) frame.
        valid_mask: boolean mask for cheirality in both cameras.
    """

    # Normalize image points.
    pts_a_norm = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), K, None).reshape(-1, 2).T
    pts_b_norm = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), K, None).reshape(-1, 2).T

    # Triangulate in the reference camera's frame.
    proj_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    proj_b = np.hstack([R_ab, t_ab])
    homog = cv2.triangulatePoints(proj_a, proj_b, pts_a_norm, pts_b_norm)
    pts_cam_a = (homog[:3] / homog[3]).T  # Nx3 in camera A frame

    # Depth in both cameras.
    pts_cam_b = (R_ab @ pts_cam_a.T + t_ab).T
    valid = (pts_cam_a[:, 2] > 0) & (pts_cam_b[:, 2] > 0)

    # Lift to the global (camera-0) frame.
    world_pts = (R_ref.T @ (pts_cam_a.T - t_ref)).T
    return world_pts, valid


def _relative_pose(R_j: np.ndarray, t_j: np.ndarray, R_i: np.ndarray, t_i: np.ndarray):
    """Compute pose of camera i expressed in camera j frame."""

    R_rel = R_i @ R_j.T
    t_rel = t_i - R_rel @ t_j
    return R_rel, t_rel


def _sample_colors(image: np.ndarray, keypoints: List[cv2.KeyPoint], indices: np.ndarray) -> np.ndarray:
    """Gather RGB colors at specified keypoint indices, clipping to bounds."""

    pts = np.round(np.array([keypoints[idx].pt for idx in indices], dtype=int))
    pts[:, 0] = np.clip(pts[:, 0], 0, image.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, image.shape[0] - 1)
    # Convert BGR (OpenCV) to RGB for export.
    return image[pts[:, 1], pts[:, 0], ::-1]


def _load_features(
    image_paths: Sequence[Path],
    detector_name: str = "SIFT",
    n_features: int = 4000,
) -> Tuple[List[np.ndarray], List[FeatureSet]]:
    """Load images and precompute features."""

    detector = create_detector(detector_name, n_features)
    images_bgr: List[np.ndarray] = []
    features: List[FeatureSet] = []
    for path in image_paths:
        img = load_image(path, grayscale=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = detect_and_describe(detector, gray)
        images_bgr.append(img)
        features.append(FeatureSet(kps, desc))
    return images_bgr, features


def _gather_correspondences(
    current_idx: int,
    registered_indices: Iterable[int],
    features: List[FeatureSet],
    point_lookup: Dict[Tuple[int, int], int],
    ratio: float = 0.75,
) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Match the current image to registered ones and collect 3D-2D pairs.

    Returns:
        point_ids: list of unique point ids with matches.
        kp_indices: list of keypoint indices in the current image (aligned with point_ids).
        img_pts: list of 2D pixels in current image.
    """

    best: Dict[int, Tuple[float, int, np.ndarray]] = {}

    desc_i = features[current_idx].descriptors
    kps_i = features[current_idx].keypoints

    for ref_idx in registered_indices:
        if ref_idx == current_idx:
            continue
        desc_ref = features[ref_idx].descriptors
        if desc_i is None or desc_ref is None:
            continue
        _raw, matches = match_descriptors(desc_i, desc_ref, ratio_thresh=ratio, norm_type=None)
        for m in matches:
            pid = point_lookup.get((ref_idx, m.trainIdx))
            if pid is None:
                continue
            score = float(m.distance)
            if pid in best and score >= best[pid][0]:
                continue  # keep best match per point id
            best[pid] = (score, m.queryIdx, np.array(kps_i[m.queryIdx].pt, dtype=np.float64))

    point_ids = list(best.keys())
    kp_indices = [best[pid][1] for pid in point_ids]
    img_pts = [best[pid][2] for pid in point_ids]
    return point_ids, kp_indices, img_pts


def _build_obj_points(point_ids: List[int], point_cloud: List[np.ndarray]) -> List[np.ndarray]:
    """Fill object points for gathered correspondences."""

    return [point_cloud[pid] for pid in point_ids]


def _pose_from_pnp(
    obj_pts: List[np.ndarray],
    img_pts: List[np.ndarray],
    K: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Estimate camera pose with PnP + refinement."""

    if len(obj_pts) < 6:
        return None
    obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=8.0,
        confidence=0.999,
        iterationsCount=2000,
    )
    if not success or inliers is None or len(inliers) < 6:
        return None
    inliers = inliers.ravel()
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(obj[inliers], img[inliers], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec_refined)
    t = tvec_refined.reshape(3, 1)
    return R, t, inliers


def _projection_matrix(cam: CameraPose, K: np.ndarray) -> np.ndarray:
    """Build a 3x4 projection matrix for a camera pose."""

    return K @ np.hstack([cam.R, cam.t])


def _retriangulate_points(
    point_cloud: List[np.ndarray],
    point_lookup: Dict[Tuple[int, int], int],
    cams: List[CameraPose],
    features: List[FeatureSet],
    K: np.ndarray,
) -> Tuple[List[np.ndarray], int]:
    """Refine 3D points with multi-view linear triangulation across all observations.

    Points keep their original indexing; only coordinates are updated for points
    with at least two registered-view observations and positive depth in all
    contributing cameras.
    """

    new_points = list(point_cloud)
    improved = 0

    # Build observation lists per point id.
    observations: Dict[int, List[Tuple[int, int]]] = {}
    for (img_idx, kp_idx), pid in point_lookup.items():
        observations.setdefault(pid, []).append((img_idx, kp_idx))

    for pid, obs in observations.items():
        if len(obs) < 2:
            continue
        A_rows = []
        proj_mats = []
        for img_idx, kp_idx in obs:
            if img_idx >= len(cams):
                continue
            cam = cams[img_idx]
            if not cam.registered:
                continue
            P = _projection_matrix(cam, K)
            proj_mats.append(P)
            pt = features[img_idx].keypoints[kp_idx].pt
            pt_norm = cv2.undistortPoints(
                np.array(pt, dtype=np.float64).reshape(1, 1, 2), K, None
            )[0, 0]
            x, y = pt_norm
            A_rows.append(x * P[2] - P[0])
            A_rows.append(y * P[2] - P[1])

        if len(proj_mats) < 2:
            continue

        A = np.asarray(A_rows, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        if abs(X_h[3]) < 1e-9:
            continue
        X = X_h[:3] / X_h[3]

        depths = [(P @ X_h)[2] for P in proj_mats]
        if min(depths) <= 0:
            continue

        new_points[pid] = X
        improved += 1

    return new_points, improved


def _register_initial_pair(
    idx_a: int,
    idx_b: int,
    features: List[FeatureSet],
    images_bgr: List[np.ndarray],
    K: np.ndarray,
    ratio: float = 0.75,
) -> Tuple[CameraPose, CameraPose, List[np.ndarray], List[np.ndarray], Dict[Tuple[int, int], int]]:
    """Bootstrap with an Essential matrix and triangulation."""

    kps_a, des_a = features[idx_a].keypoints, features[idx_a].descriptors
    kps_b, des_b = features[idx_b].keypoints, features[idx_b].descriptors
    _raw, matches = match_descriptors(des_a, des_b, ratio_thresh=ratio, norm_type=None)
    if len(matches) < 8:
        raise RuntimeError("Not enough matches in the initial pair.")

    pts_a = np.float64([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float64([kps_b[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None:
        raise RuntimeError("Failed to estimate Essential matrix for the initial pair.")
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    inliers = mask_pose.ravel().astype(bool)
    if inliers.sum() < 20:
        raise RuntimeError("Insufficient inliers after recoverPose for the initial pair.")

    pts_a_in = pts_a[inliers]
    pts_b_in = pts_b[inliers]

    world_pts, valid = _triangulate_pair(
        pts_a_in,
        pts_b_in,
        R_ab=R,
        t_ab=t,
        R_ref=np.eye(3),
        t_ref=np.zeros((3, 1)),
        K=K,
    )
    world_pts = world_pts[valid]
    if len(world_pts) == 0:
        raise RuntimeError("Triangulation produced no valid points for the initial pair.")

    # Prepare tracks and colors.
    point_lookup: Dict[Tuple[int, int], int] = {}
    point_cloud: List[np.ndarray] = []
    colors: List[np.ndarray] = []

    valid_idx = np.nonzero(inliers)[0][valid]
    kp_idx_a = np.array([matches[i].queryIdx for i in valid_idx], dtype=int)
    kp_idx_b = np.array([matches[i].trainIdx for i in valid_idx], dtype=int)

    clr = _sample_colors(images_bgr[idx_a], kps_a, kp_idx_a)
    for pid, (pt, ca, cb, col) in enumerate(zip(world_pts, kp_idx_a, kp_idx_b, clr)):
        point_cloud.append(pt)
        colors.append(col)
        point_lookup[(idx_a, ca)] = pid
        point_lookup[(idx_b, cb)] = pid

    cam0 = CameraPose(R=np.eye(3), t=np.zeros((3, 1)), image_path=Path(""), registered=True)
    cam1 = CameraPose(R=R, t=t, image_path=Path(""), registered=True)
    return cam0, cam1, point_cloud, colors, point_lookup


def run_incremental_sfm(
    asset_dir: Path,
    detector: str = "SIFT",
    ratio_thresh: float = 0.75,
    output_path: Optional[Path] = None,
    refine: bool = True,
    min_correspondences: int = 12,
) -> IncrementalResult:
    """Incrementally register all images in `asset_dir` and return a point cloud."""

    image_paths = collect_image_paths(asset_dir)
    if len(image_paths) < 2:
        raise FileNotFoundError("Need at least two images to run SfM.")

    # Precompute features and intrinsics.
    images_bgr, features = _load_features(image_paths, detector_name=detector)
    K = approximate_intrinsics(images_bgr[0])

    # Bootstrap.
    cam0, cam1, point_cloud, colors, point_lookup = _register_initial_pair(
        idx_a=0,
        idx_b=1,
        features=features,
        images_bgr=images_bgr,
        K=K,
        ratio=ratio_thresh,
    )
    cams: List[CameraPose] = [cam0, cam1]
    cams[0].image_path = image_paths[0]
    cams[1].image_path = image_paths[1]

    registered_indices = {0, 1}
    skipped: List[int] = []
    pose_inliers: Dict[int, int] = {0: 0, 1: 0}

    # Process remaining images in sequence order.
    for idx in range(2, len(image_paths)):
        desc_i = features[idx].descriptors
        if desc_i is None:
            skipped.append(idx)
            continue

        # Gather 3D-2D correspondences.
        point_ids, kp_indices, img_pts = _gather_correspondences(
            current_idx=idx,
            registered_indices=registered_indices,
            features=features,
            point_lookup=point_lookup,
            ratio=ratio_thresh,
        )
        obj_pts = _build_obj_points(point_ids, point_cloud)

        if len(obj_pts) < min_correspondences:
            skipped.append(idx)
            cams.append(CameraPose(R=np.eye(3), t=np.zeros((3, 1)), image_path=image_paths[idx], registered=False))
            continue

        pose_est = _pose_from_pnp(obj_pts, img_pts, K)
        if pose_est is None:
            skipped.append(idx)
            cams.append(CameraPose(R=np.eye(3), t=np.zeros((3, 1)), image_path=image_paths[idx], registered=False))
            continue

        R_i, t_i, inliers = pose_est
        cams.append(CameraPose(R=R_i, t=t_i, image_path=image_paths[idx], registered=True))
        registered_indices.add(idx)
        pose_inliers[idx] = int(len(inliers))

        # Record inlier observations.
        for loc in inliers:
            pid = point_ids[loc]
            kp_idx = kp_indices[loc]
            point_lookup[(idx, kp_idx)] = pid

        # Triangulate new points against the best matching previous view (last registered for simplicity).
        ref_idx = max(registered_indices - {idx})
        kps_ref = features[ref_idx].keypoints
        kps_i = features[idx].keypoints
        desc_ref = features[ref_idx].descriptors
        if desc_ref is None or desc_i is None:
            continue
        _raw_tri, matches_tri = match_descriptors(desc_ref, desc_i, ratio_thresh=ratio_thresh, norm_type=None)
        if len(matches_tri) == 0:
            continue

        # Build lists for unmatched keypoints only.
        pts_ref = []
        pts_i = []
        kp_idx_ref = []
        kp_idx_i = []
        for m in matches_tri:
            if (ref_idx, m.queryIdx) in point_lookup or (idx, m.trainIdx) in point_lookup:
                continue
            pts_ref.append(kps_ref[m.queryIdx].pt)
            pts_i.append(kps_i[m.trainIdx].pt)
            kp_idx_ref.append(m.queryIdx)
            kp_idx_i.append(m.trainIdx)

        if len(pts_ref) == 0:
            continue

        R_ref, t_ref = cams[ref_idx].R, cams[ref_idx].t
        R_rel, t_rel = _relative_pose(R_ref, t_ref, R_i, t_i)
        pts_ref_np = np.asarray(pts_ref, dtype=np.float64)
        pts_i_np = np.asarray(pts_i, dtype=np.float64)
        world_pts, valid_mask = _triangulate_pair(
            pts_ref_np,
            pts_i_np,
            R_ab=R_rel,
            t_ab=t_rel,
            R_ref=R_ref,
            t_ref=t_ref,
            K=K,
        )

        if valid_mask.sum() == 0:
            continue

        valid_world = world_pts[valid_mask]
        kp_ref_valid = np.asarray(kp_idx_ref, dtype=int)[valid_mask]
        kp_i_valid = np.asarray(kp_idx_i, dtype=int)[valid_mask]
        clr = _sample_colors(images_bgr[ref_idx], kps_ref, kp_ref_valid)

        offset = len(point_cloud)
        for j, pt in enumerate(valid_world):
            pid = offset + j
            point_cloud.append(pt)
            colors.append(clr[j])
            point_lookup[(ref_idx, kp_ref_valid[j])] = pid
            point_lookup[(idx, kp_i_valid[j])] = pid

    # Optional per-view refinement (lightweight drift reduction).
    if refine:
        for idx, cam in enumerate(cams):
            if not cam.registered or idx == 0:
                continue
            # Build correspondences from observations.
            obj_pts = []
            img_pts = []
            for (img_idx, kp_idx), pid in point_lookup.items():
                if img_idx != idx:
                    continue
                obj_pts.append(point_cloud[pid])
                img_pts.append(np.array(features[idx].keypoints[kp_idx].pt, dtype=np.float64))
            if len(obj_pts) < 6:
                continue
            obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
            img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
            rvec, _ = cv2.Rodrigues(cam.R)
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(obj, img, K, None, rvec, cam.t)
            R_refined, _ = cv2.Rodrigues(rvec_ref)
            cam.R, cam.t = R_refined, tvec_ref

        # Re-triangulate all points using all available observations.
        point_cloud, improved = _retriangulate_points(point_cloud, point_lookup, cams, features, K)
        stats_improved = improved
    else:
        stats_improved = 0

    points_arr = np.asarray(point_cloud, dtype=np.float64)
    colors_arr = np.asarray(colors, dtype=np.float64) if len(colors) > 0 else None

    ply_path: Optional[Path] = None
    if output_path is None:
        output_path = asset_dir.parent / "outputs" / "reconstruction" / "multi_view_points_refined.ply"
    ply_path = Path(save_ply(output_path, points_arr, colors_arr))

    stats = {
        "images": len(image_paths),
        "registered": len(registered_indices),
        "skipped": skipped,
        "points": len(points_arr),
        "pose_inliers": pose_inliers,
        "retriangulated": stats_improved,
    }

    return IncrementalResult(
        K=K,
        poses=cams,
        points_3d=points_arr,
        colors_rgb=colors_arr if colors_arr is not None else np.empty((0, 3)),
        ply_path=ply_path,
        stats=stats,
    )


__all__ = ["run_incremental_sfm", "IncrementalResult", "CameraPose"]
