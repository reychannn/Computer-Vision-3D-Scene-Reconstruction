"""Feature matching pipeline for Week 1 deliverable.

This module loads a sequence of images from the assets directory, detects local
features, filters matches with Lowe's ratio test, and writes visualization
images that highlight the surviving correspondences for each image pair.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

ImagePair = Tuple[Path, Path]


@dataclass
class MatchSummary:
    """Container for reporting match statistics."""

    image_a: Path
    image_b: Path
    keypoints_a: int
    keypoints_b: int
    raw_matches: int
    filtered_matches: int
    output_path: Path


def collect_image_paths(
    image_dir: Path,
    prefix: str = "img_",
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    """Return all image files that follow the naming convention."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    candidates = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in extensions
        and path.stem.startswith(prefix)
    ]

    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Need at least two images named '{prefix}x' inside '{image_dir}'."
        )

    return candidates


def create_detector(name: str = "ORB", n_features: int = 2000) -> cv2.Feature2D:
    """Instantiate a feature detector/descriptor."""
    upper_name = name.upper()
    if upper_name == "SIFT":
        return cv2.SIFT_create(n_features)
    if upper_name == "ORB":
        return cv2.ORB_create(nfeatures=n_features, fastThreshold=5)

    raise ValueError(f"Unsupported detector '{name}'. Choose 'ORB' or 'SIFT'.")


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Load an image from disk."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def detect_and_describe(
    detector: cv2.Feature2D, image: np.ndarray
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Run feature detection and description."""
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) == 0:
        raise RuntimeError("No features detected. Verify image quality.")
    return keypoints, descriptors


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio_thresh: float = 0.75,
    norm_type: int | None = None,
) -> Tuple[List[cv2.DMatch], List[cv2.DMatch]]:
    """Perform brute-force matching with optional ratio filtering."""
    if norm_type is None:
        # ORB delivers binary descriptors (uint8), SIFT returns float32.
        norm_type = cv2.NORM_HAMMING if descriptors_a.dtype == np.uint8 else cv2.NORM_L2

    matcher = cv2.BFMatcher(norm_type)
    raw_knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    filtered = [
        m for m, n in raw_knn if n is not None and m.distance < ratio_thresh * n.distance
    ]
    filtered.sort(key=lambda match: match.distance)
    raw = [m for pair in raw_knn for m in pair if m is not None]
    return raw, filtered


def draw_and_save_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a: List[cv2.KeyPoint],
    keypoints_b: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    output_path: Path,
    max_matches: int = 80,
) -> None:
    """Generate a side-by-side visualization of the feature matches."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches_to_draw = matches[:max_matches]
    visualization = cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if not cv2.imwrite(str(output_path), visualization):
        raise IOError(f"Failed to save visualization to {output_path}")


def build_pairs_from_sequence(images: Sequence[Path]) -> List[ImagePair]:
    """Create consecutive image pairs with sufficient parallax."""
    return list(zip(images[:-1], images[1:]))


def build_pairs_from_tokens(images: Sequence[Path], tokens: Iterable[str]) -> List[ImagePair]:
    """Build explicit image pairs from CLI tokens like '1-3'."""
    index_map = {path.stem.split("_")[-1]: path for path in images}
    pairs: List[ImagePair] = []

    for token in tokens:
        if "-" not in token:
            raise ValueError(f"Invalid pair token '{token}'. Use the form '1-2'.")
        left_idx, right_idx = token.split("-", maxsplit=1)
        img_a = index_map.get(left_idx)
        img_b = index_map.get(right_idx)
        if img_a is None or img_b is None:
            raise ValueError(f"Pair '{token}' references missing images.")
        pairs.append((img_a, img_b))
    return pairs


def process_pairs(
    image_pairs: Sequence[ImagePair],
    output_dir: Path,
    detector_name: str = "ORB",
    ratio_thresh: float = 0.75,
) -> List[MatchSummary]:
    """Run the full matching workflow for the requested pairs."""
    detector = create_detector(detector_name)
    summaries: List[MatchSummary] = []

    for img_a_path, img_b_path in image_pairs:
        gray_a = load_image(img_a_path, grayscale=True)
        gray_b = load_image(img_b_path, grayscale=True)

        color_a = load_image(img_a_path, grayscale=False)
        color_b = load_image(img_b_path, grayscale=False)

        kps_a, des_a = detect_and_describe(detector, gray_a)
        kps_b, des_b = detect_and_describe(detector, gray_b)
        raw_matches, filtered_matches = match_descriptors(
            des_a, des_b, ratio_thresh=ratio_thresh
        )

        output_name = f"{img_a_path.stem}_{img_b_path.stem}_matches.jpg"
        output_path = output_dir / output_name

        draw_and_save_matches(
            color_a, color_b, kps_a, kps_b, filtered_matches, output_path
        )

        summaries.append(
            MatchSummary(
                image_a=img_a_path,
                image_b=img_b_path,
                keypoints_a=len(kps_a),
                keypoints_b=len(kps_b),
                raw_matches=len(raw_matches),
                filtered_matches=len(filtered_matches),
                output_path=output_path,
            )
        )

    return summaries


def run_pipeline(
    image_dir: Path,
    output_dir: Path,
    detector: str = "ORB",
    ratio_thresh: float = 0.75,
    pair_tokens: Iterable[str] | None = None,
) -> List[MatchSummary]:
    """Entry-point used by both CLI and notebook."""
    images = collect_image_paths(image_dir)
    image_pairs = (
        build_pairs_from_tokens(images, pair_tokens) if pair_tokens else build_pairs_from_sequence(images)
    )

    summaries = process_pairs(
        image_pairs=image_pairs,
        output_dir=output_dir,
        detector_name=detector,
        ratio_thresh=ratio_thresh,
    )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and visualize filtered feature matches for an image sequence."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("assets"),
        help="Directory containing images named img_<index>.<ext>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/feature_matches"),
        help="Directory that will store the visualization images.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["ORB", "SIFT"],
        default="ORB",
        help="Feature detector/descriptor to use.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio threshold for filtering KNN matches.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Optional list of explicit index pairs like '1-3 2-4'. Defaults to consecutive pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        detector=args.detector,
        ratio_thresh=args.ratio,
        pair_tokens=args.pairs,
    )

    print(f"Processed {len(summaries)} image pair(s).")
    for summary in summaries:
        print(
            f"{summary.image_a.name} vs {summary.image_b.name} -> "
            f"{summary.filtered_matches}/{summary.raw_matches} filtered matches "
            f"(kps: {summary.keypoints_a}/{summary.keypoints_b}) "
            f"saved to {summary.output_path}"
        )


if __name__ == "__main__":
    main()
