"""View graph construction for virtual tour navigation."""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ViewEdge:
    """Edge connecting two camera views."""
    camera_a: int
    camera_b: int
    shared_points: int
    weight: float  # normalized by min(points_a, points_b)


def build_view_graph(
    point_lookup: Dict[Tuple[int, int], int],
    poses: List,
    min_shared: int = 30
) -> Dict[int, List[ViewEdge]]:
    """
    Build a view graph where nodes are cameras and edges connect views
    that share >= min_shared 3D points.
    
    Args:
        point_lookup: dict mapping (img_idx, kp_idx) to 3D point ID
        poses: list of CameraPose objects
        min_shared: minimum number of shared points to create an edge
    
    Returns:
        adjacency list: {camera_idx: [ViewEdge, ...]}
    """
    # Group observations by camera
    observations: Dict[int, Set[int]] = {}
    for (img_idx, kp_idx), pid in point_lookup.items():
        observations.setdefault(img_idx, set()).add(pid)
    
    # Initialize graph with only registered cameras
    graph: Dict[int, List[ViewEdge]] = {}
    for i, pose in enumerate(poses):
        if pose.registered:
            graph[i] = []
    
    # Compare all camera pairs
    cameras = sorted(observations.keys())
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            cam_a, cam_b = cameras[i], cameras[j]
            
            # Skip if either camera is not registered
            if cam_a not in graph or cam_b not in graph:
                continue
            
            points_a = observations.get(cam_a, set())
            points_b = observations.get(cam_b, set())
            
            shared = len(points_a & points_b)
            if shared < min_shared:
                continue
            
            # Normalize by smaller point count to favor well-connected pairs
            weight = shared / min(len(points_a), len(points_b)) if min(len(points_a), len(points_b)) > 0 else 0
            
            edge = ViewEdge(cam_a, cam_b, shared, weight)
            graph[cam_a].append(edge)
            graph[cam_b].append(edge)
    
    return graph


def find_best_neighbor(
    current_idx: int,
    graph: Dict[int, List[ViewEdge]],
    poses: List,
    user_direction: np.ndarray | None = None
) -> int | None:
    """
    Find the best camera to navigate to from current_idx.
    
    Args:
        current_idx: current camera index
        graph: view graph from build_view_graph
        poses: list of CameraPose objects
        user_direction: optional 3D direction vector (in world coords)
    
    Returns:
        index of best neighbor camera, or None if no valid neighbors
    """
    edges = graph.get(current_idx, [])
    if not edges:
        return None
    
    # Sort by shared point count (descending)
    edges = sorted(edges, key=lambda e: e.shared_points, reverse=True)
    
    if user_direction is None:
        # No user input: just pick most connected neighbor
        neighbor = edges[0].camera_b if edges[0].camera_a == current_idx else edges[0].camera_a
        return neighbor
    
    # Score each neighbor by alignment with user direction
    current_t = poses[current_idx].t.ravel()
    best_score = -np.inf
    best_neighbor = None
    
    for edge in edges:
        neighbor_idx = edge.camera_b if edge.camera_a == current_idx else edge.camera_a
        neighbor_t = poses[neighbor_idx].t.ravel()
        
        # Direction from current to neighbor
        direction = neighbor_t - current_t
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            continue
        direction /= direction_norm
        
        # Score = dot product (cosine similarity) * shared point weight
        alignment = np.dot(direction, user_direction / np.linalg.norm(user_direction))
        score = alignment * edge.weight
        
        if score > best_score:
            best_score = score
            best_neighbor = neighbor_idx
    
    return best_neighbor