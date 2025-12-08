"""Interactive virtual tour viewer with smooth camera transitions."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from dataclasses import dataclass

from src.multi_view_sfm import IncrementalResult, CameraPose
from src.view_graph import build_view_graph, find_best_neighbor
from src.interpolation import interpolate_pose


@dataclass
class TourState:
    """Current state of the virtual tour."""
    current_camera: int
    target_camera: Optional[int]
    transition_progress: float  # 0.0 to 1.0
    is_transitioning: bool


class VirtualTourViewer:
    """Interactive viewer for navigating a reconstructed scene."""
    
    def __init__(self, result: IncrementalResult, image_dir: Path):
        """
        Initialize the virtual tour viewer.
        
        Args:
            result: IncrementalResult from run_incremental_sfm
            image_dir: directory containing source images
        """
        self.result = result
        self.image_dir = image_dir
        
        # Load all source images
        self.images = self._load_images()
        
        # Build view graph
        self.view_graph = build_view_graph(
            point_lookup=result.point_lookup,
            poses=result.poses,
            min_shared=30
        )
        
        # Find first registered camera
        first_registered = next((i for i, p in enumerate(result.poses) if p.registered), 0)
        
        # Tour state
        self.state = TourState(
            current_camera=first_registered,
            target_camera=None,
            transition_progress=0.0,
            is_transitioning=False
        )
        
        # Animation parameters
        self.transition_duration = 2.0  # seconds
        self.fps = 30
        self.frame_interval = 1000 // self.fps  # milliseconds
        
        # UI setup
        self.setup_ui()
    
    def _load_images(self) -> list[np.ndarray]:
        """Load all registered camera images."""
        images = []
        for cam in self.result.poses:
            if not cam.registered:
                continue
            
            img_path = self.image_dir / cam.image_path.name
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        return images
    
    def setup_ui(self):
        """Create the Tkinter window and matplotlib canvas."""
        self.root = tk.Tk()
        self.root.title("Virtual Tour Viewer")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.axis('off')
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Display initial image
        # Map current_camera index to images list
        registered_indices = [i for i, p in enumerate(self.result.poses) if p.registered]
        img_idx = registered_indices.index(self.state.current_camera)
        self.image_display = self.ax.imshow(self.images[img_idx])
        self.canvas.draw()
        
        # Bind keyboard/mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.bind('<Left>', lambda e: self.navigate('left'))
        self.root.bind('<Right>', lambda e: self.navigate('right'))
        self.root.bind('<Up>', lambda e: self.navigate('forward'))
        self.root.bind('<Down>', lambda e: self.navigate('backward'))
        self.root.bind('q', lambda e: self.root.quit())
        
        # Animation timer
        self.timer = None
    
    def on_click(self, event):
        """Handle mouse click to navigate to neighboring view."""
        if event.inaxes != self.ax or self.state.is_transitioning:
            return
        
        neighbors = self.view_graph.get(self.state.current_camera, [])
        if not neighbors:
            print(f"No neighbors for camera {self.state.current_camera}")
            return
        
        next_cam = find_best_neighbor(
            self.state.current_camera,
            self.view_graph,
            self.result.poses,
            user_direction=None
        )
        
        if next_cam is not None:
            self.start_transition(next_cam)
    
    def navigate(self, direction: str):
        """Navigate using keyboard arrows."""
        if self.state.is_transitioning:
            return
        
        # Get list of registered camera indices
        registered_indices = [i for i, p in enumerate(self.result.poses) if p.registered]
        
        if self.state.current_camera not in registered_indices:
            return
        
        current_pos = registered_indices.index(self.state.current_camera)
        
        if direction == 'right':
            next_pos = (current_pos + 1) % len(registered_indices)
            next_cam = registered_indices[next_pos]
        elif direction == 'left':
            next_pos = (current_pos - 1) % len(registered_indices)
            next_cam = registered_indices[next_pos]
        else:
            # Use view graph for forward/backward
            next_cam = find_best_neighbor(self.state.current_camera, self.view_graph, self.result.poses)
        
        if next_cam is not None and next_cam != self.state.current_camera:
            self.start_transition(next_cam)
    
    def start_transition(self, target_camera: int):
        """Begin animated transition to target camera."""
        self.state.target_camera = target_camera
        self.state.transition_progress = 0.0
        self.state.is_transitioning = True
        
        # Start animation loop
        self.animate_transition()
    
    def animate_transition(self):
        """Render one frame of the transition animation."""
        if not self.state.is_transitioning:
            return
        
        # Update progress
        delta = 1.0 / (self.transition_duration * self.fps)
        self.state.transition_progress += delta
        
        if self.state.transition_progress >= 1.0:
            # Transition complete
            self.state.current_camera = self.state.target_camera
            self.state.is_transitioning = False
            self.state.transition_progress = 1.0
        
        # Render current frame
        self.render_frame()
        
        # Schedule next frame
        if self.state.is_transitioning:
            self.timer = self.root.after(self.frame_interval, self.animate_transition)
    
    def render_frame(self):
        """Render the current view (blended image + point cloud overlay)."""
        alpha = self.state.transition_progress
        
        # Get current and target poses
        cam_start = self.result.poses[self.state.current_camera]
        cam_end = self.result.poses[self.state.target_camera]
        
        # Interpolate pose
        R_interp, t_interp = interpolate_pose(
            cam_start.R, cam_start.t,
            cam_end.R, cam_end.t,
            alpha
        )
        
        # Get image indices in the images list
        registered_indices = [i for i, p in enumerate(self.result.poses) if p.registered]
        img_start_idx = registered_indices.index(self.state.current_camera)
        img_end_idx = registered_indices.index(self.state.target_camera)
        
        # Cross-fade images
        img_start = self.images[img_start_idx]
        img_end = self.images[img_end_idx]
        blended = ((1 - alpha) * img_start + alpha * img_end).astype(np.uint8)
        
        # Project and overlay point cloud
        pixels, depths, colors = self.project_point_cloud(
            R_interp, t_interp,
            self.result.points_3d,
            self.result.colors_rgb
        )
        
        # Draw points on blended image
        overlay = blended.copy()
        for (x, y), col in zip(pixels.astype(int), colors.astype(int)):
            cv2.circle(overlay, (x, y), 2, col.tolist(), -1)
        
        # Fade point cloud visibility (show more during mid-transition)
        point_alpha = 1.0 - abs(2 * alpha - 1)  # peaks at alpha=0.5
        final = cv2.addWeighted(blended, 1 - point_alpha, overlay, point_alpha, 0)
        
        # Update display
        self.image_display.set_data(final)
        self.canvas.draw()
    
    def project_point_cloud(
        self,
        R: np.ndarray,
        t: np.ndarray,
        points_3d: np.ndarray,
        colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points onto the current interpolated camera view.
        
        Returns:
            (pixel_coords, depths, colors) for points in front of the camera
        """
        K = self.result.K
        
        # Transform to camera frame
        points_cam = (R @ points_3d.T + t).T
        
        # Filter points behind camera
        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]
        colors_valid = colors[valid]
        
        if len(points_cam) == 0:
            return np.empty((0, 2)), np.empty(0), np.empty((0, 3))
        
        # Project to image plane
        pixels_hom = (K @ points_cam.T).T
        pixels = pixels_hom[:, :2] / pixels_hom[:, 2:3]
        depths = points_cam[:, 2]
        
        # Filter points outside image bounds
        h, w = self.images[0].shape[:2]
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        
        return pixels[in_bounds], depths[in_bounds], colors_valid[in_bounds]
    
    def run(self):
        """Start the viewer main loop."""
        self.root.mainloop()