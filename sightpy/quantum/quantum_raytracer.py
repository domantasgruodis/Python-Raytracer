"""
Quantum ray tracing implementation based on the paper.
This implements the QTrace algorithm described in the paper.
"""
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..utils.vector3 import vec3
from ..utils.constants import FARAWAY, UPWARDS, UPDOWN
from .quantum_search import QSearch, QSearchResult

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class QTraceConfig:
    """Configuration for quantum ray tracing."""
    use_image_coherence: bool = True  # Whether to use image coherence optimization
    use_termination_criterion: bool = True  # Whether to use the termination criterion
    max_iterations: int = 5  # Maximum number of iterations for QTrace
    shots_per_search: int = 1024  # Number of shots for each quantum search
    p_qs_estimate: float = 0.1  # Estimate of false negative probability
    debug: bool = False  # Whether to print debug information


def trace_ray(ray, scene, config: QTraceConfig = None) -> Tuple[int, float, vec3]:
    """
    Trace a ray using quantum ray tracing.
    
    Args:
        ray: The ray to trace
        scene: The scene containing primitives
        config: Configuration for quantum ray tracing
        
    Returns:
        Tuple containing:
        - Index of the intersected primitive
        - Distance to the intersection
        - Normal vector at the intersection
    """
    if config is None:
        config = QTraceConfig()
    
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.debug(f"Tracing ray from {ray.origin} in direction {ray.dir}")
    
    # Get the list of primitive objects from the scene
    primitives = scene.scene_primitives
    
    # Check if the scene is empty
    if not primitives:
        logger.debug("Scene is empty, no intersection possible")
        return -1, FARAWAY, None
    
    # Current minimum depth
    min_depth = FARAWAY
    min_primitive_idx = -1
    
    # Counter for consecutive non-intersections
    consecutive_non_intersections = 0
    
    # Iteration counter
    iteration = 0
    
    while iteration < config.max_iterations:
        iteration += 1
        logger.debug(f"Iteration {iteration}/{config.max_iterations}")
        
        # Use QSearch to find an intersection
        q_search = QSearch(ray.origin, ray.dir, primitives, min_depth)
        result = q_search.search(shots=config.shots_per_search)
        
        if result.found:
            logger.debug(f"Found intersection with primitive {result.primitive_idx}")
            
            # Get the primitive and its collider
            primitive = primitives[result.primitive_idx]
            collider = primitive.collider_list[0]
            
            # Calculate intersection info
            distance, orientation = collider.intersect(ray.origin, ray.dir)

            # Update minimum depth if this intersection is closer
            # Handle case where distance may be an array
            if hasattr(distance, '__len__') and len(distance) > 1:
                is_closer = np.all(distance < min_depth)
            else:
                is_closer = distance < min_depth
                
            if is_closer:
                min_depth = distance
                min_primitive_idx = result.primitive_idx
                consecutive_non_intersections = 0
                
                logger.debug(f"Updated minimum depth to {min_depth}")
                
                # If using image coherence and there are neighboring pixels
                if config.use_image_coherence and hasattr(scene, 'quantum_neighbor_data'):
                    logger.debug("Checking neighboring pixels for better intersections")
                    
                    # Get neighboring primitives
                    neighbor_primitives = scene.quantum_neighbor_data.get(ray, [])
                    
                    for neighbor_idx in neighbor_primitives:
                        if neighbor_idx < len(primitives):
                            neighbor = primitives[neighbor_idx]
                            neighbor_collider = neighbor.collider_list[0]
                            
                            # Check if this neighbor intersects with the ray
                            n_distance, n_orientation = neighbor_collider.intersect(ray.origin, ray.dir)
                            
                            # Update if this is a closer intersection
                            # Handle case where n_distance may be an array
                            if hasattr(n_distance, '__len__') and len(n_distance) > 1:
                                is_closer = np.all(n_distance < min_depth)
                            else:
                                is_closer = n_distance < min_depth
                                
                            if is_closer:
                                min_depth = n_distance
                                min_primitive_idx = neighbor_idx
                                logger.debug(f"Found better intersection in neighbor {neighbor_idx} at depth {min_depth}")
            else:
                logger.debug(f"Intersection at distance {distance} is not closer than current minimum {min_depth}")
        else:
            logger.debug("No intersection found in this iteration")
            consecutive_non_intersections += 1
            
            # Apply termination criterion if enabled
            if config.use_termination_criterion and consecutive_non_intersections > 0:
                # Calculate probability of termination based on consecutive non-intersections
                p_terminate = config.p_qs_estimate ** consecutive_non_intersections
                
                # Generate random number between 0 and 1
                rand_val = np.random.random()
                
                if rand_val > p_terminate:
                    logger.debug(f"Terminating early after {iteration} iterations (p_terminate={p_terminate})")
                    break
    
    # If we found an intersection, return the primitive index, distance, and normal
    if min_primitive_idx >= 0:
        # Get the primitive and its collider
        primitive = primitives[min_primitive_idx]
        collider = primitive.collider_list[0]
        
        # Calculate intersection point
        hit_point = ray.origin + ray.dir * min_depth
        
        # Calculate normal at intersection point
        normal = collider.get_Normal(type('Hit', (), {'point': hit_point}))
        
        logger.debug(f"Final intersection: primitive {min_primitive_idx}, distance {min_depth}")
        return min_primitive_idx, min_depth, normal
    else:
        logger.debug("No intersection found after all iterations")
        return -1, FARAWAY, None


class QuantumNeighborData:
    """
    Helper class to store and manage image-space coherence data.
    This is used to implement the optimization described in the paper
    where neighboring pixels' intersection results are reused.
    """
    def __init__(self, width, height):
        """
        Initialize the data structure.
        
        Args:
            width (int): Width of the image
            height (int): Height of the image
        """
        self.width = width
        self.height = height
        self.data = np.full((height, width), -1, dtype=int)
    
    def update(self, x, y, primitive_idx):
        """
        Update the primitive index at a pixel position.
        
        Args:
            x (int): X coordinate of the pixel
            y (int): Y coordinate of the pixel
            primitive_idx (int): Index of the primitive
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y, x] = primitive_idx
    
    def get_neighbors(self, x, y):
        """
        Get the primitive indices of neighboring pixels.
        
        Args:
            x (int): X coordinate of the pixel
            y (int): Y coordinate of the pixel
            
        Returns:
            list: List of primitive indices (may contain -1 for no intersection)
        """
        neighbors = []
        
        # Check 4-connected neighbors (von Neumann neighborhood)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbor_idx = self.data[ny, nx]
                if neighbor_idx >= 0:
                    neighbors.append(neighbor_idx)
        
        return neighbors
    
    def get(self, ray, default=None):
        """
        Get the neighboring primitive indices for a ray.
        This assumes ray has pixel coordinates stored in its metadata.
        
        Args:
            ray: The ray to get neighbors for
            default: Default value to return if no neighbors found
            
        Returns:
            list: List of primitive indices
        """
        if hasattr(ray, 'pixel_x') and hasattr(ray, 'pixel_y'):
            return self.get_neighbors(ray.pixel_x, ray.pixel_y)
        return default if default is not None else []