import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple

from ..utils.vector3 import vec3
from ..utils.constants import FARAWAY
from .quantum_search import QSearch

logger = logging.getLogger(__name__)

@dataclass
class QTraceConfig:
    use_image_coherence: bool = True
    use_termination_criterion: bool = True
    max_iterations: int = 3
    shots_per_search: int = 1024
    p_qs_estimate: float = 0.1
    debug: bool = False
    use_fallback: bool = True
    primitive_min_check: int = 2


def trace_ray(ray, scene, config: QTraceConfig = None) -> Tuple[int, float, vec3]:
    if config is None:
        config = QTraceConfig()
    
    if config.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.debug(f"Tracing ray from {ray.origin} in direction {ray.dir}")
    
    primitives = scene.scene_primitives
    
    if not primitives:
        logger.debug("Scene is empty, no intersection possible")
        return -1, FARAWAY, None
    
    min_depth = FARAWAY
    min_primitive_idx = -1
    
    consecutive_non_intersections = 0
    
    iteration = 0
    
    tested_primitives = set()
    
    while iteration < config.max_iterations:
        iteration += 1
        logger.debug(f"Iteration {iteration}/{config.max_iterations}")
        
        q_search = QSearch(ray.origin, ray.dir, primitives, min_depth)
        result = q_search.search(shots=config.shots_per_search)
        
        if result.found:
            tested_primitives.add(result.primitive_idx)
        
        if result.found:
            logger.debug(f"Found intersection with primitive {result.primitive_idx}")
            
            primitive = primitives[result.primitive_idx]
            collider = primitive.collider_list[0]
            
            distance, orientation = collider.intersect(ray.origin, ray.dir)

            if hasattr(distance, '__len__') and len(distance) > 1:
                is_closer = np.all(distance < min_depth)
            else:
                is_closer = distance < min_depth
                
            if is_closer:
                min_depth = distance
                min_primitive_idx = result.primitive_idx
                consecutive_non_intersections = 0
                
                logger.debug(f"Updated minimum depth to {min_depth}")
                
                if config.use_image_coherence and hasattr(scene, 'quantum_neighbor_data'):
                    logger.debug("Checking neighboring pixels for better intersections")
                    
                    neighbor_primitives = scene.quantum_neighbor_data.get(ray, [])
                    
                    for neighbor_idx in neighbor_primitives:
                        if neighbor_idx < len(primitives):
                            tested_primitives.add(neighbor_idx)
                            neighbor = primitives[neighbor_idx]
                            neighbor_collider = neighbor.collider_list[0]
                            
                            n_distance, n_orientation = neighbor_collider.intersect(ray.origin, ray.dir)
                            
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
            
            if config.use_termination_criterion and consecutive_non_intersections > 0:
                p_terminate = config.p_qs_estimate ** consecutive_non_intersections
                
                rand_val = np.random.random()
                
                if rand_val > p_terminate:
                    logger.debug(f"Terminating early after {iteration} iterations (p_terminate={p_terminate})")
                    break
    
    if config.use_fallback and len(tested_primitives) < min(config.primitive_min_check, len(primitives)):
        logger.debug(f"Only tested {len(tested_primitives)} primitives, checking more explicitly")

        untested_primitives = set(range(len(primitives))) - tested_primitives
        if len(untested_primitives) > 5:
            check_primitives = np.random.choice(list(untested_primitives), 
                                               min(5, len(untested_primitives)), 
                                               replace=False)
        else:
            check_primitives = untested_primitives
        
        for idx in check_primitives:
            primitive = primitives[idx]
            collider = primitive.collider_list[0]
        
            distance, orientation = collider.intersect(ray.origin, ray.dir)

            valid_intersection = False
            if hasattr(distance, '__len__') and len(distance) > 1:
                valid_intersection = np.all(distance < FARAWAY) and np.all(distance < min_depth)
            else:
                valid_intersection = distance < FARAWAY and distance < min_depth
                
            if valid_intersection:
                logger.debug(f"Found better intersection through explicit check: primitive {idx} at distance {distance}")
                min_depth = distance
                min_primitive_idx = idx
    
    if min_primitive_idx >= 0:
        primitive = primitives[min_primitive_idx]
        collider = primitive.collider_list[0]
        
        hit_point = ray.origin + ray.dir * min_depth
        
        normal = collider.get_Normal(type('Hit', (), {'point': hit_point}))
        
        logger.debug(f"Final intersection: primitive {min_primitive_idx}, distance {min_depth}")
        
        if config.use_image_coherence and hasattr(scene, 'quantum_neighbor_data') and hasattr(ray, 'pixel_x') and hasattr(ray, 'pixel_y'):
            scene.quantum_neighbor_data.update(ray.pixel_x, ray.pixel_y, min_primitive_idx)
        
        return min_primitive_idx, min_depth, normal
    else:
        logger.debug("No intersection found after all iterations")
        return -1, FARAWAY, None


class QuantumNeighborData:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.full((height, width), -1, dtype=int)
        self.update_count = np.zeros((height, width), dtype=int)
        self.total_updates = 0
    
    def update(self, x, y, primitive_idx):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y, x] = primitive_idx
            self.update_count[y, x] += 1
            self.total_updates += 1
    
    def get_neighbors(self, x, y):
        neighbors = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue 
                    
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_idx = self.data[ny, nx]
                    if neighbor_idx >= 0:
                        neighbors.append(neighbor_idx)
        
        if len(neighbors) > 5:
            neighbor_counts = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                        
                    nx, ny = x + dx, y + dy
                    
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighbor_idx = self.data[ny, nx]
                        if neighbor_idx >= 0:
                            neighbor_counts.append((neighbor_idx, self.update_count[ny, nx]))
            
            neighbor_counts.sort(key=lambda x: x[1], reverse=True)
            
            neighbors = [idx for idx, _ in neighbor_counts[:5]]
        
        return neighbors
    
    def get(self, ray, default=None):
        if hasattr(ray, 'pixel_x') and hasattr(ray, 'pixel_y'):
            return self.get_neighbors(ray.pixel_x, ray.pixel_y)
        return default if default is not None else []