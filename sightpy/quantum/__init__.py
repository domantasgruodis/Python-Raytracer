from .quantum_raytracer import trace_ray, QTraceConfig
from .quantum_search import QSearch, QSearchResult
from .quantum_oracle import build_intersection_oracle
from .quantum_utils import configure_simulator_options
from .scene_patch import patch_scene_class

__all__ = [
    'trace_ray',
    'QTraceConfig',
    'QSearch',
    'QSearchResult',
    'build_intersection_oracle',
    'configure_simulator_options',
    'create_ray_for_pixel',
    'find_intersection_classical',
    'trace_ray_with_logging',
    'visualize_differences',
    'analyze_results',
    'print_analysis',
    'debug_quantum_rendering',
    'QTraceConfigDebug',
    'trace_ray_debug',
    'compare_with_classical',
    'patch_scene_class'
]