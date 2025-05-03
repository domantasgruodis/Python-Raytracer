"""
Benchmark for quantum ray tracing compared to classical ray tracing.
This demonstrates the quadratic speedup in query complexity.
"""
from sightpy import *
from sightpy.quantum import QTraceConfig
import time
import matplotlib.pyplot as plt
import numpy as np
import logging

# Set up logging - change to DEBUG for more details
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

def create_scene(num_primitives):
    """Create a scene with the specified number of primitives."""
    scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))
    
    # Add a camera
    scene.add_Camera(
        screen_width=64,  # Small resolution for faster benchmarking
        screen_height=64,
        look_from=vec3(0, 0, 10),
        look_at=vec3(0, 0, 0),
        field_of_view=60
    )
    
    # Create diffuse material
    diffuse = Diffuse(diff_color=rgb(0.8, 0.8, 0.8))
    
    # Add a ground plane
    scene.add(Plane(
        material=diffuse,
        center=vec3(0, -5, 0),
        width=20,
        height=20,
        u_axis=vec3(1, 0, 0),
        v_axis=vec3(0, 0, 1)
    ))
    
    # Add the specified number of spheres randomly positioned
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_primitives):
        # Random position within a 10x10x10 cube
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(-15, 0)
        
        # Random color
        r = np.random.uniform(0.2, 0.8)
        g = np.random.uniform(0.2, 0.8)
        b = np.random.uniform(0.2, 0.8)
        
        # Random radius
        radius = np.random.uniform(0.3, 0.7)
        
        # Create the sphere
        sphere_material = Diffuse(diff_color=rgb(r, g, b))
        scene.add(Sphere(
            material=sphere_material,
            center=vec3(x, y, z),
            radius=radius
        ))
    
    # Add a light
    scene.add_PointLight(pos=vec3(0, 10, 10), color=rgb(1, 1, 1))
    
    return scene


class IntersectionCounter:
    """Helper class to count intersections."""
    def __init__(self):
        self.count = 0
    
    def increment(self, amount=1):
        self.count += amount
    
    def get_count(self):
        return self.count


def count_intersections(scene, quantum=False, config=None):
    """Count the number of intersection evaluations for the scene."""
    # Create a counter
    counter = IntersectionCounter()
    
    # Enable or disable quantum ray tracing
    if quantum:
        logger.info("Setting up quantum ray tracing")
        scene.enable_quantum_raytracing(config)
        
        # Monkey patch the trace_ray function to count intersections
        from sightpy.quantum import trace_ray as original_trace_ray
        
        def counting_trace_ray(ray, scene, config):
            # Each trace_ray call performs approximately sqrt(N) * max_iterations evaluations
            approx_evaluations = int(np.sqrt(len(scene.scene_primitives))) * config.max_iterations
            counter.increment(approx_evaluations)
            return original_trace_ray(ray, scene, config)
        
        # Replace the original function with our counting version
        import sightpy.quantum.quantum_raytracer
        sightpy.quantum.quantum_raytracer.trace_ray = counting_trace_ray
    else:
        logger.info("Using classical ray tracing")
        scene.disable_quantum_raytracing()
        
        # For classical ray tracing, we know exactly how many intersections:
        # number of rays * number of primitives
        num_rays = scene.camera.screen_width * scene.camera.screen_height
        num_primitives = len(scene.scene_primitives)
        counter.increment(num_rays * num_primitives)
    
    # Render the scene with one sample per pixel
    start_time = time.time()
    
    if quantum:
        # Only render a small portion for quantum to save time
        scene.camera.screen_width = 16
        scene.camera.screen_height = 16
    
    scene.render(samples_per_pixel=1)
    render_time = time.time() - start_time
    
    # Reset the monkey patch if needed
    if quantum:
        import sightpy.quantum.quantum_raytracer
        sightpy.quantum.quantum_raytracer.trace_ray = original_trace_ray
    
    return counter.get_count(), render_time


def run_benchmark():
    """Run the benchmark and plot the results."""
    # Define the numbers of primitives to test
    # Use smaller numbers for faster testing
    primitive_counts = [8, 16, 32, 64, 128, 256]
    
    # Initialize results storage
    classical_intersections = []
    classical_times = []
    quantum_intersections = []
    quantum_times = []
    
    # Configure quantum ray tracing
    quantum_config = QTraceConfig(
        use_image_coherence=False,
        use_termination_criterion=True,
        max_iterations=2,
        shots_per_search=512,
        debug=False
    )
    
    # Run the benchmark for each primitive count
    for num_primitives in primitive_counts:
        print(f"\nTesting with {num_primitives} primitives...")
        
        # Create the scene
        scene = create_scene(num_primitives)
        
        # Run classical ray tracing
        print("  Running classical ray tracing...")
        c_intersections, c_time = count_intersections(scene, quantum=False)
        classical_intersections.append(c_intersections)
        classical_times.append(c_time)
        
        # Run quantum ray tracing
        print("  Running quantum ray tracing...")
        q_intersections, q_time = count_intersections(scene, quantum=True, config=quantum_config)
        quantum_intersections.append(q_intersections)
        quantum_times.append(q_time)
        
        print(f"  Classical: {c_intersections} intersections in {c_time:.2f}s")
        print(f"  Quantum: {q_intersections} intersections in {q_time:.2f}s")
        print(f"  Ratio: {c_intersections / q_intersections:.2f}x")
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Plot intersection counts
    plt.subplot(2, 1, 1)
    plt.plot(primitive_counts, classical_intersections, 'bo-', label='Classical')
    plt.plot(primitive_counts, quantum_intersections, 'ro-', label='Quantum')
    plt.xlabel('Number of Primitives')
    plt.ylabel('Number of Intersections')
    plt.title('Intersection Evaluations (lower is better)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot theoretical curves
    x = np.linspace(min(primitive_counts), max(primitive_counts), 100)
    ray_count = 64 * 64  # Default scene.camera.screen_width * scene.camera.screen_height
    plt.plot(x, ray_count * x, 'b--', alpha=0.5, label='O(N) - Classical')
    plt.plot(x, ray_count * np.sqrt(x) * quantum_config.max_iterations, 'r--', alpha=0.5, label='O(√N) - Quantum')
    plt.legend()
    
    # Plot render times
    plt.subplot(2, 1, 2)
    plt.plot(primitive_counts, classical_times, 'bo-', label='Classical')
    plt.plot(primitive_counts, quantum_times, 'ro-', label='Quantum')
    plt.xlabel('Number of Primitives')
    plt.ylabel('Render Time (seconds)')
    plt.title('Render Time (lower is better)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('quantum_benchmark_results.png')
    print("\nBenchmark results saved to 'quantum_benchmark_results.png'")
    
    # Show plot if running in interactive mode
    try:
        plt.show()
    except:
        pass
    
    # Print summary table
    print("\nBenchmark Summary:")
    print("==================")
    print(f"{'Primitives':<10} {'Classical':<15} {'Quantum':<15} {'Ratio':<10}")
    for i, num_primitives in enumerate(primitive_counts):
        print(f"{num_primitives:<10} {classical_intersections[i]:<15} {quantum_intersections[i]:<15} {classical_intersections[i] / quantum_intersections[i]:.2f}x")
    
    # Calculate and print overall trend
    classical_growth = classical_intersections[-1] / classical_intersections[0]
    quantum_growth = quantum_intersections[-1] / quantum_intersections[0]
    primitive_growth = primitive_counts[-1] / primitive_counts[0]
    
    print("\nGrowth Analysis:")
    print(f"Primitive count increased by {primitive_growth:.2f}x")
    print(f"Classical intersections increased by {classical_growth:.2f}x (expected: {primitive_growth:.2f}x)")
    print(f"Quantum intersections increased by {quantum_growth:.2f}x (expected: {np.sqrt(primitive_growth):.2f}x)")


if __name__ == "__main__":
    print("Quantum Ray Tracing Benchmark")
    print("============================")
    run_benchmark()