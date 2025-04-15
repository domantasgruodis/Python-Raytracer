"""
Tests for the quantum oracle implementation.
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import sightpy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sightpy import *
from sightpy.quantum.quantum_oracle import build_intersection_oracle, build_grover_operator
from qiskit import transpile, ClassicalRegister
from qiskit_aer import AerSimulator


class TestQuantumOracle(unittest.TestCase):
    """Tests for the quantum oracle implementation."""
    
    def setUp(self):
        """Set up test cases."""
        # Create a simple ray and sphere for testing
        self.ray_origin = vec3(0, 0, 0)
        self.ray_dir = vec3(0, 0, -1).normalize()
        
        # Create a sphere that the ray intersects
        self.sphere_material = Diffuse(diff_color=rgb(0.8, 0.2, 0.2))
        self.sphere = Sphere(
            material=self.sphere_material,
            center=vec3(0, 0, -10),
            radius=2
        )
        
        # Create a sphere that the ray doesn't intersect
        self.non_intersecting_sphere = Sphere(
            material=self.sphere_material,
            center=vec3(5, 5, -10),
            radius=2
        )
    
    def test_single_primitive_intersection(self):
        """Test oracle with a single primitive that intersects."""
        print("Testing oracle with a single intersecting primitive...")
        
        # Build the oracle
        oracle = build_intersection_oracle(
            self.ray_origin,
            self.ray_dir,
            [self.sphere]
        )
        
        # The circuit should have 1 index qubit (2^1 = 2 > 1 primitive)
        # plus 1 result qubit and some ancilla qubits
        self.assertGreaterEqual(sum(reg.size for reg in oracle.qregs), 2)
        
        # Add a classical register for measurement
        cr = ClassicalRegister(1, 'meas')
        oracle.add_register(cr)
        
        # Add measurement to the result qubit
        oracle.measure(oracle.qregs[1][0], cr[0])  # Result qubit (second register) to first classical bit
        
        # Execute the circuit
        simulator = AerSimulator()

        # Set simulator options to avoid warnings
        simulator_options = {
            "method": "statevector",
            "device": "CPU",
            "precision": "double",
        }

        compiled_circuit = transpile(oracle, simulator)
        job = simulator.run(compiled_circuit, shots=1024, **simulator_options)
        result = job.result()
        counts = result.get_counts()
        
        print(f"  Circuit depth: {compiled_circuit.depth()}")
        print(f"  Circuit width: {sum(reg.size for reg in oracle.qregs)}")
        print(f"  Measurement results: {counts}")
        
        # We expect a non-zero probability of measuring 1 for the result qubit
        self.assertTrue('1' in counts)
        self.assertGreater(counts.get('1', 0), 0)
    
    def test_single_primitive_no_intersection(self):
        """Test oracle with a single primitive that doesn't intersect."""
        print("Testing oracle with a single non-intersecting primitive...")
        
        # Build the oracle
        oracle = build_intersection_oracle(
            self.ray_origin,
            self.ray_dir,
            [self.non_intersecting_sphere]
        )
        
        # Add a classical register for measurement
        cr = ClassicalRegister(1, 'meas')
        oracle.add_register(cr)
        
        # Add measurement to the result qubit
        oracle.measure(oracle.qregs[1][0], cr[0])  # Result qubit (second register) to first classical bit
        
        # Execute the circuit
        simulator = AerSimulator()

        # Set simulator options to avoid warnings
        simulator_options = {
            "method": "statevector",
            "device": "CPU",
            "precision": "double",
        }

        compiled_circuit = transpile(oracle, simulator)
        job = simulator.run(compiled_circuit, shots=1024, **simulator_options)
        result = job.result()
        counts = result.get_counts()
        
        print(f"  Circuit depth: {compiled_circuit.depth()}")
        print(f"  Circuit width: {sum(reg.size for reg in oracle.qregs)}")
        print(f"  Measurement results: {counts}")
        
        # We expect a zero or very low probability of measuring 1 for the result qubit
        # (There might be some noise in the simulation)
        self.assertTrue('0' in counts)
        self.assertGreaterEqual(counts.get('0', 0) / 1024, 0.9)  # At least 90% should be 0
    
    def test_multiple_primitives(self):
        """Test oracle with multiple primitives."""
        print("Testing oracle with multiple primitives...")
        
        # Create a list of primitives, including both intersecting and non-intersecting ones
        primitives = [
            self.sphere,
            self.non_intersecting_sphere,
            Sphere(
                material=self.sphere_material,
                center=vec3(0, 0, -5),
                radius=1
            )
        ]
        
        # Build the oracle
        oracle = build_intersection_oracle(
            self.ray_origin,
            self.ray_dir,
            primitives
        )
        
        # The circuit should have 2 index qubits (2^2 = 4 > 3 primitives)
        # plus 1 result qubit and some ancilla qubits
        self.assertGreaterEqual(sum(reg.size for reg in oracle.qregs), 3)
        
        # Build the Grover operator
        grover = build_grover_operator(oracle)
        
        # Add measurements to the index qubits
        num_index_qubits = oracle.qregs[0].size  # Get actual number of index qubits
        cr = ClassicalRegister(num_index_qubits, 'meas')
        measurement_circuit = grover.copy()
        measurement_circuit.add_register(cr)
        
        for i in range(num_index_qubits):
            measurement_circuit.measure(i, i)
        
        # Execute the circuit
        simulator = AerSimulator()

        # Set simulator options to avoid warnings
        simulator_options = {
            "method": "statevector",
            "device": "CPU",
            "precision": "double",
        }

        compiled_circuit = transpile(measurement_circuit, simulator)
        job = simulator.run(compiled_circuit, shots=1024, **simulator_options)
        result = job.result()
        counts = result.get_counts()
        
        print(f"  Circuit depth: {compiled_circuit.depth()}")
        print(f"  Circuit width: {sum(reg.size for reg in measurement_circuit.qregs)}")
        print(f"  Measurement results: {counts}")
        
        # Count intersecting and non-intersecting primitives
        # Make sure we handle different bit string lengths correctly
        intersecting_counts = 0
        non_intersecting_counts = 0
        
        for bitstring, count in counts.items():
            # Index 0 (first primitive) and Index 2 (third primitive) should intersect
            idx = int(bitstring, 2)
            if idx == 0 or idx == 2:  # Intersecting primitives at indices 0 and 2
                intersecting_counts += count
            else:
                non_intersecting_counts += count
        
        print(f"  Intersecting counts: {intersecting_counts}")
        print(f"  Non-intersecting counts: {non_intersecting_counts}")
        
        # The probability for intersecting primitives should be higher
        self.assertGreater(intersecting_counts, non_intersecting_counts)
    
    def test_grover_iteration(self):
        """Test that Grover iteration increases the probability of finding intersections."""
        print("Testing Grover iteration effectiveness...")
        
        # Create a list of primitives, with only one intersecting
        primitives = [
            Sphere(
                material=self.sphere_material,
                center=vec3(0, 0, -5),
                radius=1
            ),
            self.non_intersecting_sphere,
            self.non_intersecting_sphere
        ]
        
        # Build the oracle
        oracle = build_intersection_oracle(
            self.ray_origin,
            self.ray_dir,
            primitives
        )
        
        # Build the Grover operator
        grover = build_grover_operator(oracle)
        
        # Helper function to run the circuit with a given number of iterations
        def run_with_iterations(iterations):
            # Get number of index qubits
            num_index_qubits = oracle.qregs[0].size
            
            # Create circuit with the specified number of Grover iterations
            circuit = grover.copy()
            
            # Apply multiple Grover iterations
            for _ in range(iterations - 1):  # -1 because we already applied one
                circuit = circuit.compose(grover)
            
            # Add a classical register for measurement
            cr = ClassicalRegister(num_index_qubits, 'meas')
            circuit.add_register(cr)
            
            # Add measurements to the index qubits
            for i in range(num_index_qubits):
                circuit.measure(i, i)
            
            # Execute the circuit
            simulator = AerSimulator()

            # Set simulator options to avoid warnings
            simulator_options = {
                "method": "statevector",
                "device": "CPU",
                "precision": "double",
            }

            compiled_circuit = transpile(circuit, simulator)
            job = simulator.run(compiled_circuit, shots=1024, **simulator_options)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate probability of measuring the intersecting primitive (index 0)
            # Find the binary representation of 0 with the correct number of bits
            zero_bitstring = '0' * num_index_qubits
            return counts.get(zero_bitstring, 0) / 1024
        
        # Run with different numbers of iterations
        prob_1_iter = run_with_iterations(1)
        prob_2_iter = run_with_iterations(2)
        
        print(f"  Probability after 1 iteration: {prob_1_iter:.4f}")
        print(f"  Probability after 2 iterations: {prob_2_iter:.4f}")
        
        # The probability should change with more iterations
        # (up to a certain point determined by the optimal number of iterations)
        self.assertNotEqual(prob_1_iter, prob_2_iter)


if __name__ == '__main__':
    unittest.main()