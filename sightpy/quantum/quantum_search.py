"""
Quantum searching algorithm implementation based on Grover's algorithm.
This implements the QSearch algorithm described in the paper.
"""
import numpy as np
import math
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

from .quantum_oracle import build_intersection_oracle, build_grover_operator
from .quantum_utils import configure_simulator_options

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class QSearchResult:
    """Result of quantum search."""
    primitive_idx: int  # Index of the found primitive
    found: bool         # Whether a primitive was found
    probability: float  # Probability of the measured result
    all_results: dict   # All measurement results and probabilities
    iterations: int     # Number of iterations used
    circuit_depth: int  # Depth of the circuit


class QSearch:
    """
    Quantum search implementation based on Grover's algorithm.
    This implements the QSearch algorithm described in the paper.
    """
    def __init__(self, ray_origin, ray_dir, primitives, min_depth=float('inf')):
        """
        Initialize quantum search.
        
        Args:
            ray_origin (vec3): Origin of the ray
            ray_dir (vec3): Direction of the ray
            primitives (list): List of primitives to check for intersection
            min_depth (float): Minimum depth threshold for intersections
        """
        self.ray_origin = ray_origin
        self.ray_dir = ray_dir
        self.primitives = primitives
        self.min_depth = min_depth
        self.num_primitives = len(primitives)
        self.num_index_qubits = int(np.ceil(np.log2(max(2, self.num_primitives))))
        
        # Define the constant c for the exponential search (between 1 and 2)
        self.c = 1.2
        
        logger.debug(f"QSearch initialized with {self.num_primitives} primitives")
    
    def search(self, shots=1024) -> QSearchResult:
        """
        Perform quantum search to find a primitive that intersects with the ray.
        
        Args:
            shots (int): Number of shots for the quantum simulation
            
        Returns:
            QSearchResult: Result of the quantum search
        """
        # First, try random sampling (as in the paper's algorithm)
        result = self._try_random_sampling(shots)
        if result.found:
            logger.debug("Found intersection through random sampling")
            return result
        
        # If random sampling fails, perform the adaptive exponential search
        logger.debug("Random sampling did not find intersection, starting adaptive search")
        return self._adaptive_exponential_search(shots)
    
    def _try_random_sampling(self, shots) -> QSearchResult:
        """
        Try to find an intersection through random sampling.
        This corresponds to the first part of the QSearch algorithm.
        
        Args:
            shots (int): Number of shots for the quantum simulation
            
        Returns:
            QSearchResult: Result of the search
        """
        # Create a circuit with uniform superposition
        index_reg = QuantumRegister(self.num_index_qubits, 'idx')
        result_reg = QuantumRegister(1, 'res')
        meas_reg = ClassicalRegister(self.num_index_qubits, 'meas')
        
        circuit = QuantumCircuit(index_reg, result_reg, meas_reg)
        
        # Create uniform superposition
        for i in range(self.num_index_qubits):
            circuit.h(index_reg[i])
        
        # Measure
        for i in range(self.num_index_qubits):
            circuit.measure(index_reg[i], meas_reg[i])
        
        # Execute the circuit using AerSimulator with modern options
        simulator = AerSimulator()
        
        # Get simulator options from utility function
        simulator_options = configure_simulator_options()
        
        # Transpile with optimization level and seed for reproducibility
        compiled_circuit = transpile(
            circuit, 
            simulator, 
            optimization_level=1,
            seed_transpiler=42
        )
        
        # Run the circuit with options
        job = simulator.run(
            compiled_circuit, 
            shots=shots,
            seed_simulator=42,
            **simulator_options
        )
        
        result = job.result()
        counts = result.get_counts()
        
        # Find the most probable result
        max_count = 0
        max_bitstring = None
        
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                max_bitstring = bitstring
        
        # Convert to primitive index
        primitive_idx = int(max_bitstring, 2) if max_bitstring else 0
        
        # Check if this primitive actually intersects (classical verification)
        if primitive_idx < self.num_primitives:
            collider = self.primitives[primitive_idx].collider_list[0]
            distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
            
            # Import here to avoid circular imports
            from ..utils.constants import FARAWAY

            # Handle the case where sitance is an array
            if hasattr(distance, '__len__') and len(distance) > 1:
                # If distance is an array, check if all elements satisfy the condition
                found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
            else:
                # If distance is a scalar value
                found = distance < self.min_depth and distance < FARAWAY
            
            if found:
                probability = max_count / shots
                return QSearchResult(
                    primitive_idx=primitive_idx,
                    found=True,
                    probability=probability,
                    all_results=counts,
                    iterations=0,
                    circuit_depth=compiled_circuit.depth()
                )
        
        # Return not found
        return QSearchResult(
            primitive_idx=-1,
            found=False,
            probability=0.0,
            all_results=counts,
            iterations=0,
            circuit_depth=compiled_circuit.depth()
        )
    
    def _adaptive_exponential_search(self, shots) -> QSearchResult:
        """
        Perform adaptive exponential search.
        This corresponds to the main loop in the QSearch algorithm.
        
        Args:
            shots (int): Number of shots for the quantum simulation
            
        Returns:
            QSearchResult: Result of the search
        """
        # Initialize the search parameters
        l = 0  # Iteration counter
        M_l = 0  # Maximum number of Grover iterations
        sqrt_N = math.sqrt(self.num_primitives)
        found = False
        
        oracle_circuit = build_intersection_oracle(
            self.ray_origin, self.ray_dir, self.primitives, self.min_depth
        )
        
        while not found and M_l < sqrt_N:
            # Increment iteration counter
            l += 1
            
            # Calculate M_l = min(c^l, sqrt(N))
            M_l = min(self.c ** l, sqrt_N)
            
            # Randomly select number of Grover iterations
            r_l = int(np.random.randint(1, math.ceil(M_l) + 1))
            
            logger.debug(f"Iteration {l}: Using {r_l} Grover iterations (M_l = {M_l})")
            
            # Build the circuit with r_l iterations of Grover
            circuit = self._build_grover_circuit(oracle_circuit, r_l)
            
            # Execute the circuit using AerSimulator with modern options
            simulator = AerSimulator()
            
            # Get simulator options from utility function
            simulator_options = configure_simulator_options()
            
            # Transpile with optimization level and seed for reproducibility
            compiled_circuit = transpile(
                circuit, 
                simulator, 
                optimization_level=1,
                seed_transpiler=l*42  # Use different seeds for different iterations
            )
            
            # Run the circuit with options
            job = simulator.run(
                compiled_circuit, 
                shots=shots,
                seed_simulator=l*42,  # Use different seeds for different iterations
                **simulator_options
            )
            
            result = job.result()
            counts = result.get_counts()
            
            # Find the most probable result
            max_count = 0
            max_bitstring = None
            
            for bitstring, count in counts.items():
                if count > max_count:
                    max_count = count
                    max_bitstring = bitstring
            
            # Convert to primitive index
            primitive_idx = int(max_bitstring, 2) if max_bitstring else 0
            
            # Check if this primitive actually intersects (classical verification)
            if primitive_idx < self.num_primitives:
                collider = self.primitives[primitive_idx].collider_list[0]
                distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
                
                # Import here to avoid circular imports
                from ..utils.constants import FARAWAY
                
                 # Handle the case where distance is an array
                if hasattr(distance, '__len__') and len(distance) > 1:
                    # If distance is an array, check if all elements satisfy the condition
                    found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
                else:
                    # If distance is a scalar value
                    found = distance < self.min_depth and distance < FARAWAY
                
                if found:
                    logger.debug(f"Found intersection with primitive {primitive_idx} at distance {distance}")
                    probability = max_count / shots
                    return QSearchResult(
                        primitive_idx=primitive_idx,
                        found=True,
                        probability=probability,
                        all_results=counts,
                        iterations=l,
                        circuit_depth=compiled_circuit.depth()
                    )
            
            logger.debug(f"No intersection found with primitive {primitive_idx}")
        
        # If we reach here, no intersection was found
        return QSearchResult(
            primitive_idx=-1,
            found=False,
            probability=0.0,
            all_results={},
            iterations=l,
            circuit_depth=0
        )
    
    def _build_grover_circuit(self, oracle_circuit, num_iterations):
        """
        Build a circuit with the specified number of Grover iterations.
        
        Args:
            oracle_circuit (QuantumCircuit): The oracle circuit
            num_iterations (int): Number of Grover iterations
            
        Returns:
            QuantumCircuit: The complete Grover circuit
        """
        # Get register sizes from the oracle circuit
        num_index_qubits = oracle_circuit.qregs[0].size
        
        # Create registers
        index_reg = QuantumRegister(num_index_qubits, 'idx')
        result_reg = QuantumRegister(1, 'res')
        ancilla_size = oracle_circuit.qregs[2].size if len(oracle_circuit.qregs) > 2 else 4
        ancilla_reg = QuantumRegister(ancilla_size, 'anc')
        meas_reg = ClassicalRegister(num_index_qubits, 'meas')
        
        # Create circuit
        circuit = QuantumCircuit(index_reg, result_reg, ancilla_reg, meas_reg)
        
        # Initial state preparation: Apply H gates to index qubits
        for i in range(num_index_qubits):
            circuit.h(index_reg[i])
        
        # Apply H gate to result qubit and flip it to |->
        circuit.x(result_reg[0])
        circuit.h(result_reg[0])
        
        # Create the Grover operator
        grover_operator = build_grover_operator(oracle_circuit)
        
        # Apply the Grover operator num_iterations times
        for _ in range(num_iterations):
            circuit = circuit.compose(grover_operator)
        
        # Measure the index qubits
        for i in range(num_index_qubits):
            circuit.measure(index_reg[i], meas_reg[i])
        
        return circuit


# Import the constants from the parent module
from ..utils.constants import FARAWAY, UPWARDS, UPDOWN