import numpy as np
import math
import logging
from dataclasses import dataclass

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile
from qiskit_aer import AerSimulator

from .quantum_oracle import build_intersection_oracle, build_grover_operator
from .quantum_utils import configure_simulator_options

logger = logging.getLogger(__name__)

@dataclass
class QSearchResult:
    primitive_idx: int 
    found: bool 
    probability: float
    all_results: dict
    iterations: int
    circuit_depth: int

class QSearch:
    def __init__(self, ray_origin, ray_dir, primitives, min_depth=float('inf')):
        self.ray_origin = ray_origin
        self.ray_dir = ray_dir
        self.primitives = primitives
        self.min_depth = min_depth
        self.num_primitives = len(primitives)
        self.num_index_qubits = int(np.ceil(np.log2(max(2, self.num_primitives))))
        
        self.c = 1.6
        
        logger.debug(f"QSearch initialized with {self.num_primitives} primitives")
    
    def search(self, shots=1024) -> QSearchResult:
        result = self._try_enhanced_random_sampling(shots)
        if result.found:
            logger.debug("Found intersection through enhanced random sampling")
            return result
        
        logger.debug("Enhanced random sampling did not find intersection, starting adaptive search")
        return self._adaptive_exponential_search(shots)
    
    def _try_enhanced_random_sampling(self, shots) -> QSearchResult:
        index_reg = QuantumRegister(self.num_index_qubits, 'idx')
        result_reg = QuantumRegister(1, 'res')
        meas_reg = ClassicalRegister(self.num_index_qubits, 'meas')
        
        circuit = QuantumCircuit(index_reg, result_reg, meas_reg)
        
        for i in range(self.num_index_qubits):
            circuit.h(index_reg[i])
        
        for i in range(self.num_index_qubits):
            circuit.measure(index_reg[i], meas_reg[i])
        
        simulator = AerSimulator()
        
        simulator_options = configure_simulator_options()
        
        compiled_circuit = transpile(
            circuit, 
            simulator, 
            optimization_level=1,
            seed_transpiler=42
        )
        
        job = simulator.run(
            compiled_circuit, 
            shots=shots,
            seed_simulator=42,
            **simulator_options
        )
        
        result = job.result()
        counts = result.get_counts()
        
        sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bitstring, count in sorted_results:
            primitive_idx = int(bitstring, 2)
            if primitive_idx >= self.num_primitives:
                continue

            collider = self.primitives[primitive_idx].collider_list[0]
            distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
            
            from ..utils.constants import FARAWAY
            
            if hasattr(distance, '__len__') and len(distance) > 1:
                found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
            else:
                found = distance < self.min_depth and distance < FARAWAY
            
            if found:
                probability = count / shots
                return QSearchResult(
                    primitive_idx=primitive_idx,
                    found=True,
                    probability=probability,
                    all_results=counts,
                    iterations=0,
                    circuit_depth=compiled_circuit.depth()
                )
        
        special_primitives = self._sample_specific_primitives(shots // 2)
        if special_primitives.found:
            return special_primitives
        
        return QSearchResult(
            primitive_idx=-1,
            found=False,
            probability=0.0,
            all_results=counts,
            iterations=0,
            circuit_depth=compiled_circuit.depth()
        )
    
    def _sample_specific_primitives(self, shots) -> QSearchResult:
        shots_per_primitive = max(1, shots // self.num_primitives)
        remaining_shots = shots
        
        for idx in range(self.num_primitives):
            if remaining_shots <= 0:
                break

            curr_shots = min(shots_per_primitive, remaining_shots)
            remaining_shots -= curr_shots
            
            circuit = QuantumCircuit(self.num_index_qubits, 1)

            bin_idx = format(idx, f'0{self.num_index_qubits}b')
            for j, bit in enumerate(bin_idx):
                if bit == '0':
                    pass
                else:
                    circuit.x(j)
            
            cr = ClassicalRegister(self.num_index_qubits, 'meas')
            circuit.add_register(cr)
            for j in range(self.num_index_qubits):
                circuit.measure(j, j)
            
            simulator = AerSimulator()
            simulator_options = configure_simulator_options()
            compiled_circuit = transpile(circuit, simulator, optimization_level=1)
            
            job = simulator.run(
                compiled_circuit,
                shots=curr_shots,
                **simulator_options
            )
            
            result = job.result()
            counts = result.get_counts()
            
            idx_bitstring = format(idx, f'0{self.num_index_qubits}b')
            
            if idx_bitstring in counts and counts[idx_bitstring] > 0:
                collider = self.primitives[idx].collider_list[0]
                distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
                
                from ..utils.constants import FARAWAY
                
                if hasattr(distance, '__len__') and len(distance) > 1:
                    found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
                else:
                    found = distance < self.min_depth and distance < FARAWAY
                
                if found:
                    return QSearchResult(
                        primitive_idx=idx,
                        found=True,
                        probability=1.0,
                        all_results={idx_bitstring: curr_shots},
                        iterations=0,
                        circuit_depth=compiled_circuit.depth()
                    )
        
        return QSearchResult(
            primitive_idx=-1,
            found=False,
            probability=0.0,
            all_results={},
            iterations=0,
            circuit_depth=0
        )
    
    def _adaptive_exponential_search(self, shots) -> QSearchResult:
        l = 0
        M_l = 0
        sqrt_N = math.sqrt(self.num_primitives)
        found = False
        
        oracle_circuit = build_intersection_oracle(
            self.ray_origin,
            self.ray_dir,
            self.primitives,
            self.min_depth
        )
        
        all_measured_primitives = set()
        best_result = None
        
        min_iterations = 2
        
        while (not found and M_l < sqrt_N) or l < min_iterations:
            l += 1
            
            M_l = min(self.c ** l, sqrt_N)
            
            r_l = int(np.random.randint(1, math.ceil(M_l) + 1))
            
            logger.debug(f"Iteration {l}: Using {r_l} Grover iterations (M_l = {M_l})")
            
            circuit = self._build_grover_circuit(oracle_circuit, r_l)
            
            simulator = AerSimulator()
            
            simulator_options = configure_simulator_options()
            
            compiled_circuit = transpile(
                circuit, 
                simulator, 
                optimization_level=1,
                seed_transpiler=l*42
            )
            
            job = simulator.run(
                compiled_circuit, 
                shots=shots,
                seed_simulator=l*42,
                **simulator_options
            )
            
            result = job.result()
            counts = result.get_counts()
            
            sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            iteration_found = False
            for bitstring, count in sorted_results:
                primitive_idx = int(bitstring, 2)
                
                if primitive_idx >= self.num_primitives:
                    continue
                
                all_measured_primitives.add(primitive_idx)
                
                collider = self.primitives[primitive_idx].collider_list[0]
                distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
                
                from ..utils.constants import FARAWAY
                
                if hasattr(distance, '__len__') and len(distance) > 1:
                    curr_found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
                else:
                    curr_found = distance < self.min_depth and distance < FARAWAY
                
                if curr_found:
                    logger.debug(f"Found intersection with primitive {primitive_idx} at distance {distance}")
                    probability = count / shots
                    
                    curr_result = QSearchResult(
                        primitive_idx=primitive_idx,
                        found=True,
                        probability=probability,
                        all_results=counts,
                        iterations=l,
                        circuit_depth=compiled_circuit.depth()
                    )
                    
                    if best_result is None or curr_result.probability > best_result.probability:
                        best_result = curr_result
                    
                    found = True
                    iteration_found = True
                    break

            if not iteration_found and len(all_measured_primitives) >= min(self.num_primitives, 2**self.num_index_qubits):
                logger.debug(f"All {len(all_measured_primitives)} primitives have been measured without finding an intersection")
                break
        
        if found and best_result is not None:
            return best_result
        
        for idx in range(self.num_primitives):
            collider = self.primitives[idx].collider_list[0]
            distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
            
            from ..utils.constants import FARAWAY
            
            if hasattr(distance, '__len__') and len(distance) > 1:
                curr_found = np.all(distance < self.min_depth) and np.all(distance < FARAWAY)
            else:
                curr_found = distance < self.min_depth and distance < FARAWAY
            
            if curr_found:
                logger.debug(f"Found intersection with primitive {idx} through direct verification")
                return QSearchResult(
                    primitive_idx=idx,
                    found=True,
                    probability=1.0,
                    all_results={},
                    iterations=l,
                    circuit_depth=0
                )
        
        return QSearchResult(
            primitive_idx=-1,
            found=False,
            probability=0.0,
            all_results={},
            iterations=l,
            circuit_depth=0
        )
    
    def _build_grover_circuit(self, oracle_circuit, num_iterations):
        num_index_qubits = oracle_circuit.qregs[0].size
        
        index_reg = QuantumRegister(num_index_qubits, 'idx')
        result_reg = QuantumRegister(1, 'res')
        ancilla_size = oracle_circuit.qregs[2].size if len(oracle_circuit.qregs) > 2 else 4
        ancilla_reg = QuantumRegister(ancilla_size, 'anc')
        meas_reg = ClassicalRegister(num_index_qubits, 'meas')
        
        circuit = QuantumCircuit(index_reg, result_reg, ancilla_reg, meas_reg)
        
        for i in range(num_index_qubits):
            circuit.h(index_reg[i])
        
        circuit.x(result_reg[0])
        circuit.h(result_reg[0])
        
        grover_operator = build_grover_operator(oracle_circuit)
        
        for _ in range(num_iterations):
            circuit = circuit.compose(grover_operator)
        
        for i in range(num_index_qubits):
            circuit.measure(index_reg[i], meas_reg[i])
        
        return circuit

from ..utils.constants import FARAWAY, UPWARDS, UPDOWN