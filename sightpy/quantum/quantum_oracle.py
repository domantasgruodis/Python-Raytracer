import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
import logging

from ..utils.constants import FARAWAY

logger = logging.getLogger(__name__)

class QuantumOracle:

    def __init__(self, ray_origin, ray_dir, primitives, min_depth=float('inf')):
        self.ray_origin = ray_origin
        self.ray_dir = ray_dir
        self.primitives = primitives
        self.min_depth = min_depth
        self.num_primitives = len(primitives)
        self.num_index_qubits = int(np.ceil(np.log2(max(2, self.num_primitives))))
        
        logger.debug(f"Creating oracle for {self.num_primitives} primitives using {self.num_index_qubits} qubits")

    def build_oracle_circuit(self):
        self.num_index_qubits = max(1, int(np.ceil(np.log2(max(2, self.num_primitives)))))
        logger.debug(f"Creating oracle with {self.num_index_qubits} index qubits for {self.num_primitives} primitives")
        
        index_reg = QuantumRegister(self.num_index_qubits, 'idx')
        result_reg = QuantumRegister(1, 'res')
        
        num_ancilla = max(8, self.num_index_qubits * 2)
        ancilla_reg = QuantumRegister(num_ancilla, 'anc')
        
        circuit = QuantumCircuit(index_reg, result_reg, ancilla_reg)
        
        max_encodable_primitives = min(self.num_primitives, 2**self.num_index_qubits)
        logger.debug(f"Processing {max_encodable_primitives} primitives")
        
        for i in range(max_encodable_primitives):
            bin_idx = format(i, f'0{self.num_index_qubits}b')
            logger.debug(f"Processing primitive {i}, binary: {bin_idx}")
            
            controls = []
            try:
                for j, bit in enumerate(bin_idx):
                    if j < self.num_index_qubits:
                        if bit == '0':
                            circuit.x(index_reg[j])
                            controls.append(index_reg[j])
                        else:
                            controls.append(index_reg[j])
                
                if i < self.num_primitives:
                    primitive = self.primitives[i]
                    self._implement_intersection_for_primitive(circuit, i, primitive, controls, result_reg, ancilla_reg)
                
                for j, bit in enumerate(bin_idx):
                    if j < self.num_index_qubits and bit == '0':
                        circuit.x(index_reg[j])
            except Exception as e:
                logger.warning(f"Error processing primitive {i}: {e}")
                for j, bit in enumerate(bin_idx):
                    if j < self.num_index_qubits and bit == '0':
                        try:
                            circuit.x(index_reg[j])
                        except:
                            pass
                continue
        
        return circuit

    def _implement_intersection_for_primitive(self, circuit, idx, primitive, controls, result_reg, ancilla_reg):
        collider = primitive.collider_list[0]
        distance, orientation = collider.intersect(self.ray_origin, self.ray_dir)
        
        has_intersection = False
        
        if hasattr(distance, '__len__') and len(distance) > 1:
            has_intersection = np.all(distance < FARAWAY)
        else:
            has_intersection = distance < FARAWAY
        
        within_min_depth = False
        if has_intersection:
            if hasattr(distance, '__len__') and len(distance) > 1:
                within_min_depth = np.all(distance < self.min_depth)
            else:
                within_min_depth = distance < self.min_depth
        
        if within_min_depth:
            logger.debug(f"Primitive {idx} intersects at distance {distance}, marking in circuit")
            
            if len(controls) > 0:
                try:
                    if len(controls) == 1:
                        circuit.cx(controls[0], result_reg[0])
                    elif len(controls) == 2:
                        circuit.ccx(controls[0], controls[1], result_reg[0])
                    else:
                        available_ancilla = min(len(controls) - 2, len(ancilla_reg))
                        if available_ancilla >= len(controls) - 2:
                            ancilla_to_use = list(ancilla_reg)[:available_ancilla]
                            all_qubits = controls + [result_reg[0]] + ancilla_to_use
                            mcx_gate = MCXGate(num_ctrl_qubits=len(controls))
                            circuit.append(mcx_gate, all_qubits)
                        else:
                            self._apply_mcx_decomposition(circuit, controls, result_reg[0], list(ancilla_reg))
                except Exception as e:
                    logger.warning(f"Failed to apply MCX gate for primitive {idx}: {e}")
                    try:
                        circuit.x(result_reg[0])
                    except Exception as e2:
                        logger.warning(f"Fallback also failed: {e2}")
            else:
                circuit.x(result_reg[0])

    def _apply_mcx_decomposition(self, circuit, controls, target, ancilla_qubits):
        if len(controls) <= 2:
            if len(controls) == 2:
                circuit.ccx(controls[0], controls[1], target)
            elif len(controls) == 1:
                circuit.cx(controls[0], target)
            else:
                circuit.x(target)
            return

        if len(ancilla_qubits) > 0:
            ancilla = ancilla_qubits[0]
            remaining_ancilla = ancilla_qubits[1:]
            
            m = len(controls) // 2
            controls_first_half = controls[:m]
            controls_second_half = controls[m:]
            
            self._apply_mcx_decomposition(circuit, controls_first_half, ancilla, remaining_ancilla)
            self._apply_mcx_decomposition(circuit, controls_second_half + [ancilla], target, remaining_ancilla)
            
            self._apply_mcx_decomposition(circuit, controls_first_half, ancilla, remaining_ancilla)
        else:
            for i in range(len(controls) - 2):
                circuit.cx(controls[i], controls[i+1])

            circuit.ccx(controls[-2], controls[-1], target)
            
            for i in range(len(controls) - 3, -1, -1):
                circuit.cx(controls[i], controls[i+1])


def build_intersection_oracle(ray_origin, ray_dir, primitives, min_depth=float('inf')):
    oracle = QuantumOracle(ray_origin, ray_dir, primitives, min_depth)
    return oracle.build_oracle_circuit()


def build_grover_operator(oracle_circuit):
    circuit = QuantumCircuit(*oracle_circuit.qregs)
    num_index_qubits = oracle_circuit.qregs[0].size
    
    for i in range(num_index_qubits):
        circuit.h(i)
 
    circuit.x(num_index_qubits)
    circuit.h(num_index_qubits)
    
    circuit = circuit.compose(oracle_circuit)
    
    for i in range(num_index_qubits):
        circuit.h(i)
    
    for i in range(num_index_qubits):
        circuit.x(i)
    
    if num_index_qubits > 0:
        if num_index_qubits == 1:
            circuit.z(0)
        elif num_index_qubits == 2:
            circuit.h(1)
            circuit.cx(0, 1)
            circuit.h(1)
        else:
            circuit.h(num_index_qubits - 1)
            
            controls = list(range(num_index_qubits - 1))
            target = num_index_qubits - 1
            
            ancilla_qubits = list(oracle_circuit.qregs[2])
            
            try:
                if len(controls) <= 2:
                    if len(controls) == 2:
                        circuit.ccx(controls[0], controls[1], target)
                    elif len(controls) == 1:
                        circuit.cx(controls[0], target)
                else:
                    if len(ancilla_qubits) >= len(controls) - 2:
                        ancilla_to_use = ancilla_qubits[:len(controls)-2]
                        all_qubits = controls + [target] + ancilla_to_use
                        mcx_gate = MCXGate(num_ctrl_qubits=len(controls))
                        circuit.append(mcx_gate, all_qubits)
                    else:
                        for i in range(len(controls) - 1):
                            circuit.cx(controls[i], controls[i+1])
                        circuit.cx(controls[-1], target)
                        
                        for i in range(len(controls) - 2, -1, -1):
                            circuit.cx(controls[i], controls[i+1])
            except Exception as e:
                logger.warning(f"Failed to apply MCZ in diffusion: {e}")
                for i in range(num_index_qubits - 1):
                    circuit.cx(i, num_index_qubits - 1)
                circuit.h(num_index_qubits - 1)
                circuit.x(num_index_qubits - 1)
                circuit.h(num_index_qubits - 1)
                for i in range(num_index_qubits - 1):
                    circuit.cx(i, num_index_qubits - 1)
            
            circuit.h(num_index_qubits - 1)
    
    for i in range(num_index_qubits):
        circuit.x(i)
    
    for i in range(num_index_qubits):
        circuit.h(i)
    
    return circuit