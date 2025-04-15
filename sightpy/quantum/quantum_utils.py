"""
Utility functions for quantum ray tracing.
"""
import warnings
import re

def filter_qiskit_warnings():
    """
    Filter out Qiskit deprecation warnings.
    Call this function at the beginning of any script using quantum ray tracing.
    """
    # Filter all Qiskit deprecation warnings by module
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit_aer.*")
    
    # Filter specific messages that occur frequently
    specific_warnings = [
        # DAGCircuit properties
        "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.duration``",
        "The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.unit``",
        
        # Instruction condition
        "The property ``qiskit.circuit.instruction.Instruction.condition``",
        
        # Other potential future warnings
        "is deprecated as of qiskit"
    ]
    
    for message in specific_warnings:
        warnings.filterwarnings(
            "ignore", 
            message=message, 
            category=DeprecationWarning
        )

def configure_simulator_options():
    """
    Configure standard simulator options to minimize deprecation warnings.
    
    Returns:
        dict: Simulator options that reduce deprecation warnings
    """
    return {
        "method": "statevector",
        "device": "CPU",
        "precision": "double",
        "max_parallel_threads": 0,  # Use all available threads
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "cusparse_path": None,  # Avoid GPU-related warnings
    }
    
def get_simulator():
    """
    Get an AerSimulator instance using the correct import path.
    
    Returns:
        AerSimulator: An instance of the AerSimulator
    """
    try:
        # Try the modern import path first
        from qiskit_aer import AerSimulator
        return AerSimulator()
    except ImportError:
        # Fall back to the older import path
            from qiskit import Aer
            return Aer.get_backend('aer_simulator')
            