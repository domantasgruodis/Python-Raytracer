def configure_simulator_options():
   
    return {
        "method": "statevector",
        "device": "CPU",
        "precision": "double",
        "max_parallel_threads": 0,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "cusparse_path": None,
    }
    
def get_simulator():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator()
    except ImportError:
            from qiskit import Aer
            return Aer.get_backend('aer_simulator')
            