from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_aer  import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp


def create_qnn(device):
    # Define the feature map and ansatz
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    
    # Combine feature map and ansatz into a single circuit
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # Configure the AerSimulator
    sim = AerSimulator(method='statevector', device=device)
    
    # Transpile the circuit for the simulator
    transpiled_qc = transpile(qc, sim)
    
    # Create an EstimatorV2 using the simulator as the backend
    estimator = EstimatorV2.from_backend(sim)

    observable = SparsePauliOp.from_list([("ZZ", 1.0)])  # 'ZZ' with a coefficient of 1.0

    return {
        "feature_map_params": feature_map.parameters,
        "ansatz_params": ansatz.parameters,
        "estimator": estimator,
        "transpiled_circuit": transpiled_qc,
        "observable":observable
    }