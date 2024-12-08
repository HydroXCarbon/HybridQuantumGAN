from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer import AerSimulator

def create_qnn(device):
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    sim = AerSimulator(method='statevector', device='GPU')

    transpiled_qc = transpile(qc, sim)

    qnn = EstimatorQNN(
        circuit=transpiled_qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    
    return qnn