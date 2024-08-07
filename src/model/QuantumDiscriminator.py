class Discriminator:
  def __init__(self, n_qubits):
    self.n_qubits = n_qubits
    self.model = self.create_model()