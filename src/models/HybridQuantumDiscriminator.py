from torch import nn

from qiskit_machine_learning.connectors import TorchConnector
from .initializer.hqnn_initializer import create_qnn

class HybridQuantumDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()
    qnn = create_qnn()
    self.model = nn.Sequential(
      nn.Conv2d(1, 2, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(2, 16, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout2d(),
      nn.Flatten(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
      TorchConnector(qnn),
      nn.Linear(1, 1),
      nn.Sigmoid(), 
    )

  def forward(self, x):
    x = self.model(x)
    return x