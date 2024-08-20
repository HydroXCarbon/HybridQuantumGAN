import torch
from torch import nn
from qiskit_machine_learning.connectors import TorchConnector
from .qnn_initializer import create_qnn
class QuantumDiscriminator(nn.Module):
    def __init__(self):
        super(QuantumDiscriminator, self).__init__()
        qnn = create_qnn()
        self.qnn = TorchConnector(qnn)

    def forward(self, x):
        x = self.qnn(x)
        return torch.cat((x, 1 - x), -1)