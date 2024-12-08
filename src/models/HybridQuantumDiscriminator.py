from torch import nn

from qiskit_machine_learning.connectors import TorchConnector
from .initializer.hqnn_initializer import create_qnn

class HybridQuantumDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.device = 'cuda:0'  # Default to GPU first
        
    self.qnn = create_qnn(device=self.device)

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
      TorchConnector(self.qnn),
      nn.Linear(1, 1),
      nn.Sigmoid(), 
    )

  def forward(self, x):
    x = self.model(x)
    return x
  
  #def to(self, device):
    #self.device = device
    #self.model.to(device)
    #print(f'testsetestsetestsetsesetseestset: {device}')
    #self.qnn = create_qnn(device=device)

    #self.model[12] = TorchConnector(self.qnn) 
    
    #return self