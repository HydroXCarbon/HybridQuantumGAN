from torch import nn
import numpy as np
import torch

from qiskit_machine_learning.connectors import TorchConnector
from .initializer.hqnn_initializer import create_qnn

import torch
from torch.autograd import Function

class QuantumFunction(Function):
  @staticmethod
  def forward(ctx, qc, observable, estimator, x, theta):
    ctx.qc = qc
    ctx.observable = observable
    ctx.estimator = estimator
    ctx.save_for_backward(x, theta)

    quantum_inputs = torch.cat([x, theta], dim=1).detach().cpu().numpy()
    quantum_results = []
    for q_input in quantum_inputs:
      job = estimator.run([(qc, observable, q_input)])
      result = job.result()
      quantum_results.append(result[0].data.evs.item())

    quantum_outputs = torch.tensor(quantum_results, device=x.device)
    return quantum_outputs.unsqueeze(1)

  @staticmethod
  def backward(ctx, grad_output):
    x, theta = ctx.saved_tensors
    grad_x = grad_theta = None

    # Compute gradients for x and theta if needed
    if ctx.needs_input_grad[3]:
      grad_x = grad_output.expand_as(x)
    if ctx.needs_input_grad[4]:
      grad_theta = grad_output.expand_as(theta)

    return None, None, None, grad_x, grad_theta

class HybridQuantumDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = 'cuda'
        
        # Initialize QNN components
        qnn_data = create_qnn(device='GPU')
        self.estimator = qnn_data["estimator"]
        self.qc = qnn_data["transpiled_circuit"]
        self.feature_map_params = qnn_data["feature_map_params"]
        self.ansatz_params = qnn_data["ansatz_params"]
        self.observable = qnn_data["observable"]

        # Define the quantum trainable parameters
        self.theta = nn.Parameter(torch.ones(len(self.ansatz_params), device=self.device))

        self.pre_quantum = nn.Sequential(
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
        )

        self.post_quantum = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through classical layers
        x = self.pre_quantum(x)
        
        # Ensure self.theta is of shape (batch_size, num_theta)
        theta_repeated = self.theta.unsqueeze(0).expand(x.size(0), -1)  # Repeat theta for batch_size

        # Use custom autograd function for quantum circuit execution
        quantum_outputs = QuantumFunction.apply(self.qc, self.observable, self.estimator, x, theta_repeated)

        # Pass through post-quantum layers
        x = self.post_quantum(quantum_outputs)

        return x