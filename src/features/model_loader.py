import torch.optim as optim
import torch.nn as nn
from models import Generator, ClassicalDiscriminator, QuantumDiscriminator, HybridQuantumDiscriminator

def get_model(models, model_selector, device):
  model_list = []
  optimizer_list = []
  model_selector = model_selector.split(',')
  model_selector = tuple(model.strip() for model in model_selector)
  
  # Initialize models dynamically
  for model_name in models:
    if model_name not in model_selector:
      continue
    learning_rate = models[model_name]['learning_rate']
    betas = models[model_name]['betas']
    betas = betas.split(',')
    betas = tuple(float(beta.strip()) for beta in betas)

    # Get the class object dynamically using globals()
    model_class_name = models[model_name]['model_class']
    model_loss_function_name = models[model_name]['loss_function']
    model_optimizer_name = models[model_name]['optimizer']
    model_class = globals()[model_class_name]

    # Set up models dynamically
    model_instance = model_class()
    model_instance = model_instance.to(device)
    model_instance.name = model_name
    
    # Dynamically retrieve the loss function class from torch.nn
    loss_function_class = getattr(nn, model_loss_function_name)
    model_instance.loss_function = loss_function_class()

    model_list.append(model_instance)

    # Dynamically retrieve the optimizer class from torch.optim
    optimizer_class = getattr(optim, model_optimizer_name)
    optimizer_instance = optimizer_class(model_instance.parameters(), lr=learning_rate, betas=betas)
    optimizer_instance.name = 'optimizer_' + model_name
    
    optimizer_list.append(optimizer_instance)

  return model_list, optimizer_list