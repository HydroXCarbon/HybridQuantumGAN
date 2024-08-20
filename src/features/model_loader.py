from torch.optim import Adam

def get_model(models, device):
  model_list = []
  optimizer_list = []

  # Initialize models dynamically
  for model_name in models:
    learning_rate = models[model_name]['learning_rate']

    # Get the class object dynamically using globals
    model_class_name = models[model_name]['model_class']
    model_class = globals()[model_class_name]

    # Set up models dynamically
    model_instance = model_class()
    model_instance = model_instance.to(device)
    model_instance.name = model_name
    model_list.append(model_instance)

    # Set up optimizers dynamically
    optimizer_instance = Adam(model_instance.parameters(), lr=learning_rate)
    optimizer_instance.name = 'optimizer_'+model_name
    optimizer_list.append(optimizer_instance)

  return model_list, optimizer_list