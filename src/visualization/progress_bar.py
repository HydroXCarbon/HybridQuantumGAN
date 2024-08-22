from tqdm import tqdm

class ProgressBar(tqdm):
    def __init__(self, total_batches, epoch, num_epochs):
        super().__init__(total=total_batches, 
                         desc=f"Training [{100 * epoch / num_epochs:.1f}%] Epoch {epoch}", 
                         unit="batch", 
                         bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}')