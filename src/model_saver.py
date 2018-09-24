import torch

class ModelSaver():

    def __init__(self, model, metrics_collector, fpath_model):
        self.model = model
        self.metrics_collector = metrics_collector
        self.fpath_model = fpath_model
        
    def save_best_model(self, _, __):
        min_val_loss = min(self.metrics_collector.val_losses) # TODO max accuracy
        last_val_loss = self.metrics_collector.val_losses[-1]
        if last_val_loss == min_val_loss:
            torch.save(self.model, self.fpath_model)
            print (f"Best model so far saved to '{self.fpath_model}'")
                


