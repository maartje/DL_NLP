import torch

class ModelSaver():

    def __init__(self, model, metrics_collector, fpath_model):
        self.model = model
        self.metrics_collector = metrics_collector
        self.fpath_model = fpath_model
        
    def save_best_model(self, _, __):
        max_val_acc = max(self.metrics_collector.val_accuracies) 
        last_val_acc = self.metrics_collector.val_accuracies[-1]
        if last_val_acc == max_val_acc:
            torch.save(self.model, self.fpath_model)
            print (f"best model so far saved")
        else:
            print()
                


