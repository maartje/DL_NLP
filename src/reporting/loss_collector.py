import numpy as np
from src.nn.train import predict

class LossCollector(object):

    def __init__(self, model, val_data, loss_criterion):
        self.model = model
        self.val_data = val_data
        self.loss_criterion = loss_criterion

        self.train_losses = []
        self.val_losses = [] 
        self.initial_val_loss = self.calculate_val_loss()

    def store_metrics(self, epoch, batch_losses):
        self.store_train_loss(epoch, batch_losses)
        self.store_val_loss(epoch, batch_losses)
        #self.store_val_accuracy(self, epoch, batch_losses)

    def store_train_loss(self, epoch, batch_losses):
        """Collect average train loss after epoch has completed."""
        train_loss = np.mean(batch_losses)
        self.train_losses.append(train_loss)

    def store_val_loss(self, epoch, batch_losses):
        """Collect validation loss after epoch has completed."""
        self.val_losses.append(self.calculate_val_loss())

    def calculate_val_loss(self):
        (log_probs, targets, _) = predict(self.model, self.val_data)
        val_loss = self.loss_criterion(log_probs.permute(0,2,1), targets)
        return val_loss.item()

    def store_val_accuracy(self, epoch, batch_losses):
        """Collect validation accuracy after epoch has completed."""
        raise NotImplementedError
