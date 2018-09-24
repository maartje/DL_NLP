import numpy as np
from src.nn.train import predict
from src.reporting.metrics import *

class LossCollector(object):

    def __init__(self, model, val_data, max_length, loss_criterion):
        self.model = model
        self.val_data = val_data
        self.loss_criterion = loss_criterion
        self.max_length = max_length

        self.train_losses = []
        self.val_losses = [] 
        self.initial_val_loss = self.calculate_metrics()

    def store_metrics(self, epoch, batch_losses):
        self.store_train_loss(epoch, batch_losses)
        val_loss = self.calculate_metrics() # TODO accuracy
        self.val_losses.append(val_loss)

    def store_train_loss(self, epoch, batch_losses):
        """Collect average train loss after epoch has completed."""
        train_loss = np.mean(batch_losses)
        self.train_losses.append(train_loss)

    def calculate_metrics(self):
        (log_probs, targets, _) = predict(self.model, self.val_data, self.max_length)
        val_loss = calculate_loss(log_probs, targets, self.loss_criterion)
        # TODO: return also accuracy: np.mean(calculate_accuracies(log_probs, targets, lengths)) 
        return val_loss

    def store_val_accuracy(self, epoch, batch_losses):
        """Collect validation accuracy after epoch has completed."""
        raise NotImplementedError
