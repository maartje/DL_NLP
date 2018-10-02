import numpy as np
from src.nn.train import predict
from src.reporting.metrics import *

class MetricsCollector(object):

    def __init__(self, model, val_data, max_length, loss_criterion):
        self.model = model
        self.val_data = val_data
        self.loss_criterion = loss_criterion
        self.max_length = max_length

        self.train_losses = []
        self.val_losses = [] 
        self.val_accuracies = [] 
        self.val_confusion_matrices = []

        # store initial metrics on validation set
        initial_val_loss, initial_acc, initial_confusion_matrix = self.calculate_metrics()
        self.val_losses.append(initial_val_loss)
        self.val_accuracies.append(initial_acc)
        self.val_confusion_matrices.append(initial_confusion_matrix)

    def store_metrics(self, _, batch_losses):
        self.store_train_loss(batch_losses)
        val_loss, accuracy, initial_confusion_matrix = self.calculate_metrics()
        self.val_losses.append(val_loss)
        self.val_accuracies.append(accuracy)
        self.val_confusion_matrices.append(initial_confusion_matrix)

    def store_train_loss(self, batch_losses):
        """Collect average train loss after epoch has completed."""
        train_loss = np.mean(batch_losses)
        self.train_losses.append(train_loss)

    def calculate_metrics(self):
        (log_probs, targets, _) = predict(self.model, self.val_data, self.max_length)
        val_loss = calculate_loss(log_probs, targets, self.loss_criterion)
        accuracy =  calculate_accuracy(log_probs.numpy(), targets.numpy())
        confusion_matrix = calculate_confusion_matrix(log_probs.numpy(), targets.numpy())
        return val_loss, accuracy, confusion_matrix
