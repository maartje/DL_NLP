import numpy as np

class LossCollector(object):

    def __init__(self):
        self.train_losses = []
        self.val_losses = [-1.0] #TODO: calculate each epoch

    def store_train_loss(self, epoch, batch_losses):
        """Collect average train loss after epoch has completed."""
        train_loss = np.mean(batch_losses)
        self.train_losses.append(train_loss)

