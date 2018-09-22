class TrainOutputWriter(object):
    def __init__(self, lossCollector):
        self.lossCollector = lossCollector
        # TODO accuracy on validation set

    def print_epoch_info(self, epoch, batch_losses):
        """
        Print train and validation loss after epoch. 
        Call this function after 'lossCollector.store_train_loss'.
        """
        if epoch == 0:
            print('epoch', 'train_loss', 'val_loss')
        train_loss = self.lossCollector.train_losses[-1]
        val_loss = self.lossCollector.val_losses[-1]
        print(epoch, f'{train_loss:0.3}', f'{val_loss:0.3}')

