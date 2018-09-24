class TrainOutputWriter(object):
    def __init__(self, lossCollector):
        self.lossCollector = lossCollector
        # TODO accuracy on validation set

    def print_epoch_info(self, epoch, _):
        """
        Print train and validation loss after epoch. 
        Call this function after 'lossCollector.store_train_loss'.
        """
        if epoch == 0:
            initial_val_loss = self.lossCollector.val_losses[0]
            print('initial val-loss:', f'{initial_val_loss:0.3}')
            print('epoch', 'train-loss', 'val-loss')
        train_loss = self.lossCollector.train_losses[-1]
        val_loss = self.lossCollector.val_losses[-1]
        print(epoch, f'{train_loss:0.3}', f'{val_loss:0.3}')

