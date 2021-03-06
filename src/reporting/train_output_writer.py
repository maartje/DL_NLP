import datetime

class TrainOutputWriter(object):
    def __init__(self, metricCollector):
        self.metricCollector = metricCollector

    def print_epoch_info(self, epoch, _):
        """
        Print train and validation loss after epoch. 
        Call this function after 'metricCollector.store_train_loss'.
        """
        if epoch == 0:
            initial_val_loss = self.metricCollector.val_losses[0]
            initial_val_acc = self.metricCollector.val_accuracies[0]
            print('initial val-loss:', f'{initial_val_loss:0.3}',
                  '\t\t initial val-accuracy:', f'{initial_val_acc:0.3}')
            print('epoch \t train loss \t validation loss \t accuracy \t time')
        train_loss = self.metricCollector.train_losses[-1]
        val_loss = self.metricCollector.val_losses[-1]
        accuracy = self.metricCollector.val_accuracies[-1]
        print(epoch, 
              '\t', f'{train_loss:0.3}', 
              '\t\t', f'{val_loss:0.3}', 
              '\t\t\t', f'{accuracy:0.3}',
              '\t', datetime.datetime.now().time().isoformat(timespec='seconds'),
              end='\t')
