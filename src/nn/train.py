import torch

def fit(model, train_data, loss_criterion, optimizer, 
        epochs, model_name, device = torch.device('cpu'), 
        fn_epoch_listeners = []):
    """Fit the model on the training data.
    
    Args:
        model: Language Identification Model
        train_data: iterator over batches
        loss_criterion: for example nn.NLLLoss
        optimizer: optimizer
        epochs: number of epochs used to train the model
        fn_epoch_listeners: list of functions that are called after each epoch
    """

    for epoch in range(epochs):
        model.train() # set in train mode
        batch_losses = []
        for batch_index, batch in enumerate(train_data):
            optimizer.zero_grad()
            (seq_vectors, targets, lengths) = batch
            seq_vectors, targets, lengths = seq_vectors.to(device), targets.to(device), lengths.to(device)
            probs = model(seq_vectors, lengths)
            if model_name == 'rnn':
                loss = loss_criterion(probs.permute(0, 2, 1), targets)
            elif model_name == 'cnn':
                loss = loss_criterion(probs, targets[:, -1])

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(epoch, batch_losses)
        

def predict(model, test_data, max_length, model_name='rnn', device=torch.device('cpu')):
    """ Predicts the probabilities of the target classes.
    
    Args:
        model: Language Identification Model
        test_data: iterator over batches of testdata

    Returns:
        log probabilities [BatchSize x Max-SequenceLength x NrOfTargetClasses]
        targets [BatchSize x Max-SequenceLength] (0 is used for padding)
        lengths [BatchSize]
    """
    model.eval() # set in predict mode

    log_probs_batches = [] # BatchSize x Max-SequenceLength x TargetClasses
    targets_batches = []   # BatchSize x Max-SequenceLength (0 is used for padding)
    lengths_batches = []   # BatchSize
    with torch.no_grad():
        for _, batch in enumerate(test_data):
            (seq_vectors, targets, lengths) = batch
            seq_vectors, targets, lengths = seq_vectors.to(device), targets.to(device), lengths.to(device)
            log_probs = model(seq_vectors, lengths)

            if model_name == 'rnn':
                # pad with 0 in seq dimension (= 1)
                log_probs_padded = torch.zeros(log_probs.shape[0], max_length, log_probs.shape[2])
                log_probs_padded[:, :log_probs.shape[1], :] = log_probs.cpu()
                targets_padded = torch.zeros(targets.shape[0], max_length, dtype=torch.long)
                targets_padded[:, :targets.shape[1]] = targets.cpu()

                log_probs_batches.append(log_probs_padded)
                targets_batches.append(targets_padded) 
                lengths_batches.append(lengths) 
            elif model_name == 'cnn':
                log_probs_batches.append(log_probs)
                targets_batches.append(targets[:, -1].cpu())
                lengths_batches.append(torch.ones_like(lengths))

    all_log_probs = torch.cat(log_probs_batches, dim=0)
    all_targets = torch.cat(targets_batches, dim=0)     
    all_lengths = torch.cat(lengths_batches, dim=0)
    return all_log_probs, all_targets, all_lengths

