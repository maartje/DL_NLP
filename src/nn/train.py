import torch

def fit(model, train_data, loss_criterion, optimizer, 
        epochs, fn_epoch_listeners = []):
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
            log_probs = model(seq_vectors, lengths)
            loss = loss_criterion(log_probs.permute(0,2,1), targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            print(loss.item())
        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(epoch, batch_losses)
        
# def predict(model, user_idxs, movie_idxs):
#     """Predict language probabilities for texts."""
#     model.eval() # set in predict mode
#     with torch.no_grad():
#         predicted_ratings = model(user_idxs, movie_idxs).squeeze()
#     return predicted_ratings

# def calculate_loss(model, dataset, loss_criterion):
#     """Calculate average loss over dataset without auto grad."""
#     dl = data.DataLoader(dataset, batch_size = len(dataset))
#     full_dataset = next(iter(dl))
#     (user_idxs, movie_idxs, target_ratings) = full_dataset
#     with torch.no_grad():
#         predicted_ratings = predict(model, user_idxs, movie_idxs)
#         loss = loss_criterion(predicted_ratings, target_ratings.float())
#     return loss.item()
