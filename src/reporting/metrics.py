def calculate_loss(log_probs, targets, loss_criterion):
    loss = loss_criterion(log_probs.permute(0,2,1), targets)
    return loss.item()

def calculate_accuracies(log_probs, targets, lengths):
    """
    Calculates the average accuracies for each character position.
    Remark: the paddings must be excluded from the calculation of the average accuracy

    With 'N' the batch_size, 'S' the max character sequence length and 'O' the nr of output classes (including 0 for paddings)
    Args:
        log_probs: N x S x O probability of the output classes per input string per character position 
        targets: N x S output class per example per character position (all positions have the same output class or they represent pads given by a zero)
        lengths: length of the input sequence (without the pads)

    Returns:
        average accuracies: S the average accuracies for each character position
    """
    raise NotImplementedError