import numpy.ma as ma

def calculate_loss(log_probs, targets, loss_criterion):
    loss = loss_criterion(log_probs.permute(0,2,1), targets)
    return loss.item()

def calculate_accuracy(probs, targets, t_axis=None):
    """
    Calculates the average accuracy 
    WARNING: this implementation does not take padding into account! 

    With 'N' the batch_size, 'S' the max character sequence length and 'O' the nr of output classes
    Args:
        log_probs: N x S x O probability of the output classes  
        targets: N x S target class (all positions in the sequence have in fact the same output class)
        t_axis: 
            None => average over all instances and all char positions (returns an int)
            0    => average over all instances (returns a list representing average accuracy over batch instances  for each seq position)
            1    => average over all sequence positions (returns a list representing average accuracy over sequence positions for each instance in a batch)

    Returns:
        average accuracie(s)
    """

    predictions = probs.argmax(axis=2) 
    corrects = (predictions == targets).sum(axis=t_axis)
    totals = ma.count(targets, axis=t_axis)
    return corrects / totals