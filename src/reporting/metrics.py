import numpy.ma as ma
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_loss(log_probs, targets, loss_criterion):
    loss = loss_criterion(log_probs.permute(0,2,1), targets)
    return loss.item()

def calculate_accuracy(log_probs, targets, t_axis=None):
    """
    Calculates the average accuracy (ignoring the paddings)

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
    mask = targets == 0
    masked_targets = ma.MaskedArray(targets, mask)

    predictions = log_probs.argmax(axis=2)
    masked_predictions = ma.MaskedArray(predictions, mask)

    corrects = (masked_predictions == masked_targets).sum(axis=t_axis)
    totals = ma.count(masked_targets, axis=t_axis)
    return corrects / totals


def calculate_confusion_matrix(log_probs, targets, counts=False, include_padding=False):
    """
    Calculates the confusion matrix (ignoring the paddings)

    With 'N' the batch_size, 'S' the max character sequence length and 'O' the nr of output classes
    Args:
        log_probs: N x S x O probability of the output classes  
        targets: N x S target class (all positions in the sequence have in fact the same output class)
        counts: bool flag for outputting counts of confusion or percentages
    Returns:
        confusion matrix for different languages
        list of language indices
    """
    mask = targets == 0
    predictions = log_probs.argmax(axis=2)
    
    targets, predictions = targets[~mask], predictions[~mask]
    confusion_mat = confusion_matrix(targets, predictions)

    # remove languages which were not present in the target
    language_mask = confusion_mat.sum(axis=1) > 0
    confusion_mat = confusion_mat[language_mask]

    targets_languages = np.unique(targets)
    predictions_languages = np.unique(predictions)

    if not counts:
        confusion_mat = confusion_mat/confusion_mat.sum(axis=1)[:, None]
        
    return confusion_mat, (targets_languages, predictions_languages)
