def process_labels(targets_train, targets_test):
    # create dictionaries: 'label -> index' and 'index -> label',
    # i.e. ( { 0 -> 'eng', ...} , {'eng' -> 0, ...})
    label2index = {'PAD' : 0}
    all_labels = targets_train + targets_test
    index = 1
    for label in all_labels:
        if label in label2index: continue
        label2index[label] = index
        index += 1

    index2label = {index: label for label, index in label2index.items()}

    # convert targets_train and targets_test to list of indices
    targets_train_indices = [label2index[target] for target in targets_train]
    targets_test_indices = [label2index[target] for target in targets_test]
    
    return targets_train_indices, targets_test_indices, label2index, index2label
