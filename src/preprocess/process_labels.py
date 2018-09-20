def process_labels(targets_train, targets_test):
    # create dictionaries: 'label -> index' and 'index -> label',
    # i.e. ( { 0 -> 'eng', ...} , {'eng' -> 0, ...})
    label_to_index = {}
    all_labels = targets_train + targets_test
    index = 0
    for label in all_labels:
        if label in label_to_index: continue
        label_to_index[label] = index
        index += 1

    index_to_label = {index: label for label, index in label_to_index.items()}

    # convert targets_train and targets_test to list of indices
    targets_train_indices = [label_to_index[target] for target in targets_train]
    targets_test_indices = [label_to_index[target] for target in targets_test]
    
    return targets_train_indices, targets_test_indices, label_to_index, index_to_label
