import os

def get_all_files(directory, skipext=None):
    """Returns the absolute path to all files in the directory."""
    file_list = []
    for _file in os.listdir(directory):
        if skipext and skipext in file:
            continue
        full_path = os.path.join(directory, _file)
        file_list.append(full_path)

    return sorted(file_list)

def get_all_files_containing(directory, containing, ext=None):
    """Returns the absolute path to all files in that have containgin in filename"""
    file_list = []
    for _file in os.listdir(directory):
        if containing in _file:
            if not ext and ext != _file.split('.')[-1]:
                continue
            full_path = os.path.join(directory, _file)
            file_list.append(full_path)

    return sorted(file_list)

def split_lists(list_of_lists, train_ratio, val_ratio, test_ratio):
    """Splits all the lists into the ratios.

    The sublists should all be of the same size N for this method.

    Parameters
    ----------
        list_of_lists: list
            A list of lists containing the data that should be split
            into the test and validation sets
        train_ratio: Ratio of the lists that should be training data
        val_ratio: Ratio of the lists that should be validation data
        test_ratio: Ratio of the lists that should be test data

    Returns
    -------
        If N sublists where given, the output would be 3 lists of size N
        where the list order is: training, validation, test

    """
    n_samples = len(list_of_lists[0])
    train_idx = int(n_samples * train_ratio)
    val_idx = train_idx + int(n_samples * val_ratio)

    train_split = []
    val_split = []
    test_split = []
    for iList in list_of_lists:
        train = iList[:train_idx]
        val = iList[train_idx:val_idx]
        test = iList[val_idx:]

        train_split.append(train)
        val_split.append(val)
        test_split.append(test)

    return train_split, val_split, test_split