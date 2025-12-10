import os


def get_file_list(dataset_dir, sort_func=None, reverse_sort=False):
    filenames = os.listdir(dataset_dir)
    if sort_func is None:
        filenames.sort(reverse=reverse_sort)
    else:
        filenames.sort(key=sort_func, reverse=reverse_sort)

    file_list = [os.path.join(dataset_dir, filename) for filename in filenames]
    return file_list


def get_cell_label_file_list(dataset_dir, sort_func=None, reverse_sort=False):
    data_dir = os.path.join(dataset_dir, 'data')
    labels_dir = os.path.join(dataset_dir, 'labels')
    data_filenames = os.listdir(data_dir)
    labels_filenames = os.listdir(labels_dir)

    if sort_func is None:
        data_filenames.sort(reverse=reverse_sort)
        labels_filenames.sort(reverse=reverse_sort)
    else:
        data_filenames.sort(key=sort_func, reverse=reverse_sort)
        labels_filenames.sort(key=sort_func, reverse=reverse_sort)

    file_list = [(os.path.join(data_dir, item[0]), os.path.join(labels_dir, item[1])) for item in zip(data_filenames, labels_filenames)]
    return file_list
