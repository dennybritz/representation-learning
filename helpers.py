import numpy as np
import random

def batch_iter(data, batch_size, num_epochs, seed=None, fill=False):
    """
    Generates a batch iterator for a dataset.
    """
    random = np.random.RandomState(seed)
    data = np.array(data)
    data_length = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    if len(data) % batch_size != 0:
        num_batches_per_epoch += 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = random.permutation(np.arange(data_length))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_length)
            selected_indices = shuffle_indices[start_index:end_index]
            # If we don't have enough data left for a whole batch, fill it randomly
            if fill is True and end_index >= data_length:
                num_missing = batch_size - len(selected_indices)
                selected_indices = np.concatenate([selected_indices, random.randint(0, data_length, num_missing)])
            yield data[selected_indices]