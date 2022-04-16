

"""
We iterate over our data in given batches to improve performance
We also ensure that examples in each batch are shuffled to reduce over fitting to the data

This code is adapted from Joel Grus's Git hub that was instrumental getting me started with Neural Nets.
https://github.com/joelgrus/joelnet
Some variable changes have been made no additions were made except comments.


"""
# Iterator generic used for type hinting Iterators
# Using Named tuples are tuples that can be accessed by names like dictionary of tuples but are immutable
# A general tensor class to short hand np.array as it used quiet allot in the module
from typing import Iterator, NamedTuple
import numpy as np
from tensor import Tensor


Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])



class DataIterator:
    # use call dunder to make this class callable
    # not impemented error used to ensure any inheritor of data iterator must have implemented a 'call' Dunder function
    # the function returns an iterator of batches- yeild statement used to access the next value
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        print(f' DataIterator created')
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 50, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        print(f' BatchIterator(DataIterator) class created')

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:

        print(f' BatchIterator(DataIterator) object called')
        # array of integers of interval size batch size
        array_of_starts = np.arange(0, len(inputs), self.batch_size)
        # helps reduce the problem of over fitting without looking to another dataset
        if self.shuffle:
            np.random.shuffle(array_of_starts)

        for start in array_of_starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]

            yield Batch(batch_inputs, batch_targets)

