from abc import ABCMeta, abstractmethod
from collections import Sequence
import logging
import os
import random

from dependency.data_directory import DataDirectory
from gtd.utils import random_seed


class Dataset(Sequence, metaclass=ABCMeta):
    """Encapsulates an entire dataset or fetches the data if necessary."""

    def __init__(self):
        self._examples = []

    def __getitem__(self, i):
        return self._examples[i]

    def __len__(self):
        return len(self._examples)


class DatasetFromFile(Dataset):
    """Dataset that is loaded from a file.
    An ExampleFactory is used to read the file and yield Examples.
    """

    # TODO: Write this to a FileSequence
    def __init__(self, filenames, filename_to_examples, relative_path=True, shuffle=True):
        """Construct the dataset based on the data in the files.

        Args:
            filenames (unicode or list[unicode]): names of the files
            filename_to_examples: a callable that takes a filename
                and yields Examples
            relative_path: whether to resolve the filename on DataDirectory.root
        """
        self._examples = []
        if isinstance(filenames, str):
            filenames = [filenames]
        for filename in filenames:
            if relative_path:
                filename = os.path.join(DataDirectory.root, filename)
            self._examples.extend(filename_to_examples(filename))
        if shuffle:
            with random_seed(42):
                random.shuffle(self._examples)
        logging.info('Read {} examples ({}) from {}'.format(
                len(self._examples), 'shuffled' if shuffle else 'not shuffled', filenames))
