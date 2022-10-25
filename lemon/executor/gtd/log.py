import json
import logging
import math
import numbers
import os
import platform
import resource
import sys
from collections import MutableMapping
from contextlib import contextmanager

from IPython.core.display import display, HTML
from pyhocon import ConfigFactory
from pyhocon import ConfigMissingException
from pyhocon import ConfigTree
from pyhocon import HOCONConverter

from gtd.utils import NestedDict, Config


def in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def print_with_fonts(tokens, sizes, colors, background=None):

    def style(text, size=12, color='black'):
        return '<span style="font-size: {}px; color: {};">{}</span>'.format(size, color, text)

    styled = [style(token, size, color) for token, size, color in zip(tokens, sizes, colors)]
    text = ' '.join(styled)

    if background:
        text = '<span style="background-color: {};">{}</span>'.format(background, text)

    display(HTML(text))


def gb_used():
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        # on Linux, used is in terms of kilobytes
        power = 2
    else:
        # on Mac, used is in terms of bytes
        power = 3
    return float(used) / math.pow(1024, power)


class Metadata(MutableMapping):
    """A wrapper around ConfigTree.

    Supports a name_scope contextmanager.
    """
    def __init__(self, config_tree=None):
        if config_tree is None:
            config_tree = ConfigTree()

        self._config_tree = config_tree
        self._namestack = []

    @contextmanager
    def name_scope(self, name):
        self._namestack.append(name)
        yield
        self._namestack.pop()

    def _full_key(self, key):
        return '.'.join(self._namestack + [key])

    def __getitem__(self, key):
        try:
            val = self._config_tree.get(self._full_key(key))
        except ConfigMissingException:
            raise KeyError(key)

        if isinstance(val, ConfigTree):
            return Metadata(val)
        return val

    def __setitem__(self, key, value):
        """Put a value (key is a dot-separated name)."""
        self._config_tree.put(self._full_key(key), value)

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._config_tree)

    def __len__(self):
        return len(self._config_tree)

    def __repr__(self):
        return self.to_str()

    def to_str(self):
        return HOCONConverter.convert(self._config_tree, 'hocon')

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(self.to_str())

    @classmethod
    def from_file(cls, path):
        config_tree = ConfigFactory.parse_file(path)
        return cls(config_tree)


class SyncedMetadata(Metadata):
    """A Metadata object which writes to file after every change."""
    def __init__(self, path):
        if os.path.exists(path):
            m = Metadata.from_file(path)
        else:
            m = Metadata()

        super(SyncedMetadata, self).__init__(m._config_tree)
        self._path = path

    def __setitem__(self, key, value):
        super(SyncedMetadata, self).__setitem__(key, value)
        self.to_file(self._path)


def print_list(l):
    for item in l:
        print(item)


def print_no_newline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def set_log_level(level):
    """Set the log-level of the root logger of the logging module.

    Args:
        level: can be an integer such as 30 (logging.WARN), or a string such as 'WARN'
    """
    if isinstance(level, str):
        level = logging._levelNames[level]

    logger = logging.getLogger()  # gets root logger
    logger.setLevel(level)