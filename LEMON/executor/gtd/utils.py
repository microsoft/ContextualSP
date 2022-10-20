'''
Created on Oct 23, 2015

@author: kelvinguu
'''
import logging
import operator
import os.path
import random
import shutil
import traceback
import types
import json
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, defaultdict, MutableMapping, Mapping
from contextlib import contextmanager

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyhocon import ConfigTree, HOCONConverter, ConfigFactory

# from gtd.io import makedirs


def sorted_by_value(d, ascending=True):
    return OrderedDict(sorted(list(d.items()), key=operator.itemgetter(1), reverse=not ascending))


class FunctionWrapper(object, metaclass=ABCMeta):
    """Turn a function or method into a callable object.

    Can be used as a decorator above method definitions, e.g.

    class Something(object):
        ...
        @FunctionWrapper
        def some_method(self, ...):
            ...

    Or, bound methods of an instance can be directly overriden
        obj = Something()
        obj.some_method = FunctionWrapper(obj.some_method)
    """

    def __init__(self, fxn):
        self._orig_fxn = fxn

    @property
    def orig_fxn(self):
        return self._orig_fxn

    def __get__(self, instance, objtype=None):
        """Implement descriptor functionality."""
        return self.as_method(instance, objtype)

    def as_method(self, instance, objtype=None):
        """Make this object a method of the given object instance.

        Args:
            instance: any object instance
        """
        return types.MethodType(self, instance, objtype)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Memoized(FunctionWrapper, metaclass=ABCMeta):
    def __init__(self, fxn):
        """Create memoized version of a function.

        Args:
            fxn (Callable): function to be memoized
        """
        super(Memoized, self).__init__(fxn)
        self._cache_hits = 0
        self._calls = 0.

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.pop('use_cache', True)
        if not use_cache:
            return self.orig_fxn(*args, **kwargs)

        key = self._cache_key(args, kwargs)
        # logging.debug('cache key: {}'.format(key))
        if self._in_cache(key):
            # logging.debug('load from cache')
            self._cache_hits += 1  # successfully return from cache
            return self._from_cache(key)

        # logging.debug('compute and save to cache')
        val = self.orig_fxn(*args, **kwargs)
        self._to_cache(key, val)
        return val

    @property
    def hit_rate(self):
        if self._calls <= 0:
            return 0.
        return self._cache_hits / self._calls

    @abstractmethod
    def _cache_key(self, args, kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear_cache(self):
        raise NotImplementedError

    @abstractmethod
    def _in_cache(self, key):
        raise NotImplementedError

    @abstractmethod
    def _from_cache(self, key):
        raise NotImplementedError

    @abstractmethod
    def _to_cache(self, key, val):
        raise NotImplementedError

    @abstractproperty
    def cache_size(self):
        pass


class DictMemoized(Memoized):
    def __init__(self, fxn, custom_key_fxn=None):
        super(DictMemoized, self).__init__(fxn)
        self.cache = {}
        self._custom_key_fxn = custom_key_fxn

    def _cache_key(self, args, kwargs):
        if self._custom_key_fxn:
            return self._custom_key_fxn(*args, **kwargs)
        kwargs_key = tuple(sorted(kwargs.items()))
        return (args, kwargs_key)

    def clear_cache(self):
        self.cache = {}

    def _in_cache(self, key):
        return key in self.cache

    def _from_cache(self, key):
        return self.cache[key]

    def _to_cache(self, key, val):
        self.cache[key] = val

    @property
    def cache_size(self):
        return len(self.cache)


def memoize(fxn):
    return DictMemoized(fxn)


def memoize_with_key_fxn(key_fxn):
    return lambda fxn: DictMemoized(fxn, custom_key_fxn=key_fxn)


def args_as_string(args, kwargs):
    args_str = '_'.join([str(a) for a in args])
    kwargs_str = '_'.join(['{}={}'.format(k, v) for k, v in kwargs.items()])
    items = [args_str, kwargs_str]
    items = [s for s in items if s]  # remove empty elements
    key_str = '_'.join(items)
    if not key_str:
        key_str = 'NO_KEY'
    return key_str


class FileMemoized(Memoized):
    def __init__(self, fxn, cache_dir, serialize, deserialize):
        super(FileMemoized, self).__init__(fxn)
        self.cache_dir = cache_dir
        self.serialize = serialize
        self.deserialize = deserialize
        makedirs(cache_dir)

    def _cache_key(self, args, kwargs):
        """Compute the name of the file."""
        key_str = args_as_string(args, kwargs)
        return os.path.join(self.cache_dir, '{}.txt'.format(key_str))

    def _in_cache(self, key):
        return os.path.exists(key)

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)
        makedirs(self.cache_dir)

    def _to_cache(self, key, val):
        with open(key, 'w') as f:
            self.serialize(f, val)

    def _from_cache(self, key):
        with open(key, 'r') as f:
            return self.deserialize(f)

    @property
    def cache_size(self):
        raise NotImplementedError


def file_memoize(cache_dir, serialize, deserialize):
    return lambda fxn: FileMemoized(fxn, cache_dir, serialize, deserialize)


def sample_if_large(arr, max_size, replace=True):
    if len(arr) > max_size:
        idx = np.random.choice(len(arr), size=max_size, replace=replace)
        return [arr[i] for i in idx]

    return list(arr)


def flatten(lol):
    """
    Flatten a list of lists
    """
    return [item for sublist in lol for item in sublist]


def chunks(l, n):
    """
    Return a generator of lists, each of size n (the last list may be less than n)
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ensure_unicode(s):
    assert isinstance(s, str)
    if not isinstance(s, str):
        s = str(s, 'utf-8')
    return s


class UnicodeMixin(object):
    __slots__ = []
    @abstractmethod
    def __unicode__(self):
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self).encode('utf-8')


class EqualityMixinSlots(object):
    """Equality mixin for classes using __slots__"""
    __slots__ = []

    class Missing(object):
        pass  # just a special object to denote that a value is missing. Is only equal to itself.

    __MISSING = Missing()

    @property
    def _slot_vals(self):
        vals = []
        for slots in [getattr(cls, '__slots__', tuple()) for cls in type(self).__mro__]:
            for slot in slots:
                try:
                    val = getattr(self, slot)
                except AttributeError:
                    val = self.__MISSING
                vals.append(val)
        return tuple(vals)

    def __eq__(self, other):
        # must be strictly same type
        if type(other) != type(self):
            return False
        if self._slot_vals != other._slot_vals:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._slot_vals)


class EqualityMixin(object):
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def data_split(items, dev_part=0.1, test_part=0.1):
    # don't allow duplicates
    assert len(set(items)) == len(items)

    # remaining portion is set aside for train
    assert dev_part + test_part < 1.0

    items_copy = list(items)
    random.shuffle(items_copy)

    n = len(items_copy)
    ndev = int(n * dev_part)
    ntest = int(n * test_part)

    dev = items_copy[:ndev]
    test = items_copy[ndev:ndev + ntest]
    train = items_copy[ndev + ntest:]

    # verify that there is no overlap
    train_set = set(train)
    dev_set = set(dev)
    test_set = set(test)

    assert len(train_set.intersection(dev_set)) == 0
    assert len(train_set.intersection(test_set)) == 0

    print(('train {}, dev {}, test {}'.format(len(train), len(dev), len(test))))
    return train, dev, test


def compute_if_absent(d, key, keyfunc):
    val = d.get(key)
    if val is None:
        val = keyfunc(key)
        d[key] = val
    return val


class Bunch(object):
    """A simple class for holding arbitrary attributes. Recommended by the famous Martelli bot."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return repr(self.__dict__)


def best_threshold(scores, labels, debug=False):
    # find best threshold in O(nlogn)
    # does not handle scores of infinity or -infinity
    items = list(zip(scores, labels))
    items.sort()
    total = len(items)
    total_pos = len([l for l in labels if l])

    def accuracy(p, n):
        correct_n = n
        correct_p = total_pos - p
        return float(correct_n + correct_p) / total

    # predict True iff score > thresh
    pos = 0  # no. pos <= thresh
    neg = 0  # no. neg <= thresh

    thresh_accs = [(float('-inf'), accuracy(pos, neg))]
    for thresh, label in items:
        if label:
            pos += 1
        else:
            neg += 1
        thresh_accs.append((thresh, accuracy(pos, neg)))

    if debug:
        import matplotlib.pyplot as plt
        from gtd.plot import plot_pdf
        x, y = list(zip(*thresh_accs))
        plt.figure()
        plt.plot(x, y)
        pos_scores = [s for s, l in items if l]
        neg_scores = [s for s, l in items if not l]
        plot_pdf(pos_scores, 0.1, color='b')
        plot_pdf(neg_scores, 0.1, color='r')
        plt.show()

    return max(thresh_accs, key=operator.itemgetter(1))[0]


def as_batches(l, batch_size):
    assert batch_size >= 1
    batch = []
    for item in l:
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)

    # final batch may be smaller
    if len(batch) != 0:
        yield batch


# TODO: test
def get_batch(data, batch_size, k):
    """Get the kth batch from a data sequence

    If the final batch is less than batch_size, this function loops back to the beginning of data
    so that the returned batch is exactly batch_size.

    Args:
        data: a list of examples
        batch_size: the size of the returned batch
        k: the batch index you want to get.
    """
    return [data[i % len(data)] for i in range(k * batch_size, (k + 1) * batch_size)]


# TODO: test
def batch_compute(data, batch_fxn, batch_size):
    """Evaluate the batch function on a list of items.

    Args:
        data: a list of examples
        batch_fxn: a function which only accepts a list of exactly length batch_size,
            and returns a list of the same length
        batch_size: the batch size

    Returns:
        a list of length = len(data)
    """
    n = len(data)
    num_batches = n / batch_size + 1
    final_trim_size = n % batch_size

    # map
    results = []
    for k in range(num_batches):
        batch = get_batch(data, batch_size, k)  # circles around
        result = batch_fxn(batch)
        results.append(result)

    # remove the examples that looped around to the beginning of data
    results[-1] = results[-1][:final_trim_size]

    return flatten(results)


def fixed_length(l, length, pad_val):
    """Given a list of arbitrary length, make it fixed length by padding or truncating.

    (Makes a shallow copy of l, then modifies this copy.)

    Args:
        l: a list
        length: desired length
        pad_val: values padded to the end of l, if l is too short

    Returns:
        a list of with length exactly as specified.
    """
    if len(l) < length:
        fixed = list(l)  # make shallow copy
        fixed += [pad_val] * (length - len(l))  # pad
        return fixed
    else:
        return l[:length]  # truncate


class HomogeneousBatchSampler(object):
    def __init__(self, data, bucket_fxn):
        buckets = defaultdict(list)
        for d in data:
            buckets[bucket_fxn(d)].append(d)

        keys = list(buckets.keys())
        freqs = np.array([len(buckets[k]) for k in keys], dtype=float)
        probs = freqs / np.sum(freqs)

        self.keys = keys
        self.probs = probs
        self.buckets = buckets

    def sample(self, batch_size):
        # WARNING! This sampling scheme is only "correct" if each len(bucket) > batch_size

        # sample a bucket according to its frequency
        key = np.random.choice(self.keys, p=self.probs)
        bucket = self.buckets[key]

        # sample a batch from the bucket
        batch = np.random.choice(bucket, size=batch_size, replace=True)
        return batch


class Frozen(object):
    """Objects that inherit from Frozen cannot set or add new attributes unless inside an `unfreeze` context."""

    __frozen = True

    @staticmethod
    @contextmanager
    def unfreeze():
        prev_state = Frozen.__frozen
        Frozen.__frozen = False
        yield
        Frozen.__frozen = prev_state  # set back to previous state

    def __setattr__(self, key, value):
        if Frozen.__frozen:
            raise NotImplementedError('Object is frozen.')
        else:
            super(Frozen, self).__setattr__(key, value)

    def __delattr__(self, item):
        if Frozen.__frozen:
            raise NotImplementedError('Object is frozen.')
        else:
            super(Frozen, self).__delattr__(item)


def sigmoid(x):
    # scipy.special.expit will return NaN if x gets larger than about 700, which is just wrong

    # compute using two different approaches
    # they are each stable over a different interval of x
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        numer = np.exp(x)
        s0 = numer / (1.0 + numer)

        denom = 1.0 + np.exp(-x)
        s1 = 1.0 / denom

    # replace nans
    if isinstance(x, float):
        if np.isnan(s0):
            s0 = s1
    else:
        nans = np.isnan(s0)
        s0[nans] = s1[nans]

    return s0


class NestedDict(MutableMapping):
    def __init__(self, d=None):
        """Create a NestedDict.

        Args:
            d (dict): a nested Python dictionary. Defaults to an empty dictionary.

        NOTE: if d contains empty dicts at its leaves, these will be dropped.
        """
        if d is None:
            d = {}

        self.d = {}
        for keys, val in self._flatten(d).items():
            self.set_nested(keys, val)

    def __iter__(self):
        """Iterate through top-level keys."""
        return iter(self.d)

    def __delitem__(self, key):
        del self.d[key]

    def __getitem__(self, key):
        return self.d[key]

    def __len__(self):
        """Total number of leaf nodes."""
        l = 0
        for v in self.values():
            if isinstance(v, NestedDict):
                l += len(v)
            else:
                l += 1
        return l

    def __setitem__(self, key, value):
        """Set a key-value pair.

        If value is a Mapping, it will be converted into a NestedDict.
        """
        if isinstance(value, Mapping):
            value = NestedDict(value)
        self.d[key] = value

    def get_nested(self, keys):
        d = self
        for k in keys:
            if not isinstance(d, NestedDict):
                raise KeyError(keys)
            d = d[k]
        return d

    def set_nested(self, keys, val):
        first_keys, last_key = keys[:-1], keys[-1]

        d = self
        for k in first_keys:
            if k not in d:
                d[k] = NestedDict()
            d = d[k]

        d[last_key] = val

    def __repr__(self):
        return repr(self.d)

    def as_dict(self):
        d = {}
        for key, sub in self.items():
            if isinstance(sub, NestedDict):
                val = sub.as_dict()
            else:
                val = sub
            d[key] = val
        return d

    @staticmethod
    def _flatten(d):
        flattened = {}

        def helper(key_tuple, d):
            if not isinstance(d, Mapping):  # leaf node
                flattened[key_tuple] = d
                return
            for key, val in d.items():
                helper(key_tuple + (key,), val)

        helper(tuple(), d)
        return flattened

    def flattened(self):
        return self._flatten(self)

    def leaves(self):
        return list(self.flattened().values())


def ranks(scores, ascending=True):
    """Assign a rank to each score.

    Args:
        scores (list[float]): a list of scores
        ascending (bool): if True, then higher scores will have smaller rank

    Returns:
        list[int]: a list of ranks, where ranks[i] is the rank of the value scores[i]
    """
    if isinstance(scores, list):
        scores = np.array(scores)
    else:
        assert len(scores.shape) == 1

    flip = 1 if ascending else -1
    idx = np.argsort(flip * scores)
    ranks = np.empty(scores.shape, dtype=int)
    ranks[idx] = np.arange(len(scores))
    # ranks should start from 1
    ranks += 1
    return list(ranks)


def quantiles(vals, ps):
    vals = sorted(vals)
    max_idx = len(vals) - 1

    qs = []
    for p in ps:
        assert 0 <= p <= 1
        i = int(round(max_idx * p))
        qs.append(vals[i])

    return qs


def sample_excluding(items, exclude):
    candidates = list(items)  # shallow copy
    random.shuffle(candidates)
    for cand in candidates:
        if cand not in exclude:
            return cand
    # if everything is excluded, return None
    return None


def map_array(fxn, array):
    """Apply fxn to all elements of array.

    Args:
        fxn: a function
        array: a list of lists of lists of ... If it is a numpy array, converts it to a list.

    Returns:
        a new array, mapped

    >>> arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> map_array(lambda x: 10 * x, arr)
    [[[10, 20], [30, 40]], [[50, 60], [70, 80]]]
    """
    if isinstance(array, np.ndarray):
        array = array.tolist()
    new_array = []
    for val in array:
        new_val = map_array(fxn, val) if isinstance(val, list) else fxn(val)
        new_array.append(new_val)
    return new_array


def group(items, grouper):
    d = defaultdict(list)
    for item in items:
        labels = grouper(item)
        for label in labels:
            d[label].append(item)
    return d


# TODO(kelvin): test this
def generator_ignore_errors(iterator):
    """Loop through iterator, but ignore exceptions.

    Logs a warning if there is an exception.

    Args:
        iterator: any object with a __next__ method

    Yields:
        the next element of the iterator
    """
    i = 0
    while True:
        try:
            try:
                yield next(iterator)
            except StopIteration:
                # stop when we're out of elements
                break
        except Exception:
            # If this generator is closed before it is exhausted (e.g. if we break out of a for-loop)
            # it will get garbage collected, and throw a GeneratorExit error
            # GeneratorExit does not inherit from Exception in Python >2.6, so we will not catch it here
            # Critically, this line should NOT be changed to just "except:", as it would catch GeneratorExit
            logging.warn('Error parsing line {}'.format(i))
        i += 1


class SimpleExecutor(object):
    def __init__(self, fxn, max_workers=120):
        self._fxn = fxn
        self._executor = ThreadPoolExecutor(max_workers)
        self._future_to_key = {}  # map from future to a key for later access

    def submit(self, key, x):
        future = self._executor.submit(self._fxn, x)
        self._future_to_key[future] = key

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def results(self):
        for future in as_completed(self._future_to_key):
            key = self._future_to_key[future]
            try:
                result = future.result()
            except BaseException:
                f = Failure.with_message('SimpleExecutor failed to compute key: {}'.format(key))
                logging.error(f.traceback)
                result = f
            yield key, result

    def shutdown(self):
        self._executor.shutdown()


class Failure(object):
    """Represents the result of a failed computation."""

    @staticmethod
    def with_message(msg):
        f = Failure(message=msg)
        logging.error(f.message)
        return f

    @staticmethod
    def silent(msg):
        return Failure(message=msg)

    def __init__(self, uid=None, message='Failure'):
        if uid is None:
            uid = id(self)
        self._uid = uid
        self._msg = message
        self._traceback = traceback.format_exc()

    def __repr__(self):
        return self._msg

    @property
    def uid(self):
        return self._uid

    @property
    def traceback(self):
        return self._traceback

    @property
    def message(self):
        return self._msg

    def __eq__(self, other):
        if not isinstance(other, Failure):
            return False
        return self.uid == other.uid

    def __ne__(self, other):
        return not self.__eq__(other)


@contextmanager
def random_seed(seed=None):
    """Execute code inside this with-block using the specified seed.

    If no seed is specified, nothing happens.

    Does not affect the state of the random number generator outside this block.
    Not thread-safe.

    Args:
        seed (int): random seed
    """
    if seed is None:
        yield
    else:
        py_state = random.getstate()  # save state
        np_state = np.random.get_state()

        random.seed(seed)  # alter state
        np.random.seed(seed)
        yield

        random.setstate(py_state)  # restore state
        np.random.set_state(np_state)


class cached_property(object):
    """Descriptor (non-data) for building an attribute on-demand on first use."""
    def __init__(self, factory):
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)

        return attr


class set_once_attribute(object):
    def __init__(self, attr_name):
        self._attr_name = attr_name

    def __get__(self, instance, owner):
        return getattr(instance, self._attr_name)

    def __set__(self, instance, value):
        if hasattr(instance, self._attr_name):
            raise RuntimeError('Cannot set {} more than once.'.format(self._attr_name))
        setattr(instance, self._attr_name, value)


class Config(object):
    """A wrapper around the pyhocon ConfigTree object.

    Allows you to access values in the ConfigTree as attributes.
    """
    def __init__(self, config_tree=None):
        """Create a Config.

        Args:
            config_tree (ConfigTree)
        """
        if config_tree is None:
            config_tree = ConfigTree()
        self._config_tree = config_tree

    def __getattr__(self, item):
        val = self._config_tree[item]
        if isinstance(val, ConfigTree):
            return Config(val)
        else:
            return val

    def get(self, key, default=None):
        val = self._config_tree.get(key, default)
        if isinstance(val, ConfigTree):
            return Config(val)
        else:
            return val

    def put(self, key, value, append=False):
        """Put a value into the Config (dot separated)

        Args:
            key (str): key to use (dot separated). E.g. `a.b.c`
            value (object): value to put
        """
        self._config_tree.put(key, value, append=append)

    def __repr__(self):
        return self.to_str()

    def to_str(self):
        return HOCONConverter.convert(self._config_tree, 'hocon')

    def to_json(self):
        return json.loads(HOCONConverter.convert(self._config_tree, 'json'))

    def to_file(self, path):
        with open(path, 'w') as f:
            f.write(self.to_str())

    @classmethod
    def from_file(cls, path):
        config_tree = ConfigFactory.parse_file(path)
        return cls(config_tree)

    @classmethod
    def from_dict(cls, d):
        return Config(ConfigFactory.from_dict(d))

    @classmethod
    def merge(cls, config1, config2):
        assert isinstance(config1, Config)
        assert isinstance(config2, Config)
        return cls(ConfigTree.merge_configs(config1._config_tree, config2._config_tree))


def softmax(logits):
    """Take the softmax over a set of logit scores.

    Args:
        logits (np.array): a 1D numpy array

    Returns:
        a 1D numpy array of probabilities, of the same shape.
    """
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits)  # 1D array

    logits = logits - np.max(logits)  # re-center
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs


def bleu(reference, predict):
    """Compute sentence-level bleu score.

    Args:
        reference (list[str])
        predict (list[str])
    """
    from nltk.translate import bleu_score

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # TODO(kelvin): is this quite right?
    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    n = min(4, len(reference), len(predict))
    weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
    return bleu_score.sentence_bleu([reference], predict, weights)


class ComparableMixin(object, metaclass=ABCMeta):
    __slots__ = []

    @abstractproperty
    def _cmpkey(self):
        pass

    def _compare(self, other, method):
        try:
            return method(self._cmpkey, other._cmpkey)
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other".
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
