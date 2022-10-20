import inspect
import os
import signal
import sys
import time
from collections import Mapping
from contextlib import contextmanager

import faulthandler
import line_profiler
from tqdm import tqdm, tqdm_notebook

from gtd.log import in_ipython


class Profiling(object):
    @staticmethod
    def start():
        """Enable the default profiler and reset its logging."""
        Profiler.default().enable().reset()

    @staticmethod
    def report(*args, **kwargs):
        Profiler.default().report(*args, **kwargs)


class Profiler(object):
    """Just a wrapper around line_profiler.

    Supports some extra functionality like resetting.
    """
    @classmethod
    def default(cls):
        if not hasattr(cls, '_default'):
            profiler = Profiler()
            profiler.enable_by_count()
            profiler.disable()
            cls._default = profiler
        return cls._default

    def __init__(self):
        self._line_prof = line_profiler.LineProfiler()

    def report(self, *args, **kwargs):
        self.stats.report(*args, **kwargs)

    def enable(self):
        self._line_prof.enable()
        self._enable = True
        return self

    def disable(self):
        self._line_prof.disable()
        self._enable = False
        return self

    def enable_by_count(self):
        self._line_prof.enable_by_count()
        self._enable_by_count = True
        return self

    def disable_by_count(self):
        self._line_prof.disable_by_count()
        self._enable_by_count = False
        return self

    def add_function(self, fxn):
        self._line_prof.add_function(fxn)
        return self

    def add_module(self, mod):
        """Profile all functions and class methods inside this module.

        NOTE: This includes functions that are imported into the module.
        """
        from inspect import isclass, isfunction

        for item in list(mod.__dict__.values()):
            if isclass(item):
                for k, v in list(item.__dict__.items()):
                    if isinstance(v, staticmethod) or isinstance(v, classmethod):
                        underlying_fxn = v.__get__(item)
                        self.add_function(underlying_fxn)
                    if isfunction(v):
                        self.add_function(v)
            elif isfunction(item):
                self.add_function(item)

        return self

    def add_this_module(self):
        try:
            frame = inspect.currentframe()
            mod_name = frame.f_back.f_globals['__name__']
        finally:
            del frame  # failing to delete the frame can cause garbage collection problems, due to reference counting
        mod = sys.modules[mod_name]
        return self.add_module(mod)

    @property
    def stats(self):
        return ProfilerStats(self._line_prof.get_stats(), self.functions)

    def reset(self):
        functions = self.functions
        line_prof = line_profiler.LineProfiler()
        # copy settings
        if self._enable:
            line_prof.enable()
        else:
            line_prof.disable()
        if self._enable_by_count:
            line_prof.enable_by_count()
        else:
            line_prof.disable_by_count()
        # add previously registered functions
        for fxn in functions:
            line_prof.add_function(fxn)
        self._line_prof = line_prof
        return self

    @property
    def functions(self):
        return self._line_prof.functions


def function_label(fxn):
    """Return a (filename, first_lineno, func_name) tuple for a given code object.

    This is the same labelling as used by the cProfile module in Python 2.5.
    """
    code = fxn.__code__
    if isinstance(code, str):
        return ('~', 0, code)  # built-in functions ('~' sorts at the end)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)


class ProfilerStats(Mapping):
    """Wrapper around line_profiler.LineStats"""

    def __init__(self, line_stats, functions):
        """Create a ProfilerStats object.

        Args:
            line_stats (LineStats): a LineStats object returned by LineProfiler
        """
        self._line_stats = line_stats
        self._functions = functions

    def __getitem__(self, fxn):
        """Get stats for a particular fxn.

        Args:
            fxn: a Python function

        Returns:
            FunctionStats
        """
        label = function_label(fxn)
        return FunctionStats(fxn, self._line_stats.timings[label], self._line_stats.unit)

    def __iter__(self):
        return iter(self._functions)

    def __len__(self):
        return len(self._functions)

    def report(self, fxns=None):
        if fxns is None:
            fxns = list(self.keys())

        fxn_stats = [self[f] for f in fxns]
        fxn_stats = sorted(fxn_stats, key=lambda stats: stats.total_time, reverse=True)

        for stats in fxn_stats:
            if stats.empty: continue
            print(stats)


class FunctionStats(object):
    def __init__(self, function, timing, unit):
        """Create a FunctionStats object.

        Args:
            function: a Python function
            timing (list[(int, int, int)]): a list of (lineno, nhits, total_time) tuples, one per line
            unit: unit of time (e.g. seconds)
        """
        self._function = function
        self._timing = timing
        self._unit = unit

    @property
    def function(self):
        return self._function

    @property
    def _line_stats_in_seconds(self):
        """Line stats in seconds.

        Returns:
            list[(int, int, float)]: a list of (line_number, number_of_hits, total_time_in_seconds) tuples, one per line
        """
        return [(lineno, nhits, total_time * self._unit) for (lineno, nhits, total_time) in self._timing]

    def __repr__(self):
        label = function_label(self.function)
        timings = {label: self._line_stats_in_seconds}  # format needed for show_text
        unit = 1.

        class Stream(object):
            def __init__(self):
                self.items = []
            def write(self, s):
                self.items.append(s)
            def get_value(self):
                return ''.join(self.items)

        output = Stream()
        line_profiler.show_text(timings, unit, output)
        s = output.get_value()
        return s

    @property
    def empty(self):
        return len(self._timing) == 0

    @property
    def total_time(self):
        """Total time spent by this function, in seconds."""
        return sum([t for _, _, t in self._line_stats_in_seconds], 0)


def profile(f):
    """A decorator for functions you want to profile"""
    Profiler.default().add_function(f)
    return f


@contextmanager
def timer(name='unnamed'):
    print('Start: {}'.format(name))
    sys.stdout.flush()
    start = time.time()
    yield
    stop = time.time()
    print('Finish: {} ({} s)'.format(name, stop - start))
    sys.stdout.flush()


def verboserate(iterable, *args, **kwargs):
    """Iterate verbosely.

    Args:
        desc (str): prefix for the progress bar
        total (int): total length of the iterable
        See more options for tqdm.tqdm.

    """
    progress = tqdm_notebook if in_ipython() else tqdm
    for val in progress(iterable, *args, **kwargs):
        yield val


class Pulse(object):
    def __init__(self, wait):
        self.wait = wait
        self.prev = time.time()

    def __call__(self):
        """Check if it's time to pulse.

        If enough time has passed since previous pulse, return True and reset timer.
        Otherwise, return False (don't reset timer).
        """
        now = time.time()
        long_enough = now - self.prev > self.wait

        if long_enough:
            self.prev = now

        return long_enough

    def reset(self):
        """Force reset the timer."""
        self.prev = time.time()


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException('Timed out!')

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def monitor_call_stack():
    if in_ipython():
        # see this issue for why: https://github.com/ipython/ipykernel/issues/91
        f = sys.__stderr__
    else:
        f = sys.stderr

    faulthandler.register(signal.SIGUSR1, file=f)
    print('To monitor call stack, type this at command line: kill -USR1 {}'.format(os.getpid()))
    print('Call stack will be printed to stderr' \
          '(in IPython Notebook, this will show in the terminal where you launched the notebook.)')
