# import cPickle as pickle
import pickle
import codecs
import contextlib
import gzip
import json
import os
import random
import shutil
import subprocess
import sys
import time
from queue import Queue, Empty
from abc import ABCMeta, abstractmethod
from collections import Mapping, OrderedDict

from os.path import join
from threading import Thread

import jsonpickle
import numpy as np
from fabric.api import local, settings
from fabric.context_managers import hide


class MultiStream(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class redirect_stream(object):
    """Inside this context manager, inputs to a target stream are redirected to a replacement stream instead."""

    def __init__(self, replacement):
        """Redirect.

        Args:
            replacement: replace the target stream with this stream.
        """
        self._replacement = replacement

    @property
    def target_stream(self):
        """Get the target stream."""
        raise NotImplementedError

    @target_stream.setter
    def target_stream(self, s):
        """Set the target stream."""
        raise NotImplementedError

    def __enter__(self):
        self._original = self.target_stream  # save the original stream
        self.target_stream = self._replacement

    def __exit__(self, exc_type, exc_value, traceback):
        self._replacement.flush()
        self.target_stream = self._original  # put the original stream back


class redirect_stdout(redirect_stream):
    @property
    def target_stream(self):
        return sys.stdout

    @target_stream.setter
    def target_stream(self, s):
        sys.stdout = s


class redirect_stderr(redirect_stream):
    @property
    def target_stream(self):
        return sys.stderr

    @target_stream.setter
    def target_stream(self, s):
        sys.stderr = s


class save_stdout(object):
    def __init__(self, save_dir):
        makedirs(save_dir)
        save_file = lambda filename: open(join(save_dir, filename), 'a')
        self._f_out = save_file('stdout.txt')
        self._f_err = save_file('stderr.txt')

        self._redirects = [redirect_stdout(MultiStream(self._f_out, sys.stdout)),
                           redirect_stderr(MultiStream(self._f_err, sys.stderr))]

    def __enter__(self):
        for r in self._redirects:
            r.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for r in self._redirects:
            r.__exit__(exc_type, exc_val, exc_tb)
        self._f_out.close()
        self._f_err.close()


def utfopen(path, mode):
    """Open a file with UTF-8 encoding."""
    return codecs.open(path, mode, encoding='utf-8')


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def work_in_sandbox(directory):
    """Create a sandbox directory, and set cwd to sandbox.

    Deletes any existing sandbox directory!

    Args:
        directory: directory in which to put sandbox directory
    """
    os.chdir(directory)
    p = 'sandbox'
    if os.path.exists(p):  # remove if already exists
        shutil.rmtree(p)
    os.makedirs(p)
    os.chdir(p)
    print((os.getcwd()))


def makedirs(directory):
    """If directory does not exist, make it.

    Args:
        directory (str): a path to a directory. Cannot be the empty path.
    """
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)


def reset_state():
    # Reset all random seeds, as well as TensorFlow default graph
    random.seed(0)
    np.random.seed(0)
    import tensorflow as tf
    from tensorflow.python.framework import ops
    tf.set_random_seed(0)
    ops.reset_default_graph()


class EmptyFile(object):
    """Delivers a never-ending stream of empty strings."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        return ''


def read_files(*file_paths):
    files = []
    for i, p in enumerate(file_paths):
        if p:
            files.append(open(p, mode="r"))
            print(('Opened:', p))
        else:
            files.append(EmptyFile())
            print(('WARNING: no path provided for file {} in list.'.format(i)))

    with contextlib.nested(*files) as entered_files:
        for lines in zip(*entered_files):
            yield lines


class MultiFileWriter(object):

    def __init__(self, *file_paths):
        self.file_paths = file_paths

    def __enter__(self):
        self.files = [open(fp, 'w') for fp in self.file_paths]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file in self.files:
            file.__exit__(exc_type, exc_val, exc_tb)

    def write(self, lines):
        assert len(lines) == len(self.files)
        for f, line in zip(self.files, lines):
            f.write(line)


def open_or_create(path, *args, **kwargs):
    """Open a file or create it, if it does not exist.

    Args:
        path (str): path to file
        gz (bool): whether to use GZIP or not. Defaults to False.

    Returns:
        file object
    """
    gz = kwargs.pop('gz', False)

    open_file = gzip.open if gz else open

    if not os.path.isfile(path):
        with open_file(path, 'w'):
            pass  # create file
    return open_file(path, *args, **kwargs)


class Process(object):
    def __init__(self, cmd, cwd=None):
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, cwd=cwd)

    def read(self, timeout=float('inf')):
        def enqueue_output(out, queue):
            for c in iter(lambda: out.read(1), ''):
                queue.put(c)

        q = Queue()
        t = Thread(target=enqueue_output, args=(self._proc.stdout, q))
        t.daemon = True  # thread dies with the program
        t.start()

        last_yield_time = time.time()
        while True:
            try:
                yield q.get(timeout=0.001)
                last_yield_time = time.time()
            except Empty:
                # if 1 millisecond passes without new item on queue...
                if not self.alive:
                    # break if process has died
                    break
                if time.time() - last_yield_time > timeout:
                    # break if time is up
                    break

    def read_lines(self, timeout=float('inf')):
        chars = []
        for c in self.read(timeout):
            chars.append(c)
            if c == '\n':
                yield ''.join(chars[:-1])
                chars = []

    @property
    def pid(self):
        return self._proc.pid

    @property
    def alive(self):
        code = self._proc.poll()
        return code is None

    def terminate(self):
        return self._proc.terminate()

    def wait(self):
        return self._proc.wait()


def shell(cmd, cwd=None, verbose=False, debug=False):
    """Execute a command just like you would at the command line.

    Attempts to print output from the command with as little buffering as possible.
    http://stackoverflow.com/questions/18421757/live-output-from-subprocess-command

    Args:
        cmd (str): command to execute, just as you would enter at the command line
        cwd (str): current working directory to execute the command
        verbose (bool): whether to print out the results of the command
        debug (bool): if True, command is not actually executed. Typically used with verbose=True.

    Returns:
        all output from the command
    """
    if verbose:
        print(cmd)

    if debug:
        return

    output = []
    process = Process(cmd, cwd)

    for c in process.read():
        output.append(c)
        if verbose:
            sys.stdout.write(c)
            sys.stdout.flush()

    status = process.wait()
    if status != 0:
        raise RuntimeError('Error, exit code: {}'.format(status))

    # TODO: make sure we get all output
    return ''.join(output)


def local_bash(command, capture=False):
    """Just like fabric.api.local, but with shell='/bin/bash'."""
    return local(command, capture, shell='/bin/bash')


class JSONPicklable(object, metaclass=ABCMeta):
    """Uses jsonpickle to convert any picklable object to and from JSON."""

    @abstractmethod
    def __getstate__(self):
        """Return a variable with enough information to reconstruct the object."""
        pass

    @abstractmethod
    def __setstate__(self, state):
        """Use the variable from __getstate__ to restore the object.

        Note that pickle created this object without calling __init__.

        So, a common strategy is to manually call self.__init__(...) inside this function, using the information
        provided by `state`.
        """
        pass

    def to_json_str(self):
        return jsonpickle.encode(self)

    @classmethod
    def from_json_str(self, s):
        return jsonpickle.decode(s)

    def to_json(self):
        """Use jsonpickle to convert this object to JSON."""
        s = self.to_json_str()
        d = json.loads(s)  # convert str to dict
        return d

    @classmethod
    def from_json(cls, d):
        """Use jsonpickle to convert JSON into an object."""
        s = json.dumps(d)
        obj = cls.from_json_str(s)
        return obj

    def to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f)

    @classmethod
    def from_file(self, path):
        with open(path, 'r') as f:
            d = json.load(f)
        return JSONPicklable.from_json(d)


class InitPicklable(object):
    def __new__(cls, *args, **kwargs):
        obj = super(InitPicklable, cls).__new__(cls)
        obj.__initargs = args, kwargs
        return obj

    def __getstate__(self):
        return self.__initargs

    def __setstate__(self, state):
        args, kwargs = state
        self.__init__(*args, **kwargs)


def sub_dirs(root_dir):
    """Return a list of all sub-directory paths.

    Example:
        >> root_dir = '/Users/Kelvin/data'
        >> sub_dirs(root_dir)
        ['/Users/Kelvin/data/a', '/Users/Kelvin/data/b']
    """
    dir_paths = []
    for path in os.listdir(root_dir):
        full_path = join(root_dir, path)
        if os.path.isdir(full_path):
            dir_paths.append(full_path)
    return dir_paths


class IntegerDirectories(Mapping):
    """Keep track of directories with names of the form "{integer}_{something}" or just "{integer}".

    Used for organizing experiment directories.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        makedirs(root_dir)

    @property
    def _ints_to_paths(self):
        ints_to_paths = {}
        for p in sub_dirs(self.root_dir):
            name = os.path.basename(p)
            try:
                i = int(name.split('_')[0])
                if i in ints_to_paths:
                    raise IOError("Multiple directories with the same integer prefix: {} and {}".format(
                        ints_to_paths[i], p))
                ints_to_paths[i] = p
            except ValueError:
                # the first element was not an integer
                pass

        # put into an ordered dict
        ordered = OrderedDict()
        for i in sorted(ints_to_paths):
            ordered[i] = ints_to_paths[i]
        return ordered

    def __len__(self):
        return len(self._ints_to_paths)

    @property
    def largest_int(self):
        """Largest int among the integer directories."""
        if len(self._ints_to_paths) == 0:
            return None
        return max(self._ints_to_paths)

    def new_dir(self, name=None):
        """Create a new directory and return its path."""
        if self.largest_int is None:
            idx = 0
        else:
            idx = self.largest_int + 1

        path = join(self.root_dir, str(idx))

        if name:
            path = '{}_{}'.format(path, name)  # add name as suffix

        makedirs(path)
        return path

    def __getitem__(self, i):
        """Get the path to experiment i.

        Raises:
            KeyError, if experiment folder does not exist.
        """
        if i not in self._ints_to_paths:
            raise KeyError("Experiment #{} not found".format(i))
        return self._ints_to_paths[i]

    def __iter__(self):
        return iter(self._ints_to_paths)


def rsync(src_path, dest_path, src_host=None, dest_host=None, delete=False):
    """Sync a file/directory from one machine to another machine.

    Args:
        src_path (str): a file or directory on the source machine.
        dest_path (str): the corresponding file or directory on the target machine.
        src_host (str): the address of the source machine. Default is local machine.
        dest_host (str): the address of the target machine. Default is local machine.
        delete (bool): default is False. If True, deletes any extraneous files at the destination not
            present at the source!

    Options used:
        -r: recurse into directories
        -l: copy symlinks as symlinks
        -v: verbose
        -z: compress files during transfer
        -t: preserve times (needed for rsync to recognize that files haven't changed since last update!)
        --delete: delete any extraneous files at the destination
        --progress: show progress
    """
    if os.path.isdir(src_path):
        if src_path[:-1] != '/':
            src_path += '/'  # add missing trailing slash

    def format_address(host, path):
        if host is None:
            return path
        else:
            return '{}:{}'.format(host, path)

    cmds = ["rsync", "-rlvzt", "--progress"]

    if delete:
        cmds.append('--delete')

    cmds.append(format_address(src_host, src_path))
    cmds.append(format_address(dest_host, dest_path))
    cmd = ' '.join(cmds)
    local(cmd)


class Tmux(object):
    def __init__(self, name, cwd=None):
        """Create a tmux session.

        Args:
            name (str): name of the new session
            cwd (str): initial directory of the session

        Options used:
            -d: do not attach to the new session
            -s: specify a name for the session
        """
        self.name = name

        with settings(hide('warnings'), warn_only=True):
            result = local("tmux new -d -s {}".format(name))  # start tmux session

        if result.failed:
            raise TmuxSessionExists()

        if cwd is None:
            cwd = os.getcwd()

        # move to current directory
        self.run("cd {}".format(cwd))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, command):
        """Run command in tmux session.

        Assume that the session has only one window.

        Args:
            command (str)
        """
        local('tmux send -t {} "{}" Enter'.format(self.name, command))

    def close(self):
        local("tmux kill-session -t {}".format(self.name))


class TmuxSessionExists(Exception):
    pass


def tunnel(local_port, host, target, target_port, tmux_name, autossh_port=20000):
    """Make a port on a target machine appear as if it is a port on our local machine.

    Uses autossh to keep the tunnel open even with interruptions.
    Runs autossh in a new tmux session, so that it can be monitored.

    Args:
        local_port (int): a port on this machine, e.g. 18888
        host (str): the machine that will be used to create the SSH tunnel, e.g. `kgu@jamie.stanford.edu` or just `jamie`
            if we have that alias configured in ~/.ssh/config.
        target (str): the address of the target machine, e.g. `kgu@john11.stanford.edu` or just `john11`. The address
            should be RELATIVE to the host machine.
        target_port (int): port on the target machine, e.g. 8888
        tmux_name (str): name of the tmux session that will be running the autossh command.
        autossh_port (int): local port used by autossh to monitor the connection. Cannot be used by more than one
            autossh process at a time!
    """
    command = "autossh -M {} -N -n -T -L {}:{}:{} {}".format(autossh_port, local_port, target, target_port, host)
    tmux = Tmux(tmux_name)
    tmux.run(command)


class Workspace(object):
    """Manage paths underneath a top-level root directory.

    Paths are registered with this Workspace. An IOError is thrown if the path has already been registered before.
    """
    def __init__(self, root):
        """Create a Workspace.

        Args:
            root (str): absolute path of the top-level directory.
        """
        self._root = root
        makedirs(root)
        self._paths = set()

    @property
    def root(self):
        return self._root

    def _add(self, name, relative_path):
        """Register a path.

        Args:
            name (str): short name to reference the path
            relative_path (str): a relative path, relative to the workspace root.

        Returns:
            self
        """
        full_path = join(self._root, relative_path)
        if hasattr(self, name):
            raise IOError('Name already registered: {}'.format(name))
        if full_path in self._paths:
            raise IOError('Path already registered: {}'.format(relative_path))
        setattr(self, name, full_path)

    def add_dir(self, name, relative_path):
        self._add(name, relative_path)
        makedirs(getattr(self, name))

    def add_file(self, name, relative_path):
        self._add(name, relative_path)