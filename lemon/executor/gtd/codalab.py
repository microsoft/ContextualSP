"""Tools for working with CodaLab."""
import pickle as pickle
import json
import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager

import matplotlib.image as mpimg
from gtd.io import shell

__author__ = 'kelvinguu'


# need to be specified by user
worksheet = None
site = None


def get_uuids():
    """List all bundle UUIDs in the worksheet."""
    result = shell('cl ls -w {} -u'.format(worksheet))
    uuids = result.split('\n')
    uuids = uuids[1:-1]  # trim non uuids
    return uuids


@contextmanager
def open_file(uuid, path):
     """Get the raw file content within a particular bundle at a particular path.

     Path have no leading slash.
     """
     # create temporary file just so we can get an unused file path
     f = tempfile.NamedTemporaryFile()
     f.close()  # close and delete right away
     fname = f.name

     # download file to temporary path
     cmd ='cl down -o {} -w {} {}/{}'.format(fname, worksheet, uuid, path)
     try:
        shell(cmd)
     except RuntimeError:
         try:
            os.remove(fname)  # if file exists, remove it
         except OSError:
             pass
         raise IOError('Failed to open file {}/{}'.format(uuid, path))

     f = open(fname)
     yield f
     f.close()
     os.remove(fname)  # delete temp file


class Bundle(object):
    def __init__(self, uuid):
        self.uuid = uuid

    def __getattr__(self, item):
        """
        Load attributes: history, meta on demand
        """
        if item == 'history':
            try:
                with open_file(self.uuid, 'history.cpkl') as f:
                    value = pickle.load(f)
            except IOError:
                value = {}

        elif item == 'meta':
            try:
                with open_file(self.uuid, 'meta.json') as f:
                    value = json.load(f)
            except IOError:
                value = {}

            # load codalab info
            fields = ('uuid', 'name', 'bundle_type', 'state', 'time', 'remote')
            cmd = 'cl info -w {} -f {} {}'.format(worksheet, ','.join(fields), self.uuid)
            result = shell(cmd)
            info = dict(list(zip(fields, result.split())))
            value.update(info)

        elif item in ('stderr', 'stdout'):
            with open_file(self.uuid, item) as f:
                value = f.read()

        else:
            raise AttributeError(item)

        self.__setattr__(item, value)
        return value

    def __repr__(self):
        return self.uuid

    def load_img(self, img_path):
        """
        Return an image object that can be immediately plotted with matplotlib
        """
        with open_file(self.uuid, img_path) as f:
            return mpimg.imread(f)


def download_logs(bundle, log_dir):
    if bundle.meta['bundle_type'] != 'run' or bundle.meta['state'] == 'queued':
        print('Skipped {}\n'.format(bundle.uuid))
        return

    if isinstance(bundle, str):
        bundle = Bundle(bundle)

    uuid = bundle.uuid
    name = bundle.meta['name']
    log_path = os.path.join(log_dir, '{}_{}'.format(name, uuid))

    cmd ='cl down -o {} -w {} {}/logs'.format(log_path, worksheet, uuid)

    print(uuid)
    try:
        shell(cmd, verbose=True)
    except RuntimeError:
        print('Failed to download', bundle.uuid)
    print()


def report(render, uuids=None, reverse=True, limit=None):
    if uuids is None:
        uuids = get_uuids()

    if reverse:
        uuids = uuids[::-1]

    if limit is not None:
        uuids = uuids[:limit]

    for uuid in uuids:
        bundle = Bundle(uuid)
        try:
            render(bundle)
        except Exception:
            print('Failed to render', bundle.uuid)


def monitor_jobs(logdir, uuids=None, reverse=True, limit=None):
    if os.path.exists(logdir):
        delete = input('Overwrite existing logdir? ({})'.format(logdir))
        if delete == 'y':
            shutil.rmtree(logdir)
            os.makedirs(logdir)
    else:
        os.makedirs(logdir)
        print('Using logdir:', logdir)

    report(lambda bd: download_logs(bd, logdir), uuids, reverse, limit)


def tensorboard(logdir):
    print('Run this in bash:')
    shell('tensorboard --logdir={}'.format(logdir), verbose=True, debug=True)
    print('\nGo to TensorBoard: http://localhost:6006/')


def add_to_sys_path(path):
    """Add a path to the system PATH."""
    sys.path.insert(0, path)


def configure_matplotlib():
    """Set Matplotlib backend to 'Agg', which is necessary on CodaLab docker image."""
    import warnings
    import matplotlib
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        matplotlib.use('Agg')  # needed when running from server


def in_codalab():
    """Check if we are running inside CodaLab Docker container or not."""
    # TODO: below is a total hack. If the OS is not a Mac, we assume we're on CodaLab.
    return platform.system() != 'Darwin'


def upload(full_path, bundle_name=None, excludes='*.ipynb .git .ipynb_checkpoints .ignore'):
    """
    Upload a file or directory to the codalab worksheet
    Args:
        full_path: Path + filename of file to upload
        bundle_name: Name to upload file/directory as. I
    """
    directory, filename = os.path.split(full_path)
    if bundle_name is None:
        bundle_name = filename
    shell('cl up -n {} -w {} {} -x {}'.format(bundle_name, worksheet, full_path, excludes), verbose=True)


def launch_job(job_name, cmd,
               dependencies=tuple(),
               queue='john', image='kelvinguu/gtd:1.0',
               memory=None, cpus='5',
               network=False,
               debug=False, tail=False):
    """Launch a job on CodaLab (optionally upload code that the job depends on).

    Args:
        job_name: name of the job
        cmd: command to execute
        dependencies: list of other bundles that we depend on
        debug: if True, prints SSH commands, but does not execute them
        tail: show the streaming output returned by CodaLab once it launches the job
    """
    print('Remember to set up SSH tunnel and LOG IN through the command line before calling this.')
    options = '-v -n {} -w {} --request-queue {} --request-docker-image {} --request-cpus {}'.format(
        job_name, worksheet, queue, image, cpus)

    if memory:
        options += ' --request-memory {}'.format(memory)
    if network:
        options += ' --request-network'

    dep_str = ' '.join(['{0}:{0}'.format(dep) for dep in dependencies])
    full_cmd = "cl run {} {} '{}'".format(options, dep_str, cmd)
    if tail:
        full_cmd += ' -t'
    shell(full_cmd, verbose=True, debug=debug)


if in_codalab():
    configure_matplotlib()
