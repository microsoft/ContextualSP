from collections import Mapping
from os.path import join
import logging

from git import Repo, exc as git_exc

from gtd.io import IntegerDirectories, Workspace
from gtd.log import SyncedMetadata
from gtd.utils import Config, cached_property


class ExperimentWorkspace(Workspace):
    def __init__(self, root):
        super(ExperimentWorkspace, self).__init__(root)
        for attr in ['config', 'metadata']:
            self.add_file(attr, '{}.txt'.format(attr))
        self.add_dir('checkpoints', 'checkpoints')


class Experiment(object):
    def __init__(self, config, save_dir):
        """Create experiment.

        Args:
            config (Config)
            save_dir (str)
        """
        self._config = config
        self._workspace = ExperimentWorkspace(save_dir)

    @property
    def config(self):
        return self._config

    @property
    def workspace(self):
        return self._workspace

    @cached_property
    def metadata(self):
        return SyncedMetadata(self.workspace.metadata)

    def record_commit(self, src_dir):
        try:
            repo = Repo(src_dir)

            if 'dirty_repo' in self.metadata or 'commit' in self.metadata:
                raise RuntimeError('A commit has already been recorded.')

            self.metadata['dirty_repo'] = repo.is_dirty()
            self.metadata['commit'] = repo.head.object.hexsha.encode('utf-8')
        except git_exc.InvalidGitRepositoryError as e:
            # Maybe not a git repo e.g., running on CodaLab
            self.metadata['dirty_repo'] = False
            self.metadata['commit'] = 'NONE'

    def match_commit(self, src_dir):
        """Check that the current commit matches the recorded commit for this experiment.

        Raises an error if commits don't match, or if there is dirty state.

        Args:
            src_dir (str): path to the Git repository
        """
        if self.metadata['dirty_repo']:
            raise EnvironmentError('Working directory was dirty when commit was recorded.')

        repo = Repo(src_dir)
        if repo.is_dirty():
            raise EnvironmentError('Current working directory is dirty.')

        current_commit = repo.head.object.hexsha.encode('utf-8')
        exp_commit = self.metadata['commit']
        if current_commit != exp_commit:
            raise EnvironmentError("Commits don't match.\nCurrent: {}\nRecorded: {}".format(current_commit, exp_commit))


class TFExperiment(Experiment):
    def __init__(self, config, save_dir):
        super(TFExperiment, self).__init__(config, save_dir)
        self._workspace.add_dir('tensorboard', 'tensorboard')

    @cached_property
    def saver(self):
        from gtd.ml.utils import Saver
        return Saver(self.workspace.checkpoints, keep_checkpoint_every_n_hours=5)

    @cached_property
    def tb_logger(self):
        from gtd.ml.utils import TensorBoardLogger
        return TensorBoardLogger(self.workspace.tensorboard)


class Experiments(Mapping):
    """A map from integers to Experiments."""

    def __init__(self, root_dir, src_dir, experiment_factory, default_config_path, check_commit=True):
        """Create Experiments object.

        Args:
            root_dir (str): directory where all experiment data will be stored
            src_dir (str): a Git repository path (used to check commits)
            experiment_factory (Callable[[Config, str], Experiment]): a Callable, which takes a Config and a save_dir
                as arguments, and creates a new Experiment.
            default_config_path (str): path to a default config, to be used when no config is specified
            check_commit (bool): if True, checks that current working directory is on same commit as when the experiment
                was originally created.
        """
        self._int_dirs = IntegerDirectories(root_dir)
        self._src_dir = src_dir
        self._exp_factory = experiment_factory
        self._check_commit = check_commit
        self._default_config_path = default_config_path

    def _config_path(self, save_dir):
        return join(save_dir, 'config.txt')

    def __getitem__(self, i):
        """Reload an existing Experiment."""
        save_dir = self._int_dirs[i]
        config = Config.from_file(self._config_path(save_dir))
        exp = self._exp_factory(config, save_dir)
        if self._check_commit:
            exp.match_commit(self._src_dir)

        logging.info('Reloaded experiment #{}'.format(i))
        return exp

    def new(self, config=None, name=None):
        """Create a new Experiment."""
        if config is None:
            config = Config.from_file(self._default_config_path)

        save_dir = self._int_dirs.new_dir(name=name)
        cfg_path = self._config_path(save_dir)
        config.to_file(cfg_path)  # save the config
        exp = self._exp_factory(config, save_dir)
        exp.record_commit(self._src_dir)

        logging.info('New experiment created at: {}'.format(exp.workspace.root))
        logging.info('Experiment configuration:\n{}'.format(config))
        return exp

    def __iter__(self):
        return iter(self._int_dirs)

    def __len__(self):
        return len(self._int_dirs)

    def paths(self):
        return list(self._int_dirs.values())
