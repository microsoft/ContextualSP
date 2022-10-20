import abc
import os
import pickle
import time
import sys
from dependency.data_directory import DataDirectory
from prettytable import PrettyTable
from strongsup.results.entry import Entry
from strongsup.results.result_value import ResultValue


class Tracker(object, metaclass=abc.ABCMeta):
    """Tracks a set of a results. In charge of maintaining up to date
    results for each Entry.

    Args:
        name (string): name of this tracker
        parent (Tracker): a tracker or None
    """
    def __init__(self, name, parent=None):
        self._name = name
        self._parent = parent
        self._load()  # Load sub-trackers or entries

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def merge(self, other):
        """Merges two trackers together.

        Args:
            other (Tracker): the other tracker
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _load(self):
        """Loads the Tracker object from somewhere, generally from file"""
        raise NotImplementedError()

    def _match(self, x, filters=None):
        """Returns true iff x's name substring matches
        one of the filters OR filters is None

        Args:
            x: something with a name property
            filters (list[string]): the filters

        Returns:
            bool: if there's a match
        """
        if not filters:
            return True
        return any(
            [x.name.find(filt) != -1 for filt in filters])

    def __str__(self):
        return "Tracker({})".format(self.name)
    __repr__ = __str__


class TopLevelTracker(Tracker):
    def __init__(self, name, parent=None):
        super(TopLevelTracker, self).__init__(name, parent)

    def entries(self, dataset_filters=None, experiment_type_filters=None):
        """Returns all entries that substring match strings in
        experiment_type_filters

        Args:
            dataset_filters (list[string]): the substrings to match datasets
                on, None matches everything.
            experiment_type_filters (list[string]): the substrings to match,
                None matches everything

        Returns:
            list[Entry]: all matching entries
        """
        filter_fn = lambda x: self._match(x, dataset_filters)
        trackers = list(filter(filter_fn, iter(self._trackers.values())))
        entries = []
        for tracker in trackers:
            entries.extend(tracker.entries(experiment_type_filters))
        return entries

    def add_result(self, dataset, experiment_type, seed, result_value):
        """Adds a result associated with this dataset, experiment_type and
        seed

        Args:
            dataset (string)
            experiment_type (ExperimentType)
            seed (int)
            result_value (ResultValue)
        """
        tracker = self._trackers.setdefault(
                dataset, LeafTracker(dataset, self))

        tracker.add_result(experiment_type, seed, result_value)

    def _update_result(self, dataset, experiment_type, seed, result_value):
        """Should not get called externally."""
        tracker = self._trackers.setdefault(
                dataset, LeafTracker(dataset, self))

        tracker._update_result(experiment_type, seed, result_value)

    def merge(self, other):
        for dataset, tracker in other._trackers.items():
            self._trackers.setdefault(
                    dataset, LeafTracker(dataset)).merge(tracker)
            self._running_jobs.extend(other._running_jobs)
            self._complete_jobs.extend(other._complete_jobs)

    def refresh_result(self, dataset, experiment_type, seed, path):
        """Re-fetches the result at this path. Marks the experiment
        as in-progress again.

        Args:
            dataset (string): the dataset of the result
            experiment_type (ExperimentType): the experiment type of result
            seed (int): seed of result
            path (string): filesystem path of experiment directory
        """
        success, result, access = self._fetch_result(path, None)
        assert success
        self._update_result(dataset, experiment_type, seed, result)
        self.register_result(dataset, experiment_type, seed, path)

    def register_result(self, dataset, experiment_type, seed, path):
        """Registers a result to be loaded next time.

        Args:
            dataset (string): the dataset of the result
            experiment_type (ExperimentType): the experiment type of result
            seed (int): seed of result
            path (string): filesystem path of experiment directory
        """
        self._running_jobs.append(
                JobMetadata(dataset, experiment_type, seed, path))

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        """Writes _trackers and _running_jobs to file on clean exit"""
        # Clean exit
        if ex_type is None and ex_value is None and traceback is None:
            with open(self.filename, 'w+') as f:
                pickle.dump((self._trackers, self._running_jobs,
                    self._complete_jobs), f)

    def _load(self):
        if not os.path.exists(self.filename):
            self._trackers = {}  # name (string) --> Tracker
            self._running_jobs = []  # List of jobs to fetch from
            self._complete_jobs = []  # List of complete jobs
            return

        with open(self.filename, 'r') as f:
            self._trackers, self._running_jobs, self._complete_jobs = pickle.loads(f.read())

        self._refresh_results()
        if len(self._running_jobs) != 0:
            warn("There are still running jobs or dead jobs: {}".format(self._running_jobs))
            warn("You should probably not merge this tracker")

    def _refresh_results(self):
        """Fetches all of the running jobs"""
        to_remove = []
        for index, job in enumerate(self._running_jobs):
            accessed, result, timestamp = self._fetch_result(
                    job.path, job.last_accessed)
            if not accessed:
                if timestamp == 0:
                    to_remove.append(index)
            else:
                job.last_accessed = timestamp
                self._update_result(
                        job.dataset, job.experiment_type, job.seed, result)

        # Remove jobs that are dead
        for index in reversed(to_remove):
            job = self._running_jobs.pop(index)
            job.last_accessed = None
            self._complete_jobs.append(job)

    def _fetch_result(self, exp_path, last_accessed):
        """Fetches the most up to date results if last_accessed is earlier
        than the events file timestamp.

        Args:
            exp_path (string): the path to experiment directory
            last_accessed (float): the time in seconds since file was last
                accessed, None for never

        Returns:
            bool: if the result was accessed again
            ResultValue: the new result if accessed, otherwise None
            float: the new last accessed time
        """
        from tensorflow.python.summary import event_accumulator as ea
        KEYS = [
            'VALID_denoAcc_silent_1utts_1',
            'VALID_denoAcc_silent_2utts_1',
            'VALID_denoAcc_silent_3utts_1',
            'VALID_denoAcc_silent_4utts_1',
            'VALID_denoAcc_silent_5utts_1',
            'FINAL_denoAcc_silent_1utts_1',
            'FINAL_denoAcc_silent_2utts_1',
            'FINAL_denoAcc_silent_3utts_1',
            'FINAL_denoAcc_silent_4utts_1',
            'FINAL_denoAcc_silent_5utts_1',
        ]

        events_file = exp_path + "/tensorboard"

        # Last accessed is up to date
        if (last_accessed is not None and
                os.path.getmtime(exp_path) <= last_accessed):
            return False, None, 0

        last_accessed = time.time()
        print('Reading from', events_file, \
                '(could take a while ...)', file=sys.stderr)
        acc = ea.EventAccumulator(events_file, size_guidance={ea.SCALARS: 0})
        acc.Reload()
        available_keys = set(acc.Tags()['scalars'])
        values = []
        for key in KEYS:
            # Key not available to load yet
            if key not in available_keys:
                warn("No results found for {}".format(exp_path))
                print("Perhaps your job has died?")
                return False, None, None
            if key in available_keys:
                values.append([scalar.value for scalar in acc.Scalars(key)])
        values = list(zip(*values))
        if len(values) == 0:
            assert False

        best_index, best_value = max(
            [(i, sum(value)) for i, value in enumerate(values)],
            key=lambda x: x[1])
        return True, ResultValue(list(values[best_index][:5]),
                list(values[best_index][5:])), last_accessed

    @property
    def datasets(self):
        return iter(self._trackers.keys())

    @property
    def filename(self):
        return DataDirectory.results + "/" + self.name + ".trk"

    def __eq__(self, other):
        if not isinstance(other, TopLevelTracker):
            return False

        return self._trackers == other._trackers and self.name == other.name


class LeafTracker(Tracker):
    """A Tracker typically in charge of a single Dataset

    Args:
        name (string): the name (typically the dataset)
        parent (Tracker): A TopLevelTracker
    """
    def __init__(self, name, parent=None):
        super(LeafTracker, self).__init__(name, parent)
        self._entries = {}  # ExperimentType --> Entry

    def entries(self, experiment_type_filters=None):
        """Returns all entries that substring match strings in
        experiment_type_filters

        Args:
            experiment_type_filters (list[string]): the substrings to match,
                None matches everything

        Returns:
            list[Entry]: all matching entries
        """
        filter_fn = lambda entry: self._match(entry, experiment_type_filters)
        entries = list(filter(filter_fn, iter(self._entries.values())))
        return entries

    def add_result(self, experiment_type, seed, result_value):
        """Adds the result value associated with this experiment type and
        seed to the Tracker.

        Args:
            experiment_type (ExperimentType)
            seed (int)
            result_value (ResultValue): the result
        """
        entry = self._entries.setdefault(experiment_type,
                                         Entry(experiment_type))
        entry.add_seed(seed, result_value)

    def _update_result(self, experiment_type, seed, result_value):
        """Should not get called externally."""
        entry = self._entries.setdefault(experiment_type,
                                         Entry(experiment_type))
        entry.update_seed(seed, result_value)

    def merge(self, other):
        for (experiment_type, entry) in other._entries.items():
            if experiment_type not in self._entries:
                self._entries[experiment_type] = entry
            else:
                for seed in entry.seeds:
                    if self._entries[experiment_type].contains_seed(seed):
                        best_result = max(
                            [self._entries[experiment_type].get_value(seed),
                             entry.get_value(seed)])
                        self._entries[experiment_type].update_seed(
                                seed, best_result)
                    else:
                        self._entries[experiment_type].add_seed(
                                seed, entry.get_value(seed))

    def _load(self):
        # TopLevelTrackers are responsible for loading this
        return

    def __eq__(self, other):
        if not isinstance(other, LeafTracker):
            return False

        return self._entries == other._entries and self.name == other.name


class JobMetadata(object):
    """Light-weight struct for maintaining info about running jobs"""
    def __init__(self, dataset, experiment_type, seed, path, last_accessed=None):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.seed = seed
        self.path = path
        self.last_accessed = last_accessed

    def __getstate__(self):
        """Sets the last_accessed to None when pickling, to be platform
        independent. The epoch in OS X is different than the epoch in Linux
        distros"""
        return (self.dataset, self.experiment_type, self.seed, self.path, self.last_accessed)

    def __setstate__(self, state):
        dataset, experiment_type, seed, path, last_accessed = state
        self.__init__(dataset, experiment_type, seed, path, last_accessed)

    def __str__(self):
        return "JobMetadata({}, {}, {}, {}, {})".format(
                self.experiment_type, self.dataset, self.seed, self.path, self.last_accessed)
    __repr__ = __str__

def warn(msg):
    print("=" * 10 + "WARNING: " + msg + "=" * 10)
