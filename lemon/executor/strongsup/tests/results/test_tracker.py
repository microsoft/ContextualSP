import copy
import os
import pytest
import shutil
from strongsup.results.tracker import LeafTracker, TopLevelTracker
from strongsup.results.entry import Entry, ExperimentType
from strongsup.results.result_value import ResultValue


class TestTracker(object):
    @pytest.fixture
    def filters(self):
        return ["match", "other"]

    @pytest.fixture
    def result(self):
        return ResultValue([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])

    @pytest.fixture
    def experiment_types(self):
        match = ExperimentType(["should-match", "config"], "base")
        also_match = ExperimentType(["config", "other"], "base")
        no_match = ExperimentType(["filter"], "base")
        other = ExperimentType(["config"], "base")
        return [match, also_match, no_match, other]

    def _entries_equal(self, entries, expected_entries):
        """Returns if two lists of entries contain equal entries"""
        return sorted(entries, key=lambda entry: str(entry)) == sorted(
                expected_entries, key=lambda entry: str(entry))

class TestLeafTracker(TestTracker):
    def test_merge(self, tracker, result, experiment_types):
        tracker.add_result(experiment_types[0], 0, result)
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries = [expected_entry]

        # Test merge of two seeds
        other = LeafTracker("other")
        other.add_result(experiment_types[0], 1, result * 2)
        tracker.merge(other)
        expected_entry.add_seed(1, result * 2)
        assert tracker.entries() == expected_entries

        # Test merge on two Entries
        other = LeafTracker("other")
        other.add_result(experiment_types[1], 0, result)
        tracker.merge(other)
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        self._entries_equal(tracker.entries(), expected_entries)

        # Test merge updates to best seed
        other = LeafTracker("other")
        other.add_result(experiment_types[1], 0, result * 2)
        tracker.merge(other)
        expected_entry.update_seed(0, result * 2)
        self._entries_equal(tracker.entries(), expected_entries)

    def test_entries(self, tracker, result, experiment_types, filters):
        # Make sure is empty upon construction
        entries = tracker.entries()
        assert len(entries) == 0

        # Test filtering
        # No matches
        tracker.add_result(experiment_types[2], 0, result)
        entries = tracker.entries(filters)
        assert len(entries) == 0

        # Matches both
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entry.add_seed(1, result * 2)
        expected_entries = [expected_entry]
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)

        tracker.add_result(experiment_types[1], 0, result)
        tracker.add_result(experiment_types[0], 0, result)
        tracker.add_result(experiment_types[0], 1, result * 2)
        entries = tracker.entries(filters)
        assert self._entries_equal(entries, expected_entries)

    def test_add_entry(self, tracker, result, experiment_types):
        # Test adding a single entry
        tracker.add_result(experiment_types[0], 0, result)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries = [expected_entry]
        assert entries == expected_entries

        # Test adding a duplicate entry
        with pytest.raises(ValueError) as excinfo:
            tracker.add_result(experiment_types[0], 0, result*2)
        assert excinfo.match("Seed 0 already in Entry")

        # Test adding multiple seeds
        tracker.add_result(experiment_types[0], 1, result * 2)
        entries = tracker.entries()
        expected_entry.add_seed(1, result * 2)
        assert entries == expected_entries

        # Test adding multiple entries
        tracker.add_result(experiment_types[1], 0, result)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        assert self._entries_equal(entries, expected_entries)

    @pytest.fixture
    def tracker(self):
        tracker = LeafTracker("name")
        return tracker

class TestTopLevelTracker(TestTracker):
    def test_register_result(self, result, experiment_types,
                             datasets, teardown_tensorboard):
        # Clear out previous tracker
        if os.path.exists(TopLevelTracker("test_tracker").filename):
            os.remove(TopLevelTracker("test_tracker").filename)

        # Register result
        with TopLevelTracker("test_tracker") as tracker:
            tracker.register_result(
                    datasets[0], experiment_types[0], 0, ".")
            assert tracker.entries() == []

        # Make sure that result gets loaded
        with TopLevelTracker("test_tracker") as tracker:
            expected_entry = Entry(experiment_types[0])
            expected_entry.add_seed(0, ResultValue([0.0] * 5, [0.0] * 5))
            expected_entries = [expected_entry]
            assert tracker.entries() == expected_entries

        # Update result
        shutil.move("tensorboard", "backup")
        shutil.move("other_tensorboard", "tensorboard")

        result = ResultValue(
                [0.9396985173225403, 0.839195966720581, 0.6281406879425049,
                 0.49246230721473694, 0.3467336595058441],
                [0.9012500047683716, 0.8087499737739563, 0.6499999761581421,
                 0.4737499952316284, 0.3449999988079071])

        # Make sure result gets loaded again
        with TopLevelTracker("test_tracker") as tracker:
            entries = tracker.entries()
            expected_entry.update_seed(0, result)
            assert tracker.entries() == expected_entries

        # Make sure result doesn't change
        with TopLevelTracker("test_tracker") as tracker:
            entries = tracker.entries()
            expected_entry.update_seed(0, result)
            assert tracker.entries() == expected_entries

    @pytest.fixture
    def teardown_tensorboard(self):
        yield
        # Restore files to correct place
        shutil.move("tensorboard", "other_tensorboard")
        shutil.move("backup", "tensorboard")

    def test_persist(self, result, experiment_types, datasets):
        # Clear out previous tracker
        if os.path.exists(TopLevelTracker("test_tracker").filename):
            os.remove(TopLevelTracker("test_tracker").filename)

        # Test reloading empty tracker
        with TopLevelTracker("test_tracker") as tracker:
            clone = copy.deepcopy(tracker)

        assert clone == TopLevelTracker("test_tracker")

        # Test reloading non-empty tracker
        with TopLevelTracker("test_tracker") as tracker:
            # Multiple datasets
            tracker.add_result(datasets[0], experiment_types[0], 0, result)
            tracker.add_result(datasets[1], experiment_types[0], 0, result * 2)
            tracker.add_result(datasets[2], experiment_types[0], 0, result * 3)

            # Multiple entries per dataset
            tracker.add_result(datasets[0], experiment_types[1], 0, result)

            # Multiple seeds per entry
            tracker.add_result(datasets[0], experiment_types[1], 1, result * 2)

            clone = copy.deepcopy(tracker)
        assert clone == TopLevelTracker("test_tracker")

    def test_merge(self, tracker, result, experiment_types, datasets):
        # Merge two empty trackers
        other = TopLevelTracker("other")
        tracker.merge(other)
        assert tracker.entries() == []

        # Merge non-empty into empty
        other.add_result(datasets[0], experiment_types[0], 0, result)
        tracker.merge(other)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries = [expected_entry]
        assert self._entries_equal(entries, expected_entries)

        # Merge empty into non-empty
        other = TopLevelTracker("other")
        tracker.merge(other)
        entries = tracker.entries()
        assert self._entries_equal(entries, expected_entries)

        # Merge two different datasets
        other.add_result(datasets[1], experiment_types[0], 0, result)
        tracker.merge(other)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        assert self._entries_equal(entries, expected_entries)

        # Merge on same dataset
        other = TopLevelTracker("other")
        other.add_result(datasets[0], experiment_types[1], 0, result)
        tracker.merge(other)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        assert self._entries_equal(entries, expected_entries)

        # Merge on same Entry
        other = TopLevelTracker("other")
        other.add_result(datasets[0], experiment_types[0], 1, result)
        tracker.merge(other)
        entries = tracker.entries()
        expected_entries[0].add_seed(1, result)
        assert self._entries_equal(entries, expected_entries)

        # Merge on same seed
        other = TopLevelTracker("other")
        other.add_result(datasets[0], experiment_types[0], 1, result * 2)
        tracker.merge(other)
        expected_entries[0].update_seed(1, result * 2)
        assert self._entries_equal(entries, expected_entries)

    def test_entries(self, tracker, result, experiment_types,
                     filters, datasets):
        # Empty at beginning
        assert tracker.entries() == []

        # Filter on experiment type
        tracker.add_result(datasets[0], experiment_types[2], 0, result)
        entries = tracker.entries(experiment_type_filters=filters)
        assert entries == []

        # Filter on dataset
        tracker.add_result(datasets[2], experiment_types[0], 0, result)
        entries = tracker.entries(dataset_filters=filters)
        expected_entry = Entry(experiment_types[2])
        expected_entry.add_seed(0, result)
        expected_entries = [expected_entry]
        assert self._entries_equal(entries, expected_entries)

        # Filter on experiment type and dataset
        entries = tracker.entries(dataset_filters=filters,
                                  experiment_type_filters=filters)
        assert entries == []

        # Match both
        tracker.add_result(datasets[1], experiment_types[1], 1, result * 2)
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(1, result * 2)
        expected_entries = [expected_entry]
        entries = tracker.entries(dataset_filters=filters,
                                  experiment_type_filters=filters)
        assert self._entries_equal(entries, expected_entries)

    def test_add_result(self, tracker, result, experiment_types, datasets):
        # Add a single result
        tracker.add_result(datasets[0], experiment_types[0], 0, result)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries = [expected_entry]
        assert entries == expected_entries

        # Add multiple results to same dataset
        tracker.add_result(datasets[0], experiment_types[1], 0, result)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[1])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        assert self._entries_equal(entries, expected_entries)

        # Add invalid result to same dataset
        with pytest.raises(ValueError) as excinfo:
            tracker.add_result(datasets[0], experiment_types[1], 0, result * 2)
        assert excinfo.match("Seed 0 already in Entry")
        assert self._entries_equal(entries, expected_entries)

        # Add to multiple datasets
        tracker.add_result(datasets[1], experiment_types[0], 0, result)
        entries = tracker.entries()
        expected_entry = Entry(experiment_types[0])
        expected_entry.add_seed(0, result)
        expected_entries.append(expected_entry)
        assert self._entries_equal(entries, expected_entries)

    @pytest.fixture
    def tracker(self):
        tracker = TopLevelTracker("tracker")
        return tracker

    @pytest.fixture
    def datasets(self):
        return ["match-dataset", "dataset-other", "filtered-dataset"]
