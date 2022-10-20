import json
import logging
import time
from abc import ABCMeta, abstractmethod
from collections import Counter

import pytest
from gtd.persist import LazyMapping, EagerMapping, TableMapping, ORM, ORMColumn, FileSequence, FileSerializer, SimpleORM, \
    ShardedSequence, CustomSerializer, LazyIterator, BatchIterator, SimpleBatchMapping, SequenceSlice
from sqlalchemy import MetaData, String, Integer, Table, create_engine, select, Column
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.inspection import inspect


class BatchMappingTester(object):
    pass  # TODO
    # make sure getitem throws KeyError when appropriate


class BatchMutableMappingTester(BatchMappingTester):
    pass  # TODO


class MetaDataExample(MetaData):
    def __init__(self):
        url = URL(drivername='postgresql+psycopg2', username='Kelvin',
                  host='localhost', port=5432, database='test_db')
        try:
            engine = create_engine(url)
            engine.connect()
            logging.info('Using Postgres test database.')
        except OperationalError:
            # postgres test database not available
            url = 'sqlite:///:memory:'
            engine = create_engine(url)
            logging.warn('Using SQLite test database.')
        super(MetaDataExample, self).__init__(engine)


class LazyMappingExample(LazyMapping):
    def __init__(self, cache):
        super(LazyMappingExample, self).__init__(cache)
        self.computes_called = Counter()

    def compute_batch(self, keys):
        for key in keys:
            self.computes_called[key] += 1
        return [k * 2 for k in keys]


class TestLazyMapping(object):
    @pytest.fixture
    def lazy_dict(self):
        cache = SimpleBatchMapping()
        return LazyMappingExample(cache)

    def test_getitem(self, lazy_dict):
        d = lazy_dict
        cache = d.cache
        assert len(cache) == 0
        assert d[3] == 6

        # check that it entered cache
        assert cache[3] == 6

        # get the same value
        assert d[3] == 6

        # every computation only done once
        for val in d.computes_called.values():
            assert val <= 1

    def test_get_batch(self, lazy_dict):
        def assert_batches(xs, correct):
            results = lazy_dict.get_batch(xs)
            results_par = LazyMapping.compute_batch_parallel(lambda k: 2 * k, xs)
            assert results == correct
            assert results_par == correct

            # every computation only done once
            for val in lazy_dict.computes_called.values():
                assert val <= 1
            # WARNING: this test could fail because computes_called is a Counter, which may
            # not be thread-safe.

        assert_batches([0, 1, 2, 3], [0, 2, 4, 6])
        assert_batches([2, 3, 4], [4, 6, 8])


class EagerMappingExample(EagerMapping):
    def __init__(self, cache):
        super(EagerMappingExample, self).__init__(cache)

    def populate(self, cache):
        cache['a'] = 1
        cache['b'] = 2


def test_eager_mapping():
    cd = EagerMappingExample({})
    assert cd.cache == {'a': 1, 'b': 2}

    # if cache is already populated, doesn't overwrite it
    cd2 = EagerMappingExample({'d': 3})
    assert cd2.cache == {'d': 3}


class ORMTester(object, metaclass=ABCMeta):
    @pytest.fixture(scope='session')
    def metadata(self):
        return MetaDataExample()

    @abstractmethod
    def object(self):
        pass

    @abstractmethod
    def orm(self):
        pass

    @pytest.yield_fixture
    def table(self, orm, metadata):
        metadata.drop_all()  # clear the database
        table_args = [c.unbound_column for c in orm.columns]
        table = Table('test_table', metadata, *table_args)
        metadata.create_all()
        yield table
        metadata.drop_all()

    def test_preserve_object(self, orm, object, table, metadata):
        orm.bind(table)
        row = orm.to_row(object)
        for key in row:
            assert isinstance(key, Column)

        eng = metadata.bind
        with eng.begin() as conn:
            conn.execute(table.insert(values=row))
            result = conn.execute(select([table]))
            new_row = result.first()

        new_object = orm.from_row(new_row)
        assert new_object == object


class ExampleKeyORM(ORM):
    def __init__(self):
        self.name = ORMColumn('name', String)
        self.age = ORMColumn('age', Integer)
        columns = [self.name, self.age]
        super(ExampleKeyORM, self).__init__(columns)

    def to_row(self, value):
        name, age = value
        return {self.name.key: name, self.age.key: age}

    def from_row(self, row):
        return row[self.name.key], row[self.age.key]


class TestExampleKeyORM(ORMTester):
    @pytest.fixture
    def object(self):
        return ('bob', 4)

    @pytest.fixture
    def orm(self):
        return ExampleKeyORM()


class ExampleValORM(ORM):
    def __init__(self):
        self.name = ORMColumn('json', String)
        super(ExampleValORM, self).__init__([self.name])

    def to_row(self, value):
        return {self.name.key: json.dumps(value)}

    def from_row(self, row):
        return json.loads(row[self.name.key])


class TestTableMapping:
    @pytest.fixture(scope='session')
    def metadata(self):
        return MetaDataExample()

    @pytest.yield_fixture
    def table_dict(self, metadata):
        metadata.drop_all()
        key_orm = ExampleKeyORM()
        val_orm = ExampleValORM()
        td = TableMapping('test_table', key_orm, val_orm, metadata)
        td[('ren', 1)] = {'hobby': 'bowling'}
        td[('bob', 2)] = {'hobby': 'bowling'}
        yield td
        metadata.drop_all()

    def test_contains(self, table_dict):
        assert ('ren', 1) in table_dict
        assert ('ren', 2) not in table_dict

    def test_contains_batch(self, table_dict):
        # note that there is a duplicate
        batch = [('ren', 1), ('ren', 2), ('ren', 1), ('bob', 2)]
        correct = [True, False, True, True]
        presence = table_dict.contains_batch(batch)
        assert presence == correct

    def test_correct_table(self, table_dict):
        correct_columns = ['name', 'age', 'json']
        correct_keys = ['name', 'age']
        names = lambda cols: [col.name for col in cols]

        table = table_dict.table
        assert names(table.columns) == correct_columns
        assert names(inspect(table).primary_key.columns) == correct_keys

    def test_set_batch(self, table_dict):
        bob_json = {'hobby': 'golf'}
        james_json = {'hobby': 'tennis'}
        ren_json = {'hobby': 'bowling'}
        table_dict.set_batch([(('bob', 2), bob_json), (('james', 3), james_json)])

        d = dict(table_dict)
        assert d == {('bob', 2): bob_json, ('james', 3): james_json, ('ren', 1): ren_json}
        # note that bob_json was overwritten from bowling to golf

    def test_getitem(self, table_dict):
        assert table_dict[('ren', 1)] == {'hobby': 'bowling'}
        with pytest.raises(KeyError):
            bob_val = table_dict[('bob', 1)]

    def test_setitem(self, table_dict):
        table_dict[('ren', 1)] = {'hobby': 'none'}
        d = dict(table_dict)
        assert d == {('ren', 1): {'hobby': 'none'},
                     ('bob', 2): {'hobby': 'bowling'},
                     }

    def test_delitem(self, table_dict):
        del table_dict[('bob', 2)]
        assert dict(table_dict) == {('ren', 1): {'hobby': 'bowling'}}
        with pytest.raises(KeyError):
            del table_dict[('bob', 1)]

    def test_iter(self, table_dict):
        assert set(iter(table_dict)) == {('ren', 1), ('bob', 2)}

    def test_len(self, table_dict):
        assert len(table_dict) == 2

    # TODO: test iterkeys, iteritems, itervalues


class AppendableSequenceTester(object):
    @abstractmethod
    def empty_list(self):
        """An empty list object to be tested."""
        pass

    @abstractmethod
    def reference_list(self):
        """A standard Python list containing at least 5 items."""
        pass

    def test_append_getitem(self, empty_list, reference_list):
        lst = empty_list
        item = reference_list[0]
        lst.append(item)
        assert lst[0] == item

    def test_extend(self, empty_list, reference_list):
        lst = empty_list
        lst.extend(reference_list)
        for i, item in enumerate(reference_list):
            assert lst[i] == item
        assert len(lst) == len(reference_list)

    def test_len(self, empty_list, reference_list):
        lst = empty_list
        item = reference_list[0]
        lst.append(item)
        lst.append(item)
        lst.append(item)
        lst.append(item)
        assert len(lst) == 4

    def test_iter(self, empty_list, reference_list):
        lst = empty_list
        lst.extend(reference_list)
        for i, item in enumerate(lst):
            assert item == reference_list[i]

    def test_slice(self, empty_list, reference_list):
        lst = empty_list
        lst.extend(reference_list)

        assert list(lst[0:2:5]) == reference_list[0:2:5]


class FileSerializerExample(FileSerializer):
    def to_line(self, s):
        return s

    def from_line(self, line):
        return line


class FileSerializerTester(object, metaclass=ABCMeta):
    @abstractmethod
    def serializer(self):
        pass

    @abstractmethod
    def object(self):
        pass

    def test_serializer(self, serializer, object):
        line = serializer.to_line(object)
        new_obj = serializer.from_line(line)
        assert new_obj == object


class TestFileSequence(AppendableSequenceTester):

    @pytest.yield_fixture
    def empty_list(self, tmpdir):
        path = tmpdir.join('test_file_list.txt')
        # whether to use gzip
        with FileSequence(str(path), FileSerializerExample()) as seq:
            yield seq

    @pytest.fixture
    def reference_list(self):
        return 'a b c d e f g'.split()

    def test_json_newline(self, tmpdir):
        path = str(tmpdir.join('test_json_items.txt'))
        ser = CustomSerializer(lambda o: json.dumps(o), lambda l: json.loads(l))
        fs = FileSequence(path, ser)

        items = ['hey\nthere', 'two\nobjects serialized']
        fs.extend(items)

        for i, val in enumerate(fs):
            assert val == items[i]

    def test_reload(self, empty_list, reference_list):
        empty_list.extend(reference_list)
        l = empty_list
        new_l = FileSequence(l.path, l._ser)

        assert len(new_l) == len(l)
        for i1, i2 in zip(new_l, l):
            assert i1 == i2


class TestShardedSequence(AppendableSequenceTester):
    @pytest.yield_fixture
    def empty_list(self, tmpdir):
        path = str(tmpdir)
        shard_size = 3
        with ShardedSequence(path, shard_size, FileSerializerExample()) as seq:
            yield seq

    @pytest.fixture
    def reference_list(self):
        return [str(i) for i in range(16)]

    def test_reload(self, empty_list, reference_list):
        empty_list.extend(reference_list)  # populate the list
        l = empty_list

        # reload it
        new_l = ShardedSequence(l.directory, l.shard_size, FileSerializerExample())

        assert len(new_l) == len(l)
        for i1, i2 in zip(new_l, l):
            assert i1 == i2


class FileSequenceExample(FileSequence):
    def __init__(self, path):
        ser = FileSerializerExample()
        super(FileSequenceExample, self).__init__(path, ser)


class TableMappingExample(TableMapping):
    def __init__(self, metadata):
        key_orm = SimpleORM(ORMColumn('key', Integer))
        val_orm = SimpleORM(ORMColumn('val', String))
        super(TableMappingExample, self).__init__('tabledict_example', key_orm, val_orm, metadata)


class TestTableMappingSpeed(object):
    @pytest.fixture(scope='session')
    def metadata(self):
        return MetaDataExample()

    @pytest.yield_fixture
    def file_list(self, tmpdir):
        path = tmpdir.join('test_file_list.txt')
        with FileSequenceExample(str(path)) as seq:
            yield seq

    @pytest.yield_fixture
    def raw_file(self, tmpdir):
        p = str(tmpdir.join('raw_file.txt'))
        with open(p, 'w') as f:
            yield f

    @pytest.yield_fixture
    def table_dict(self, metadata):
        metadata.drop_all()
        yield TableMappingExample(metadata)
        metadata.drop_all()

    def test_extend(self, raw_file, file_list, table_dict):
        def time_it(fxn):
            start = time.time()
            fxn()
            stop = time.time()
            return stop - start

        # 100 rows of text, each with 500,000 characters
        vals = ['a' * 500000] * 100

        def extend_raw():
            for v in vals:
                raw_file.write(v)
                raw_file.write('\n')

        def extend_file():
            file_list.extend(vals)

        def extend_dict():
            d = {i: v for i, v in enumerate(vals)}
            table_dict.update(d)

        raw_time = time_it(extend_raw)
        file_time = time_it(extend_file)
        dict_time = time_it(extend_dict)

        # just make sure we did the inserts
        assert len(file_list) == 100
        assert len(table_dict) == 100

        assert file_time < raw_time * 2

        # TableDict should not be more than 20x slower than file
        # On average, seems to be about 15x slower
        assert dict_time < file_time * 20

        # should take less than two seconds
        assert dict_time < 2


class LazyIteratorExample(LazyIterator):
    def __init__(self):
        cache = []
        super(LazyIteratorExample, self).__init__(cache)

    def compute_batch(self, k):
        batch = []
        for i in range(k):
            item = self.iterated + i
            if item == 15: break
            batch.append(item)
        return batch


class TestLazyIterator(object):
    @pytest.fixture
    def iterator(self):
        return LazyIteratorExample()

    def test_iter(self, iterator):
        assert list(iterator) == list(range(15))

    def test_next_batch(self, iterator):
        assert iterator.next_batch(6) == [0, 1, 2, 3, 4, 5]
        assert iterator.next_batch(2) == [6, 7]
        assert iterator.next_batch(5) == [8, 9, 10, 11, 12]
        assert iterator.next_batch(8) == [13, 14]
        with pytest.raises(StopIteration):
            iterator.next_batch(1)


class ExampleBatchIterator(BatchIterator):
    def __init__(self, total):
        self.iterated = 0
        self.total = total
        super(ExampleBatchIterator, self).__init__(default_batch_size=30)

    def next_batch(self, k):
        batch = [self.iterated + i for i in range(k)]
        batch = [b for b in batch if b < self.total]
        if len(batch) == 0:
            raise StopIteration
        self.iterated += len(batch)
        return batch


class TestBatchIterator(object):
    @pytest.fixture
    def iterator(self):
        return ExampleBatchIterator(8)

    def test_iterator(self, iterator):
        assert list(iterator) == [0, 1, 2, 3, 4, 5, 6, 7]


class TestSequenceSlice(object):
    @pytest.fixture
    def seq(self):
        return list(range(10))

    def test_full(self, seq):
        ss = list(SequenceSlice(seq, slice(2, 8, 3)))
        assert ss == [2, 5]

    def test_partial(self, seq):
        ss = list(SequenceSlice(seq, slice(None, 8, 3)))
        assert ss == [0, 3, 6]

        ss = list(SequenceSlice(seq, slice(None, 8, None)))
        assert ss == [0, 1, 2, 3, 4, 5, 6, 7]

        ss = list(SequenceSlice(seq, slice(None, None, None)))
        assert ss == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_negative(self, seq):
        ss = SequenceSlice(seq, slice(None, 8, 3))

        assert ss[-1] == 6
        assert ss[-2] == 3
        assert ss[-3] == 0

        with pytest.raises(IndexError):
            ss[-4]

