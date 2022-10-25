import itertools
import json
import logging
import os.path
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import MutableMapping, Mapping, Sequence, Iterator
from contextlib import contextmanager


from sqlalchemy import MetaData
from sqlalchemy.engine.url import URL

from gtd.io import open_or_create, JSONPicklable
from gtd.utils import ensure_unicode, SimpleExecutor, Failure
from gtd.utils import makedirs
from sqlalchemy import Column, Table
from sqlalchemy import tuple_
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import select


class Closeable(object, metaclass=ABCMeta):
    @abstractmethod
    def close(self):
        """Close this object."""
        pass

    @abstractproperty
    def closed(self):
        """A bool indicating whether this object was closed.

        Returns:
            bool
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not self.closed:
            logging.warn('{} was not properly closed.'.format(self))
            self.close()


class BatchMapping(Mapping, metaclass=ABCMeta):
    """Like the built-in Mapping class, except subclasses must implement batch versions of get and contains."""

    @abstractmethod
    def get_batch(self, keys):
        """Get value for each key in keys.

        Args:
            keys (list): a list of keys

        Returns:
            list: a list of values with the same order corresponding to the list of keys.
                If a given key does not have a value, the corresponding returned value will be a Failure object.
        """
        pass

    def __getitem__(self, key):
        """Get value for key."""
        val = self.get_batch([key])[0]
        if isinstance(val, Failure):
            raise KeyError(key)
        return val

    @abstractmethod
    def contains_batch(self, keys):
        """Check for the presence of each key in keys.

        Args:
            keys (list): a list of keys

        Returns:
            list[bool]: a list of booleans with the same order corresponding to the list of keys, indicating
                whether each key is present in the BatchMapping.
        """
        pass

    def __contains__(self, key):
        """Check if key is in the mapping."""
        return self.contains_batch([key])[0]


class BatchMutableMapping(MutableMapping, BatchMapping, metaclass=ABCMeta):
    """Like the built-in MutableMapping, except subclasses must implement batch versions of setitem and delitem."""

    @abstractmethod
    def set_batch(self, key_val_pairs):
        pass

    def __setitem__(self, key, value):
        self.set_batch([(key, value)])

    @abstractmethod
    def del_batch(self, keys):
        pass

    def __delitem__(self, key):
        self.del_batch([key])


class SimpleBatchMapping(BatchMutableMapping):
    def __init__(self, d=None):
        if d is None:
            d = {}
        self._d = d

    def get_batch(self, keys):
        f = Failure.silent("Could not get key.")
        return [self._d.get(k, f) for k in keys]

    def contains_batch(self, keys):
        return [k in self._d for k in keys]

    def set_batch(self, key_val_pairs):
        for k, v in key_val_pairs:
            self._d[k] = v

    def del_batch(self, keys):
        for k in keys:
            del self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


class CacheWrapperMixin(object):
    def _set_cache(self, cache):
        self._cache = cache

    @property
    def cache(self):
        return self._cache

    def __iter__(self):
        return iter(self.cache)

    def __len__(self):
        return len(self.cache)

    def iteritems(self):
        return iter(self.cache.items())

    def iterkeys(self):
        return iter(self.cache.keys())

    def itervalues(self):
        return iter(self.cache.values())

    def keys(self):
        return list(self.cache.keys())

    def items(self):
        return list(self.cache.items())

    def values(self):
        return list(self.cache.values())


class LazyMapping(CacheWrapperMixin, BatchMapping):
    def __init__(self, cache):
        """Create a LazyMapping.

        Args:
            cache (BatchMutableMapping)
        """
        self._set_cache(cache)

    def contains_batch(self, keys):
        """Determine whether each key in the batch is already present in the cache.

        Args:
            keys (list): a list of keys

        Returns:
            list[bool]: a list of booleans, indicating whether each key is present in the cache
        """
        return self.cache.contains_batch(keys)

    @abstractmethod
    def compute_batch(self, keys):
        """Compute the values for a batch of keys.

        Args:
            keys (list): a list of keys

        Returns:
            list: a list of values with the same order corresponding to the list of keys.
                If a given key does not have a value, the corresponding returned value will be a Failure object.
        """
        pass

    def compute(self, key):
        """Compute the value for a single key.

        Args:
            key

        Returns:
            val
        """
        return self.compute_batch([key])[0]

    def ensure_batch(self, keys, computed_list=False):
        """Ensure that the given keys are present in the cache.

        If a key is not present, its entry will be computed.

        Args:
            keys (list): a list of keys
            computed_list (bool): defaults to False. See Returns description.

        Returns:
            if computed_list:
                list(bool): a list of booleans indicating which keys were freshly computed (may include failed computations)
            else:
                int: the number of keys which were freshly computed
        """
        presence = self.cache.contains_batch(keys)
        to_compute = [key for key, present in zip(keys, presence) if not present]
        computed = self.compute_batch(to_compute)

        updates = []
        for key, val in zip(to_compute, computed):
            if not isinstance(val, Failure):
                updates.append((key, val))

        self.cache.set_batch(updates)

        if computed_list:
            return [not p for p in presence]

        return len([p for p in presence if not p])

    def get_batch(self, keys, compute=True):
        """Get value for each key in keys.

        Args:
            keys (list): a list of keys
            compute (bool): if a key is missing from the cache, compute it. When disabled, just returns Failure
                objects for missing keys.

        Returns:
            list: a list of values with the same order corresponding to the list of keys.
                If a given key's value cannot be computed, the corresponding returned value will be a Failure object.
        """
        if compute:
            self.ensure_batch(keys)
        return self.cache.get_batch(keys)

    @staticmethod
    def compute_batch_parallel(fxn, keys):
        """Execute a function in parallel on the entire batch of keys, using a multi-threaded executor.

        This is a helper function which subclasses of LazyDict can use to implement `compute_batch`.
        Note that speedups will only be obtained if compute is IO bound, due to Python's GIL.

        Args:
            fxn (Callable): function to be called in parallel
            keys (list): a list of keys

        Returns:
            list: result is equivalent to [fxn(key) for key in keys]
        """
        no_result_failure = Failure.silent('No result returned by SimpleExecutor.')
        results = [no_result_failure] * len(keys)
        with SimpleExecutor(fxn) as ex:
            for i, key in enumerate(keys):
                ex.submit(i, key)
            for i, val in ex.results():
                results[i] = val

        for result in results:
            assert result != no_result_failure
        return results


class EagerMapping(CacheWrapperMixin, BatchMapping):
    def __init__(self, cache):
        self._set_cache(cache)
        if len(cache) == 0:
            self.populate(cache)

    @abstractmethod
    def populate(self, cache):
        pass

    def get_batch(self, keys):
        return self.cache.get_batch(keys)

    def contains_batch(self, keys):
        return self.cache.contains_batch(keys)


class EagerSequence(Sequence):
    def __init__(self, cache):
        self._cache = cache
        if len(self.cache) == 0:
            self.populate(self.cache)

    @property
    def cache(self):
        return self._cache

    @abstractmethod
    def populate(self, cache):
        pass

    def __getitem__(self, key):
        return self.cache[key]

    def __iter__(self):
        return iter(self.cache)

    def __len__(self):
        return len(self.cache)


def sqlalchemy_metadata(host, port, database, username, password):
    url = URL(drivername='postgresql+psycopg2', username=username,
              password=password, host=host, port=port, database=database)
    engine = create_engine(url, server_side_cursors=True, connect_args={'connect_timeout': 4})
    # ensure that we can connect
    with engine.begin():
        pass  # this will throw OperationalError if it fails
    return MetaData(engine)


class ORM(object, metaclass=ABCMeta):
    def __init__(self, columns):
        assert isinstance(columns, list)
        for c in columns:
            assert isinstance(c, ORMColumn)
        self._columns = columns

    @property
    def columns(self):
        """Return a list of ORMColumns."""
        return self._columns

    @abstractmethod
    def to_row(self, value):
        """Convert object into database row.

        Args:
            value (object)

        Returns:
            dict[Column, object]
        """
        pass

    @abstractmethod
    def from_row(self, row):
        """Convert row back into object.

        Args:
            dict[Column, object]

        Returns:
            object
        """
        pass

    def bind(self, table):
        for orm_col in self.columns:
            orm_col.bind(table)


class SimpleORM(ORM):
    def __init__(self, column):
        self._col = column
        super(SimpleORM, self).__init__([column])

    def to_row(self, value):
        return {self._col.key: value}

    def from_row(self, row):
        return row[self._col.key]


class CustomORM(ORM):
    def __init__(self, columns, to_row, from_row):
        self._to_row = to_row
        self._from_row = from_row
        super(CustomORM, self).__init__(columns)

    def to_row(self, value):
        return self._to_row(value)

    def from_row(self, row):
        return self._from_row(row)


class ORMColumn(object):
    """Wraps a SQLAlchemy Column object."""
    def __init__(self, *args, **kwargs):
        self._rebuild(args, kwargs)

    def _rebuild(self, args, kwargs):
        if self.bound:
            raise RuntimeError('Cannot rebuild ORMColumn if it is already bound.')
        self._unbound_column = Column(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    @property
    def unbound_column(self):
        return self._unbound_column

    @property
    def name(self):
        return self.unbound_column.name

    def extend(self, *args, **kwargs):
        new_args = self._args + args
        new_kwargs = dict(self._kwargs)
        new_kwargs.update(kwargs)
        self._rebuild(new_args, new_kwargs)

    @property
    def bound(self):
        return hasattr(self, '_column')

    def bind(self, table):
        col_names = [c.name for c in table.columns]
        if len(col_names) != len(set(col_names)):
            raise ValueError('Can only bind to table with unique column names.')
        self._column = table.c[self.name]

    @property
    def column(self):
        """Return SQLAlchemy Column object."""
        if self.bound:
            return self._column
        else:
            raise RuntimeError("Need to bind ORMColumn to a Table.")

    @property
    def key(self):
        """Used to select this column from a SQLAlchemy RowProxy."""
        return self.column


class TableMapping(BatchMutableMapping):
    def __init__(self, name, key_orm, val_orm, metadata, engine=None):
        if engine is None:
            engine = metadata.bind
            assert isinstance(engine, Engine)

        # mark columns as primary keys
        for c in key_orm.columns:
            c.extend(primary_key=True)

        # Convert ORMColumns into SQLAlchemy Columns to construct Table
        orm_cols = key_orm.columns + val_orm.columns
        table_cols = [orm_col.unbound_column for orm_col in orm_cols]

        # avoid overlapping column names
        col_names = [col.name for col in table_cols]
        if len(col_names) != len(set(col_names)):
            raise ValueError("Column names must be unique.")

        try:
            # If table is already defined in metadata, return it.
            #       It is possible for the table to be defined in metadata, but not exist in database.
            #       (e.g. if metadata.drop_all() was called)
            # If not, use reflection to get information about the table from the database, and return it.
            # If table isn't in database, raise NoSuchTableError.
            table = Table(name, metadata, autoload=True)
        except NoSuchTableError:
            # Define the table.
            table = Table(name, metadata, *table_cols)

        # If table does not exist in database, create it.
        metadata.create_all()

        # make sure we only get the columns we expected
        if set([c.name for c in table.columns]) != set(col_names):
            raise ValueError("ORM column names must match table column names exactly.")

        # ORMs must have a reference to the Table's Column objects.
        key_orm.bind(table)
        val_orm.bind(table)

        self._key_orm = key_orm
        self._val_orm = val_orm
        self._table = table
        self._engine = engine

    @property
    def _key_cols(self):
        """Return a list of Columns (not ORMColumns)."""
        return [orm_column.column for orm_column in self._key_orm.columns]

    @property
    def _val_cols(self):
        """Return a list of Columns (not ORMColumns)."""
        return [orm_column.column for orm_column in self._val_orm.columns]

    @contextmanager
    def _transaction(self):
        with self._engine.begin() as conn:
            yield conn
        # connection automatically closed after transaction
        assert conn.closed

    @property
    def table(self):
        return self._table

    def _key_conditions(self, keys):
        vals = []
        for key in keys:
            row = self._key_orm.to_row(key)
            val = tuple(row[c] for c in self._key_cols)
            vals.append(val)
        return tuple_(*self._key_cols).in_(vals)

    def contains_batch(self, keys):
        if len(keys) == 0: return []

        # select all rows matching any of the keys
        condition = self._key_conditions(keys)
        cmd = select(self._key_cols).where(condition)

        # get the set of keys found
        with self._transaction() as conn:
            result = conn.execute(cmd)
            present_keys = set(self._key_orm.from_row(row) for row in result)

        return [key in present_keys for key in keys]

    def get_batch(self, keys):
        if len(keys) == 0: return []

        key_to_index = {k: i for i, k in enumerate(keys)}
        condition = self._key_conditions(keys)
        cmd = select([self.table]).where(condition)
        with self._transaction() as conn:
            results = conn.execute(cmd)
            no_result_failure = Failure.silent('No result returned from TableDict.')
            vals = [no_result_failure] * len(keys)
            for row in results:
                key = self._key_orm.from_row(row)
                val = self._val_orm.from_row(row)
                index = key_to_index[key]
                vals[index] = val
        return vals

    def _kv_to_row(self, key, val, string_cols=False):
        row = self._key_orm.to_row(key)
        row.update(self._val_orm.to_row(val))
        if string_cols:
            row = {col.name: v for col, v in row.items()}
        return row

    def del_batch(self, keys):
        if len(keys) == 0: return
        condition = self._key_conditions(keys)
        cmd = self.table.delete().where(condition)
        with self._transaction() as conn:
            result = conn.execute(cmd)
            if result.rowcount == 0:
                raise KeyError(keys)  # rollback

    def __iter__(self):
        with self._transaction() as conn:
            for row in conn.execute(select(self._key_cols)):
                yield self._key_orm.from_row(row)

    def __len__(self):
        cmd = self.table.count()
        with self._transaction() as conn:
            return conn.execute(cmd).scalar()

    def set_batch(self, key_val_pairs):
        if len(key_val_pairs) == 0: return
        keys, vals = list(zip(*key_val_pairs))

        # make sure keys are unique
        assert len(keys) == len(set(keys))

        present_keys = []
        for key, present in zip(keys, self.contains_batch(keys)):
            if present:
                present_keys.append(key)

        rows = []
        for k, v in key_val_pairs:
            row = self._kv_to_row(k, v, string_cols=True)
            rows.append(row)

        with self._transaction() as conn:
            self.del_batch(present_keys)  # delete rows that are already present
            conn.execute(self.table.insert(), rows)  # insert new rows

    def iteritems(self):
        with self._transaction() as conn:
            for row in conn.execute(select([self.table])):
                key = self._key_orm.from_row(row)
                val = self._val_orm.from_row(row)
                yield (key, val)

    def iterkeys(self):
        return iter(self)

    def itervalues(self):
        for _, val in self.items():
            yield val

    def keys(self):
        return list(self.keys())

    def items(self):
        return list(self.items())

    def values(self):
        return list(self.values())


class FileMapping(MutableMapping, Closeable):
    def __init__(self, path):
        self._path = path
        self._f = open_or_create(self._path, 'r+')
        s = self._f.read()
        if len(s) == 0:
            self._d = {}
        else:
            self._d = json.loads(s)

    def close(self):
        self._f.close()

    @property
    def closed(self):
        return self._f.closed

    def __repr__(self):
        return 'FileMapping at {}'.format(self._path)

    def _dump(self):
        f = self._f
        f.seek(0)
        f.truncate()
        json.dump(self._d, f)
        f.flush()

    def __setitem__(self, key, value):
        self._d[key] = value
        self._dump()

    def __delitem__(self, key):
        del self._d[key]
        self._dump()

    def __getitem__(self, item):
        return self._d[item]

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)


class FileSerializer(object):
    __class__ = ABCMeta

    @abstractmethod
    def to_line(self, obj):
        """Return a string that can be written as a SINGLE line in a file (cannot contain newline character)."""
        pass

    @abstractmethod
    def from_line(self, line):
        pass


class UnicodeSerializer(FileSerializer):
    def to_line(self, obj):
        u = ensure_unicode(obj)
        return u.encode('utf-8')

    def from_line(self, line):
        return line.decode('utf-8')


class CustomSerializer(FileSerializer):
    def __init__(self, to_line, from_line):
        self._to = to_line
        self._from = from_line

    def to_line(self, obj):
        return self._to(obj)

    def from_line(self, line):
        return self._from(line)


class JSONPicklableSerializer(FileSerializer):
    def to_line(self, obj):
        return obj.to_json_str()

    def from_line(self, line):
        return JSONPicklable.from_json_str(line)


class AppendableSequence(Sequence):
    __class__ = ABCMeta

    @abstractmethod
    def append(self, item):
        pass

    def extend(self, items):
        for item in items:
            self.append(item)


class SimpleAppendableSequence(AppendableSequence, Closeable):
    def __init__(self, l=None):
        if l is None:
            l = []
        self._l = l
        self._closed = False

    def __getitem__(self, item):
        if isinstance(item, slice):
            return SequenceSlice(self, item)
        return self._l[item]

    def __len__(self):
        return len(self._l)

    def append(self, item):
        self._l.append(item)

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed


class FileSequenceOffsets(Sequence, Closeable):
    def __init__(self, file_seq):
        offsets_path = file_seq.path + '.offsets'
        file_existed = os.path.isfile(offsets_path)  # check if file already existed
        self._f_write = open_or_create(offsets_path, 'a')  # open for appending only

        if file_existed:
            # load offsets from file into memory
            with open(offsets_path, 'r') as f:
                self._offsets = [int(line) for line in f]  # int cast strips newline automatically
        else:
            # build offsets (in-memory and on-file)
            self._offsets = []
            current_offset = 0
            for line in file_seq.iter_raw_lines():
                self.append(current_offset)
                current_offset += len(line)

        self._offsets_path = offsets_path

    def close(self):
        self._f_write.close()

    @property
    def closed(self):
        return self._f_write.closed

    def __repr__(self):
        return 'FileSequenceOffsets at {}'.format(self._offsets_path)

    def __getitem__(self, i):
        return self._offsets[i]

    def __len__(self):
        return len(self._offsets)

    def append(self, i):
        self.extend([i])

    def extend(self, i_list):
        self._offsets.extend(i_list)
        f = self._f_write
        for i in i_list:
            f.write(str(i))
            f.write('\n')
        f.flush()


class FileSequenceMetaData(Closeable):
    """Stores FileSequence properties in a JSON file."""
    def __init__(self, file_seq):
        """Store metadata about a FileSequence.

        Args:
            file_seq (FileSequence)
        """
        meta_path = file_seq.path + '.meta'
        file_existed = os.path.isfile(meta_path)  # check if file already exists
        self._d = FileMapping(meta_path)  # initialize underlying dict

        if not file_existed:
            self.length = len(file_seq)  # record length

    def close(self):
        self._d.close()

    @property
    def closed(self):
        return self._d.closed

    @property
    def length(self):
        try:
            return self._d['length']
        except KeyError:
            raise AttributeError()

    @length.setter
    def length(self, val):
        self._d['length'] = val

    def __str__(self):
        return str(self._d)

    def __repr__(self):
        return repr(self._d)


class FileSequence(AppendableSequence, Closeable):
    """Sequence backed by a file."""
    def __init__(self, path, serializer=None):
        if serializer is None:
            serializer = UnicodeSerializer()  # by default, just write to file as utf-8 encoded strings

        self._path = path
        self._ser = serializer

        # open or create the corresponding file
        self._f_read = open_or_create(path, 'r')  # for reading only
        self._f_write = open_or_create(path, 'a')  # for appending. Stream positioned at end of file.

        # create metadata
        self._offsets = FileSequenceOffsets(self)  # note: this must come before metadata
        self._meta = FileSequenceMetaData(self)

    def close(self):
        self._meta.close()
        self._offsets.close()
        self._f_write.close()
        self._f_read.close()

    @property
    def closed(self):
        return self._meta.closed and self._offsets.closed and self._f_write.closed and self._f_read.closed

    def __repr__(self):
        return 'FileSequence at {}'.format(self._path)

    @property
    def path(self):
        return self._path

    def _strip_newline(self, line):
        return line[:-1]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SequenceSlice(self, i)

        f = self._f_read
        f.seek(self._offsets[i])
        line = f.readline()
        line = self._strip_newline(line)
        return self._ser.from_line(line)

    def __len__(self):
        return len(self._offsets)

    def append(self, item):
        self.extend([item])

    def extend(self, items):
        f = self._f_write
        offsets = []
        for item in items:
            offset = f.tell()
            offsets.append(offset)
            line = self._ser.to_line(item)
            f.write(line)
            f.write('\n')

        f.flush()
        self._meta.length += len(offsets)  # keep metadata up-to-date
        self._offsets.extend(offsets)

    def iter_raw_lines(self):
        for line in self._f_read:
            yield line

    def __iter__(self):
        for line in self.iter_raw_lines():
            line = self._strip_newline(line)
            yield self._ser.from_line(line)


class SimpleFileSequence(FileSequence):
    def __init__(self, path):
        ser = UnicodeSerializer()
        super(SimpleFileSequence, self).__init__(path, ser)


class Shard(FileSequence):
    """A FileSequence serving as a Shard in a ShardedSequence."""
    @classmethod
    def open(cls, directory, index, max_length, serializer):
        path = cls.shard_path(directory, index)
        if not os.path.isfile(path):
            raise IOError('No such shard: {}'.format(path))
        return Shard(directory, index, max_length, serializer)

    @classmethod
    def shard_path(cls, directory, index):
        return os.path.join(directory, '{}.shard'.format(index))

    def __init__(self, directory, index, max_length, serializer):
        path = self.shard_path(directory, index)
        self._index = index
        self._max_length = max_length
        super(Shard, self).__init__(path, serializer)
        assert len(self) <= self._max_length

    @property
    def index(self):
        return self._index

    @property
    def max_length(self):
        return self._max_length

    @property
    def remaining_space(self):
        return self.max_length - len(self)


class ShardedSequence(AppendableSequence, Closeable):
    def __init__(self, directory, shard_size, serializer):
        self._directory = directory
        self._shard_size = shard_size
        self._serializer = serializer
        # create directory if it does not exist
        makedirs(directory)
        # identify shards in the directory
        self._shards = []
        for k in itertools.count():
            try:
                shard = Shard.open(directory, k, self._shard_size, serializer)
                self._shards.append(shard)
            except IOError:
                break

        # create one shard if there are none
        if len(self._shards) == 0:
            self.add_shard()

        # all shards except the last should match the shard size
        for i, shard in enumerate(self._shards):
            l = len(shard)
            if i == len(self._shards) - 1:  # final shard
                assert l <= self._shard_size
            else:
                assert l == self._shard_size

    def __repr__(self):
        return 'ShardedSequence at {}'.format(self._directory)

    def close(self):
        for shard in self._shards:
            shard.close()

    @property
    def closed(self):
        for shard in self._shards:
            if not shard.closed:
                return False
        return True

    @property
    def shard_size(self):
        return self._shard_size

    @property
    def directory(self):
        return self._directory

    def __len__(self):
        return sum(len(s) for s in self._shards)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SequenceSlice(self, i)

        index = i // self.shard_size
        shard_index = i % self.shard_size
        try:
            shard = self._shards[index]
            return shard[shard_index]
        except IndexError:
            raise IndexError('{} exceeds max index of ShardedSequence.'.format(i))

    def add_shard(self):
        index = len(self._shards)
        shard = Shard(self.directory, index, self.shard_size, self._serializer)
        self._shards.append(shard)
        return shard

    def appendable_shard(self):
        """Return the shard that we can append to.

        If the last existing shard is full, create a new shard and return that.

        Returns:
            Shard
        """
        last_shard = self._shards[-1]
        if last_shard.remaining_space == 0:
            last_shard = self.add_shard()
        return last_shard

    def append(self, item):
        self.extend([item])

    def extend(self, items):
        iter_items = iter(items)
        def get_batch(k):
            """Get up to k more elements from items."""
            results = []
            for _ in range(k):
                try:
                    results.append(next(iter_items))
                except StopIteration:
                    break
            return results

        # keep filling shards until we can't fill them anymore
        while True:
            shard = self.appendable_shard()
            requested = shard.remaining_space
            batch = get_batch(requested)
            shard.extend(batch)
            if len(batch) < requested:
                break

    def __iter__(self):
        return itertools.chain(*self._shards)


class BatchIterator(Iterator, metaclass=ABCMeta):
    def __init__(self, default_batch_size=20):
        self._default_batch_size = default_batch_size

    @abstractmethod
    def next_batch(self, k):
        """Get next batch of elements from iterator.

        Get k more elements from the iterator. If there are less than k elements remaining,
        return whatever remains.

        Raise StopIteration if and only if there are 0 more elements to yield.

        Args:
            k (int): number of elements to yield

        Returns:
            list: batch of elements
        """
        pass

    def __next__(self):
        try:
            return next(self._latest_batch)
        except (AttributeError, StopIteration):
            self._latest_batch = iter(self.next_batch(self._default_batch_size))
            return next(self._latest_batch)


class LazyIterator(BatchIterator, metaclass=ABCMeta):
    def __init__(self, cache, default_batch_size=100):
        """Create a CacheIterator.

        Args:
            cache (AppendableSequence): an appendable sequence
        """
        self._iterated = 0
        self._cache = cache
        super(LazyIterator, self).__init__(default_batch_size=default_batch_size)

    @property
    def iterated(self):
        """Number of elements produced by this iterator so far."""
        return self._iterated

    @property
    def cache(self):
        return self._cache

    @abstractmethod
    def compute_batch(self, k):
        """Compute the next k items for the iterator.

        This should be a function of self.iterated, self.cache and k.
        Besides these 3 variables, it should NOT rely on any state accumulated from previous iterations of the iterator.

        Args:
            k (int)

        Returns:
            A list of up to k items. If there aren't k more items to compute, just return whatever there
            is to compute.
        """
        pass

    @property
    def num_computed(self):
        return len(self.cache)

    def _ensure_batch(self, k):
        """Ensure that the cache has the next k items.

        If there aren't k more items to add, just add whatever can be added.

        Returns:
            the number of freshly computed new items
        """
        missing = (self.iterated + k) - len(self.cache)
        if missing <= 0:
            return 0  # cache already has everything we need
        batch = self.compute_batch(k)
        new_items = batch[k - missing:]
        self.cache.extend(new_items)
        return len(new_items)

    def next_batch(self, k):
        self._ensure_batch(k)
        cache_excess = len(self.cache) - self.iterated
        num_to_yield = min(cache_excess, k)  # sometimes the cache doesn't have k more
        if num_to_yield == 0:
            raise StopIteration  # no more elements

        i = self._iterated
        batch = list(self.cache[i:i + num_to_yield])

        self._iterated += num_to_yield
        return batch

    def advance_to(self, index):
        """Advance the iterator to the specified index.

        Args:
            index (int): the next item yielded by the iterator will be iterator[index]
        """
        if index > len(self.cache):
            raise IndexError('Cache has not been computed up to index {} yet.'.format(index))
        self._iterated = index

    def ensure_to(self, index, batch_size):
        """Ensure that every value up to (but not including) index has been computed.

        Args:
            index (int)
            batch_size (int): size of the batches used to compute missing values.
        """
        while True:
            n = self.num_computed
            if n >= index:
                break
            self.advance_to(n)
            self._ensure_batch(batch_size)


class SequenceSlice(Sequence):
    def __init__(self, seq, slice):
        self._seq = seq
        start, stop, step = slice.start, slice.stop, slice.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(seq)
        if step is None:
            step = 1

        for val in (start, stop, step):
            if val < 0:
                raise ValueError("Slice values must be non-negative.")

        self.start, self.stop, self.step = start, stop, step

    def __getitem__(self, i):
        if i < 0:  # allow for negative indexing
            if i < -len(self):  # only allow negatives in the appropriate range
                raise IndexError()
            i = i % len(self)  # convert to positive index

        idx = self.start + self.step * i
        if idx >= self.stop:
            raise IndexError()
        return self._seq[idx]

    def __len__(self):
        diff = self.stop - self.start
        num_items = diff / self.step  # integer division rounds down
        remainder = diff % self.step
        if remainder > 0:
            num_items += 1
        return num_items