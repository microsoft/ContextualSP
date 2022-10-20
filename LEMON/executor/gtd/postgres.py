from collections import defaultdict
from contextlib import contextmanager
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

from gtd.utils import Bunch


class Postgres(object):
    """Provides a wrapper around postgres.

    Args:
        db_name (str): name of database.
        schema_name (str): name of schema.
        user (str): name of user.
        verbose (bool): if True, prints queries when they are executed.
        debug (bool): if True, does not actually execute any queries.

    If the specified schema does not exist, creates it.

    Example:
        with db as Postgres(...):
            db.execute(...)
    """
    def __init__(self, db_name, schema_name, user, password=None, host=None, port=None, verbose=False, debug=False):
        self.db_name = db_name
        self.user = user
        self.verbose = verbose
        self.debug = debug
        self.cursors_opened = 0  # counts the # cursors opened over this connection's lifetime
        self._table_columns = {}
        self.connection = psycopg2.connect(database=db_name, user=user, password=password, host=host, port=port)
        self.cursor = self.connection.cursor()  # this cursor is exclusively used for formatting queries

        self._create_schema(schema_name)  # create if it doesn't exist
        self.execute("SET search_path TO {}, public".format(schema_name))
        self.schema_name = schema_name


    def format(self, query, as_is, params):
        if as_is:
            query = query.format(*as_is)  # literal substitution
        return self.cursor.mogrify(query, params)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.close()

    def close(self):
        self.cursor.close()
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    @contextmanager
    def query_cursor(self, q, lazy_fetch=False, commit=True):
        """Execute a query and yield a cursor.

        All execution performed by the Postgres object uses this method.

        Args:
            q (str): SQL query
            lazy_fetch (bool): whether to use a server-side cursor (lazily fetches results).
        """
        self.cursors_opened += 1

        if self.verbose:
            logging.debug(q)

        if self.debug:
            empty_cursor = Bunch()
            empty_cursor.fetchmany = lambda size: []
            empty_cursor.fetchall = lambda: []
            yield empty_cursor
            return

        cursor_name = 'server_side_{}'.format(self.cursors_opened) if lazy_fetch else None
        with self.connection.cursor(cursor_name, cursor_factory=RealDictCursor) as cursor:
            cursor.execute(q)
            yield cursor

        if commit:
            self.commit()

    def execute(self, q, commit=True):
        """Execute query, return nothing."""
        with self.query_cursor(q, commit=commit):
            pass

    def has_results(self, q):
        """Check if this query returns any results."""
        with self.query_cursor(q) as cursor:
            results = cursor.fetchall()
        return len(results) > 0

    def query(self, q, fetch_size=10000):
        """Return a generator of results from query.

        Uses lazy fetching.

        Args:
            q (str): a SQL query
            fetch_size (int): number of results to fetch at a time (for efficiency purposes)

        Returns:
            Generator[Dict[str, T]]: A generator of results as dicts
        """
        if self.verbose:
            logging.debug(q)

        with self.query_cursor(q, lazy_fetch=True) as cursor:
            while True:
                results = cursor.fetchmany(fetch_size)
                for result in results:
                    yield result
                if len(results) == 0:
                    break

    def iter_table(self, table_name):
        """Return a generator that iterates through all entries in a table.

        Args:
            table_name (str): name of table
        Returns:
            Generator[Dict[str, T]]
        """
        q = self.format("SELECT * from {}", (table_name,), None)
        return self.query(q)

    def match_field(self, table_name, field, value):
        """Get all rows with a particular field value.

        Args:
            table_name (str): Table to query
            field (str): Name of field
            value: Desired value of field.

        Returns:
            Generator[Dict[str, T]]
        """
        q = self.format("SELECT * from {} where {}=%s", (table_name, field), (value,))
        return self.query(q)

    def match_fields(self, table_name, fields):
        """Get all rows with a particular set of field values

        Args:
            table_name (str): name of table
            fields (dict): a map from field names to values

        Returns:
            Generator[Dict[str, T]]
        """
        keys, vals = list(zip(*list(fields.items())))
        field_query = ' AND '.join(['{}=%s'.format(k) for k in keys])
        field_vals = tuple(vals)
        q = self.format("SELECT * from {} where {}", (table_name, field_query), field_vals)
        return self.query(q)

    def match_field_any(self, table_name, field, values):
        """Get all rows with a field value in a particular set.

        Args:
            table_name (str): Table to query
            field (str): Name of field
            value: a list or set of allowed values.

        Returns:
            Generator[Dict[str, T]]
        """
        q = self.format("SELECT * from {} where {} in %s", (table_name, field), (tuple(values),))
        return self.query(q)

    def _schema_exists(self, name):
        """Check if schema exists."""
        q = self.format("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s", None, (name,))
        return self.has_results(q)

    def table_exists(self, name):
        """Check if table exists (under the default schema)."""
        name = name.lower()  # psql tables are always lower-case
        q = self.format("SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                        None, (self.schema_name, name))
        return self.has_results(q)

    def _create_schema(self, name):
        """Create schema if it doesn't exist."""
        if not self._schema_exists(name):
            q = self.format("CREATE SCHEMA {}", (name,), None)
            self.execute(q)

    def create_table(self, name, col_to_type):
        """Create table if it doesn't exist."""
        if not self.table_exists(name):
            col_to_type_pairs = [' '.join(i) for i in list(col_to_type.items())]
            col_type_str = ', '.join(col_to_type_pairs)
            q = self.format("CREATE TABLE {} ({})", (name, col_type_str), None)
            self.execute(q)

    def drop_table(self, name):
        if self.table_exists(name):
            q = self.format("DROP TABLE {}", (name,), None)
            self.execute(q)

    def add_row(self, table_name, row):
        """Add row to table.

        Args:
            table_name (str)
            row (dict[str, T]): a map from column names to values
        """
        columns, vals = list(zip(*list(row.items())))
        col_str = ', '.join(columns)
        vals = tuple(vals)
        q = self.format("INSERT INTO {} ({}) VALUES %s", (table_name, col_str), (vals,))
        self.execute(q)

    def add_rows(self, table_name, table):
        """Efficiently add a batch of rows to a table.

        For an explanation, see:
        https://trvrm.github.io/bulk-psycopg2-inserts.html
        http://stackoverflow.com/questions/8134602/psycopg2-insert-multiple-rows-with-one-query
        http://stackoverflow.com/questions/2271787/psycopg2-postgresql-python-fastest-way-to-bulk-insert

        Args:
            table_name (str): name of table
            table (dict[str, list]): map from a column name to a list of column values
        """
        col_names = list(table.keys())
        col_str = ', '.join(col_names)
        unnest = ', '.join(['unnest(%({})s)'.format(n) for n in col_names])
        for column in list(table.values()):
            assert isinstance(column, list)  # must be a list for unnest to work
        q = self.format("INSERT INTO {} ({}) SELECT {}", (table_name, col_str, unnest), table)
        self.execute(q)

    def add_table(self, table_name, table, col_types):
        """Create table in SQL and add data to it.

        Args:
            table_name (str): name of table
            table (dict[str, list]): a map from column name to column values
            col_types (dict[str, str]): a map from column name to psql column type
        """
        assert not self.table_exists(table_name)
        self.create_table(table_name, col_types)
        self.add_rows(table_name, table)

    def table(self, name):
        results = list(self.iter_table(name))
        table = defaultdict(list)
        for res in results:
            for key, val in res.items():
                table[key].append(val)
        return table

    def row_count(self, table_name, approx=False):
        q = self.format("select count(*) from {}", (table_name,), None)
        q_approx = self.format("SELECT reltuples AS approximate_row_count FROM pg_class WHERE relname = %s",
                        None, (table_name,))

        if approx:
            row = next(self.query(q_approx))
            count = row['approximate_row_count']
        else:
            row = next(self.query(q))
            count = row['count']

        return int(count)


def dict_to_table(d):
    """Convert dict into a two-column table (one col for key, one col for value)."""
    keys, vals = [list(l) for l in zip(*list(d.items()))]
    return {'key': keys, 'val': vals}


def table_to_dict(table):
    keys = table['key']
    vals = table['val']
    return {k: v for k, v in zip(keys, vals)}
