import pytest

from gtd.log import Metadata, SyncedMetadata


class TestMetadata(object):
    @pytest.fixture
    def m(self):
        m = Metadata()
        m['a'] = 10  # this is overwritten
        m['b'] = 'test'

        # namescope setitem
        with m.name_scope('c'):
            m['foo'] = 140

        # nested setitem
        m['a.foo'] = 120
        m['c.bar'] = 'what'

        return m

    def test_getitem(self, m):
        assert m['b'] == 'test'

    def test_nested_getitem(self, m):
        assert m['a.foo'] == 120
        assert m['c.foo'] == 140

    def test_namescope_getitem(self, m):
        with m.name_scope('c'):
            assert m['bar'] == 'what'

    def test_nested_metadata(self, m):
        m_sub = m['a']
        assert isinstance(m_sub, Metadata)
        assert m_sub['foo'] == 120

    def test_contains(self, m):
        assert 'b' in m
        assert 'bar' not in m
        assert 'c.bar' in m


class TestSyncedMetadata(TestMetadata):  # run all the metadata tests
    def test_syncing(self, tmpdir):
        meta_path = str(tmpdir.join('meta.txt'))
        s = SyncedMetadata(meta_path)

        with s.name_scope('job'):
            s['memory'] = 128

        s2 = SyncedMetadata(meta_path)  # reload the file

        assert s2['job.memory'] == 128