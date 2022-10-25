import pytest

from gtd.io import IntegerDirectories


class TestIntegerDirectories(object):
    @pytest.fixture
    def int_dirs(self, tmpdir):
        tmpdir.mkdir('152_blah')
        tmpdir.mkdir('153_woo')
        tmpdir.mkdir('1_')  # no suffix, should still match
        tmpdir.mkdir('-1')  # no suffix, should still match
        tmpdir.mkdir('_10')  # prefix is not integer, ignore
        tmpdir.mkdir('.DS_Store')
        tmpdir.mkdir('other')
        return IntegerDirectories(str(tmpdir))

    def test_keys(self, int_dirs):
        assert list(int_dirs.keys()) == [-1, 1, 152, 153]
        assert len(int_dirs) == 4

    def test_largest_int(self, int_dirs):
        assert int_dirs.largest_int == 153

    def test_new_dir(self, tmpdir, int_dirs):
        correct = str(tmpdir.join('154'))
        assert int_dirs.new_dir() == correct

    def test_new_dir_named(self, tmpdir, int_dirs):
        correct = str(tmpdir.join('154')) + '_foobar'
        assert int_dirs.new_dir('foobar') == correct