# -*- coding: utf-8 -*-
import pytest
from strongsup.tables.utils import (
        tsv_unescape, tsv_unescape_list, normalize,
        )

class TestStringMethods(object):

    def test_tsv_unescape(self):
        assert tsv_unescape(r'abn\ncd\p\\\pp') == 'abn\ncd|\\|p'
        assert tsv_unescape_list(r'abn\ncd\p\\\pp|u\n\pac|r||d') == [
                'abn\ncd|\\|p', 'u\n|ac', 'r', '', 'd']

    def test_normalize(self):
        assert normalize(' This  is  a  BOOK†[a][1]') == 'this is a book'
        assert normalize('Apollo 11 (1969) 「阿波罗」') == 'apollo 11 (1969) 「阿波罗」'
        assert normalize('"Apollo 11 (1969)"') == 'apollo 11'
        assert normalize('"Apollo 11" (1969)') == 'apollo 11'
        assert normalize('“Erdős café – ε’š delight.”') == 'erdos cafe - ε\'s delight'
        assert normalize('3.14') == '3.14'
