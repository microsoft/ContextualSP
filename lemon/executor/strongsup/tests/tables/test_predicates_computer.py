import pytest

from strongsup.tables.predicates_computer import (
        similarity_ratio,
        )


class TestEditDistance(object):
    CASES = [
            ('superman', 'superman', 0),
            ('kitten', 'sitting', 5),
            ('industry', 'interest', 8),
            ('to ardo', 'from ardo', 4),
            ('intend', 'interned', 2),
            ('saturday', 'sunday', 4),
            ('abbababb', 'babbabab', 2),
            ('bababaabba', 'babbabbaba', 4),
            ('bababaabba', 'baababbaba', 4),
            ('babadook', 'gagadook', 4),
            ('mickey', 'icky', 2),
            ('0000000000', '0000000000', 0),
            ('0000010000', '0000100000', 2),
            ]

    def test_similarity_ratio(self):
        for s1, s2, key in self.CASES:
            for threshold in (x * .1 + 1e-3 for x in range(12)):
                correct = 1 - key * 1. / (len(s1) + len(s2))
                correct = correct if correct >= threshold else 0.
                assert abs(similarity_ratio(s1, s2, threshold) - correct) < 1e-6


# TODO: Test other things
