import unittest

from text import terms


class TestTerms(unittest.TestCase):

    def test_extract_termsets(self):
        # one term
        self.assertEqual(terms.extract_termsets("dew"), [{'dew'}])

        # one term with a word that should not be stemmed
        self.assertEqual(terms.extract_termsets("raining"), [{'raining'}])

    def test_extract_termsets_with_normalization(self):
        # one term
        self.assertEqual(terms.extract_termsets_with_normalization("dew"), [{'dew'}])

        # one term with a word that should be normalized
        self.assertEqual(terms.extract_termsets_with_normalization("raining"), [{'rain'}])

        # one term with two words, one that gets normalized
        self.assertEqual(terms.extract_termsets_with_normalization("raining cats and dogs"), [{'raining cats and dog'}])

        # ANDed terms
        self.assertEqual(terms.extract_termsets_with_normalization("dew AND rain"), [{'dew'}, {'rain'}])

        # ORed terms
        self.assertEqual(terms.extract_termsets_with_normalization("dew OR rain"), [{'dew', 'rain'}])

        # ORed and ANDed terms
        self.assertEqual(terms.extract_termsets_with_normalization("dew OR rain AND sun"), [{'dew', 'rain'}, {'sun'}])

        # more complex arrangements
        self.assertEqual(
            terms.extract_termsets_with_normalization("dew OR rain AND sun AND foo OR bar OR baz"),
            [
                {'dew', 'rain'},
                {'sun'},
                {'foo', 'bar', 'baz'}
            ]
        )

        # as above, but "droplet" and "droplets" in the phrase should become one term "droplet"
        self.assertEqual(
            terms.extract_termsets_with_normalization("dew OR droplet OR droplets AND sun AND foo OR bar OR baz"),
            [
                {'dew', 'droplet'},
                {'sun'},
                {'foo', 'bar', 'baz'}
            ]
        )

    def test_terms_overlap(self):
        self.assertEqual(
            terms.terms_overlap(
                [{'foo'}],
                [{'foo'}]
            ),
            1
        )
        self.assertEqual(
            terms.terms_overlap(
                [{'foo'}],
                [{'bar'}]
            ),
            0
        )
        self.assertEqual(
            terms.terms_overlap(
                [{'diesel'}, {'energi'}],
                [{'diesel'}, {'petrol'}]
            ),
            1
        )
        self.assertEqual(
            terms.terms_overlap(
                [{'plant', 'anim'}],
                [{'soft tissu'}]
            ),
            0
        )
        self.assertEqual(
            terms.terms_overlap(
                [{'nitrogen'}],
                [{'fixed nitrogen', 'usable nitrogen'}]
            ),
            0
        )
        self.assertEqual(
            terms.terms_overlap(
                [{'rain'}, {'water', 'liquid'}],
                [{'rain'}, {'water'}]
            ),
            2
        )

    def test_normalization(self):
        self.assertEqual(
            terms._normalize_words(["the Raining", "DANCING", "experimenting"]),
            ["rain", "danc", "experi"]
        )


if __name__ == '__main__':
    unittest.main()
