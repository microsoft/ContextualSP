from unittest import TestCase

from gtd.graph import Graph


class TestGraph(TestCase):

    def test_shortest_path(self):

       triples = [
          ('1', '2', '3'),
          ('3', '4', '5'),
          ('1', '0', '5'),
       ]
       self.assertEqual(
          Graph(triples).shortest_path('1', '5'),
          ['1', '0', '5']
       )
       self.assertEqual(
          Graph(triples[:2]).shortest_path('1', '5'),
          ['1', '2', '3', '4', '5']
       )
