import numpy as np
from strongsup.predicate import Predicate


def softmax(stuff):
    """Quick and dirty way to compute softmax"""
    return (np.exp(stuff) / np.sum(np.exp(stuff))).tolist()


class PredicateGenerator(object):
    """Generate predicates with the specified context."""
    def __init__(self, context):
        self.context = context
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            self.cache[name] = Predicate(name, self.context)
        return self.cache[name]
