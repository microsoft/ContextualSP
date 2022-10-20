from gtd.utils import Bunch
from strongsup.example import Example, Context
from strongsup.experiment import example_to_supervised_cases
from strongsup.tests.utils import PredicateGenerator
from strongsup.utils import EOS



def test_example_to_supervised_cases():
    class DummyTablePath(object):
        graph = 'GRAPH!'
    context = Context(DummyTablePath(), 'hello', executor='dummy executor')
    p = PredicateGenerator(context)
    context._predicates = [p('a'), p('b'), p('c'), p('d'), p(EOS)]
    answer = None
    logical_form = [p('a'), p('b'), p('c'), p('d')]
    example = Example(context, answer, logical_form)

    cases = example_to_supervised_cases(example)
    c0, c1, c2, c3 = cases
    assert c0.previous_decisions == []
    assert c1.previous_decisions == [p('a')]
    assert c2.previous_decisions == [p('a'), p('b')]
    assert c3.previous_decisions == [p('a'), p('b'), p('c')]

    assert c0.decision == p('a')
    assert c1.decision == p('b')
    assert c2.decision == p('c')
    assert c3.decision == p('d')
