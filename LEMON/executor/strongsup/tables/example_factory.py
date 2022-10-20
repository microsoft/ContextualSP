import csv
import os
from codecs import open

from strongsup.example import Context, Example
from strongsup.example_factory import ExampleFactory
from strongsup.predicate import Predicate
from strongsup.utils import EOU
from strongsup.tables.value import to_value_list
from strongsup.tables.utils import tsv_unescape, tsv_unescape_list, resolve_ptb_brackets
from strongsup.tables.world import WikiTableWorld


################################
# WikiTableExampleFactory

class WikiTableExampleFactory(ExampleFactory):
    """Read example from the WikiTableQuestions dataset from a TSV file.
    The file should contain the following fields:
        id, utterance, context, targetValue
    - If the field `tokens` (CoreNLP tokenization) is present, use it instead of `utterance`
    - If the field `targetCanon` is present, also use it to construct more accurate target values
    - If supervised = True, the file should also have the field `logicalForm`
    """

    def __init__(self, filename, supervised=False):
        self._filename = filename
        self._supervised = supervised

    @property
    def examples(self):
        with open(self._filename, 'r', 'utf8') as fin:
            header = fin.readline().rstrip('\n').split('\t')
            for line in fin:
                record = dict(list(zip(header, line.rstrip('\n').split('\t'))))
                # Build Example
                table_path = WikiTableWorld(record['context'])
                if 'tokens' in record:
                    raw_utterance = resolve_ptb_brackets(tsv_unescape_list(record['tokens']))
                else:
                    raw_utterance = tsv_unescape(record['utterance']).split()
                context = Context(table_path, [raw_utterance])
                answer = to_value_list(tsv_unescape_list(record['targetValue']),
                        tsv_unescape_list(record['targetCanon'])
                        if 'targetCanon' in record else None)
                if not self._supervised:
                    logical_form = None
                else:
                    logical_form_text = record.get('logicalForm', '')
                    if logical_form_text == 'None':
                        logical_form = None
                    else:
                        logical_form = []
                        for name in logical_form_text.split():
                            logical_form.append(Predicate(name, context))
                        if logical_form[-1] != EOU:
                            logical_form.append(Predicate(EOU, context))
                example = Example(context, answer, logical_form)
                yield example


################################
# Testing purposes
if __name__ == '__main__':
    from dependency.data_directory import DataDirectory
    factory = WikiTableExampleFactory(os.path.join(
        DataDirectory.seq_questions, 'random-split-1-dev-processed.tsv'))
    for i, ex in enumerate(factory.examples):
        print(ex.context.utterances, ex.answer, ex.logical_form)
        if i == 10:
            exit(0)
