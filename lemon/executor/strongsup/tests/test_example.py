from gtd.utils import Bunch
from strongsup.example import Utterance


class TestUtterance(object):
    def test_eq(self):
        ctx1 = Bunch()
        ctx2 = Bunch()

        utt1 = Utterance(('a', 'b', 'c'), ctx1, 0)
        utt2 = Utterance(('AA', 'B', 'CCC'), ctx1, 0)
        utt3 = Utterance(('a', 'b', 'c'), ctx2, 0)

        assert utt1 == utt2
        assert utt1 != utt3
