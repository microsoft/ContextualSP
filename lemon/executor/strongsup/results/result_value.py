import math

class ResultValue(object):
    """Wrapper class to keep track of valid and final
    accs for 1-5 utterances

    Args:
        valid_acc (list[float]): valid accuracies in order 1-5
        final_acc (list[float]): final accuracies in order 1-5
    """
    def __init__(self, valid_accs, final_accs):
        self._valid_accs = valid_accs
        self._final_accs = final_accs

    @property
    def valid_accs(self):
        return self._valid_accs

    @property
    def overall_valid_acc(self):
        return sum(self._valid_accs) / len(self._valid_accs)

    @property
    def overall_final_acc(self):
        return sum(self._final_accs) / len(self._final_accs)

    @property
    def final_accs(self):
        return self._final_accs

    def squared(self):
        valid_accs = [acc * acc for acc in self._valid_accs]
        final_accs = [acc * acc for acc in self._final_accs]
        return ResultValue(valid_accs, final_accs)

    def sqrt(self):
        valid_accs = [math.sqrt(acc) for acc in self._valid_accs]
        final_accs = [math.sqrt(acc) for acc in self._final_accs]
        return ResultValue(valid_accs, final_accs)

    def __mul__(self, scalar):
        valid_accs = [float(acc) * scalar for acc in self._valid_accs]
        final_accs = [float(acc) * scalar for acc in self._final_accs]
        return ResultValue(valid_accs, final_accs)

    def __div__(self, scalar):
        assert scalar != 0
        return self.__mul__(float(1) / scalar)

    def __add__(self, other):
        valid_accs = [x + y for x, y in zip(self.valid_accs, other.valid_accs)]
        final_accs = [x + y for x, y in zip(self.final_accs, other.final_accs)]
        return ResultValue(valid_accs, final_accs)

    def __sub__(self, other):
        return self + (other * -1)

    def __lt__(self, other):
        return self.valid_accs[2] + self.valid_accs[4] < \
               other.valid_accs[2] + other.valid_accs[4]
        # return sum(self.valid_accs) < sum(other.valid_accs)

    def __eq__(self, other):
        return self.valid_accs == other.valid_accs and \
               self.final_accs == other.final_accs

    def __gt__(self, other):
        return other < self

    def __str__(self):
        # Hide final acc
        return "Valid Acc: {}".format(self._valid_accs)
    __repr__ = __str__
