class EntrySelector(object):
    """Given a list of Entries, returns single Entry based on some
    criteria.

    Args:
        entries (list[Entry]): the entries
    """
    def __init__(self, entries):
        self._entries = entries

    @property
    def best_any_seed(self):
        """Returns the Entry with the best ResultValue over any seed."""
        if len(self._entries) == 0:
            return None
        return max(self._entries, key=lambda entry: entry.best[1])

    @property
    def best_avg(self):
        """Returns the Entry with the best ResultValue averaged over
        all seeds."""
        if len(self._entries) == 0:
            return None
        return max(entries, key=lambda entry: entry.avg)
