from prettytable import PrettyTable
from strongsup.results.recipe import RLongCookbook


class TableDrawer(object):
    """Given a list of Entries, draws tables based on some criteria.

    Args:
        entries (list[Entry]): the entries
        name (string): the name of this table (typically the dataset from which
            the entries come)
    """
    def __init__(self, entries, name):
        self._entries = entries
        self._name = name

    def avg_table(self, final=False):
        """Returns the ASCII string of the table corresponding to the
        results for these entries, averaged over all seeds.

        Args:
            final (bool): table contains Final results if True, otherwise
                valid results

        Returns:
            string: the table
        """
        entries = sorted(self._entries,
                         key=lambda entry: entry.avg,
                         reverse=True)

        table = PrettyTable()
        table.field_names = self._header(final)
        cookbook = RLongCookbook()
        for entry in entries:
            etype_name = cookbook.get_recipe_name(
                    entry.experiment_type.configs, entry.experiment_type.base)
            if etype_name is None:
                etype_name = str(entry.experiment_type)
            name = "{}-{}".format(
                    self._name, truncate(etype_name))
            result = entry.avg
            row = [name]
            if final:
                row = row + [result.overall_final_acc] + result.final_accs
            else:
                row = row + [result.overall_valid_acc] + result.valid_accs
            table.add_row(row)
        return table

    def all_table(self, final=False):
        """Table with all the seeds.

        Args:
            final (bool): table contains Final results if True, otherwise
                valid results

        Returns:
            string: the table
        """
        rows = sorted(((entry, seed) for entry in self._entries
                      for seed in entry.seeds),
                      key=lambda entry_seed: entry_seed[0].get_value(entry_seed[1]),
                      reverse=True)

        table = PrettyTable()
        table.field_names = self._header(final)

        cookbook = RLongCookbook()
        for entry, seed in rows:
            etype_name = cookbook.get_recipe_name(
                    entry.experiment_type.configs, entry.experiment_type.base)
            if etype_name is None:
                etype_name = str(entry.experiment_type)
            name = "{}-{}-{}".format(
                    self._name, truncate(etype_name), seed)
            result = entry.get_value(seed)
            row = [name]
            if final:
                row = row + [result.overall_final_acc] + result.final_accs
            else:
                row = row + [result.overall_valid_acc] + result.valid_accs
            table.add_row(row)
        return table

    # TODO: Clean up...
    def stddev_table(self, final=False):
        """Table with stddevs"""
        entries = sorted(self._entries,
                         key=lambda entry: entry.avg,
                         reverse=True)

        table = PrettyTable()
        acc_type = "Final" if final else "Valid"
        header = ["Experiment Type"]
        for i in range(1, 6):
            header.append("{} stddev {} utt".format(acc_type, i))
        table.field_names = header
        cookbook = RLongCookbook()
        for entry in entries:
            etype_name = cookbook.get_recipe_name(
                    entry.experiment_type.configs, entry.experiment_type.base)
            if etype_name is None:
                etype_name = str(entry.experiment_type)
            name = "{}-{}".format(
                    self._name, truncate(etype_name))
            stddev = entry.var.sqrt()
            row = [name]
            if final:
                row = row + stddev.final_accs
            else:
                row = row + stddev.valid_accs
            table.add_row(row)
        return table

    def _header(self, final=False):
        acc_type = "Final" if final else "Valid"
        header = ["Experiment Type", "Overall {} Acc".format(acc_type)]
        for i in range(1, 6):
            header.append("{} Acc {} utt".format(acc_type, i))
        return header


def truncate(s):
    truncate_len = 50
    return s[:truncate_len - 3] + "..." if len(s) > truncate_len else s
