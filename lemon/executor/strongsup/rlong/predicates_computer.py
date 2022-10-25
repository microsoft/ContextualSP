from strongsup.predicates_computer import PredicatesComputer
from strongsup.rlong.predicate import RLongPredicate


class RLongPredicatesComputer(PredicatesComputer):
    def compute_predicates(self, tokens):
        """Return list[(Predicate, alignment)]"""
        return [(x, []) for x in self._ALL_PREDICATES]


class RLongAlchemyPredicatesComputer(RLongPredicatesComputer):
    _ALL_PREDICATES = [
            RLongPredicate(x) for x in
            [
                'r', 'y', 'g', 'o', 'p', 'b',
                '1', '2', '3', '4', '5', '6', '7',
                '-1',
                'X1/1',
                'PColor',
                'APour', 'AMix', 'ADrain',
                'all-objects', 'index',
                'H0', 'H1', 'H2',
            ]]


class RLongScenePredicatesComputer(RLongPredicatesComputer):
    _ALL_PREDICATES = [
            RLongPredicate(x) for x in
            [
                'r', 'y', 'g', 'o', 'p', 'b', 'e',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '-1',
                'PShirt', 'PHat', 'PLeft', 'PRight', 'DShirtHat',
                'ALeave', 'ASwapHats', 'AMove', 'ACreate',
                'all-objects', 'index',
                'H0', 'H1', 'H2', 'H3',
            ]]


class RLongTangramsPredicatesComputer(RLongPredicatesComputer):
    _ALL_PREDICATES = [
            RLongPredicate(x) for x in
            [
                '1', '2', '3', '4', '5',
                '-1',
                'AAdd', 'ASwap', 'ARemove',
                'all-objects', 'index',
                'H0', 'H1', 'H2',
            ]]


class RLongUndogramsPredicatesComputer(RLongPredicatesComputer):
    _ALL_PREDICATES = [
            RLongPredicate(x) for x in
            [
                '1', '2', '3', '4', '5',
                '-1',
                'AAdd', 'ASwap', 'ARemove',
                'all-objects', 'index',
                'H0', 'H1', 'H2', 'HUndo',
            ]]


################################
# Singletons

SINGLETONS = {
        'alchemy': RLongAlchemyPredicatesComputer(),
        'scene': RLongScenePredicatesComputer(),
        'tangrams': RLongTangramsPredicatesComputer(),
        'undograms': RLongUndogramsPredicatesComputer(),
        }

def get_predicates_computer(domain_name):
    return SINGLETONS[domain_name]

def get_fixed_predicates(domain_name):
    return SINGLETONS[domain_name]._ALL_PREDICATES
