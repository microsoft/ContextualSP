from collections import defaultdict, Counter, deque
import numpy as np
import random

from gtd import utils


# defines whether an edge is inverted or not
inverted = lambda r: r[:2] == '**'
invert = lambda r: r[2:] if inverted(r) else '**' + r


class Graph(object):
    def __init__(self, triples):
        self.triples = triples
        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        for s, r, t in triples:
            relation_args[r]['s'].add(s)
            relation_args[r]['t'].add(t)
            neighbors[s][r].add(t)
            neighbors[t][invert(r)].add(s)

        def freeze(d):
            frozen = {}
            for key, subdict in d.items():
                frozen[key] = {}
                for subkey, set_val in subdict.items():
                    frozen[key][subkey] = tuple(set_val)
            return frozen

        # WARNING: both neighbors and relation_args must not have default initialization.
        # Default init is dangerous, because we sometimes perform uniform sampling over
        # all keys in the dictionary. This distribution will get altered if a user asks about
        # entities or relations that weren't present.

        # self.neighbors[start][relation] = (end1, end2, ...)
        # self.relation_args[relation][position] = (ent1, ent2, ...)
        # position is either 's' (domain) or 't' (range)
        self.neighbors = freeze(neighbors)
        self.relation_args = freeze(relation_args)
        self.random_entities = []

        # cpp_graph = graph_traversal.Graph()
        # for s, r, t in triples:
        #     cpp_graph.add_edge(s, r, t)
        #     cpp_graph.add_edge(t, invert(r), s)
        # self.cpp_graph = cpp_graph
        cpp_graph = None

    def shortest_path(self, source, target):
        # use breadth-first search

        queue = deque()
        explored = {}  # stores backpointers

        def enqueue(node, backpointer):
            queue.appendleft(node)
            explored[node] = backpointer

        def path(node):
            current = node
            path = [current]
            while True:
                backpointer = explored[current]
                if backpointer:
                    rel, current = backpointer
                    path.extend((rel, current))
                else:
                    break  # we've hit the source
            return path[::-1]  # reverse

        enqueue(source, None)

        while len(queue) != 0:
            current = queue.pop()
            for rel, nbrs in self.neighbors[current].items():
                for nbr in nbrs:
                    if nbr not in explored:
                        enqueue(nbr, (rel, current))
                    if nbr == target:
                        return path(nbr)


    def random_walk_probs(self, start, path):
        return self.cpp_graph.exact_random_walk_probs(start, list(path))

    def walk_all(self, start, path, positive_branch_factor=float('inf')):
        if positive_branch_factor == 0:
            return set()

        approx = positive_branch_factor != float('inf')

        if approx:
            return set(self.cpp_graph.approx_path_traversal(start, list(path), positive_branch_factor))
        else:
            return set(self.cpp_graph.path_traversal(start, list(path)))

    def is_trivial_query(self, start, path):
        return self.cpp_graph.is_trivial_query(start, list(path))

    def type_matching_entities(self, path, position):
        if position == 's':
            r = path[0]
        elif position == 't':
            r = path[-1]
        else:
            raise ValueError(position)

        try:
            if not inverted(r):
                return self.relation_args[r][position]
            else:
                inv_pos = 's' if position == 't' else 't'
                return self.relation_args[invert(r)][inv_pos]
        except KeyError:
            # nothing type-matches
            return tuple()

    # TODO: test this
    def random_walk(self, start, length, no_return=False):
        """
        If no_return, the random walk never revisits the same node. Can sometimes return None, None.
        """
        max_attempts = 1000
        for i in range(max_attempts):

            sampled_path = []
            visited = set()
            current = start
            for k in range(length):
                visited.add(current)

                r = random.choice(list(self.neighbors[current].keys()))
                sampled_path.append(r)

                candidates = self.neighbors[current][r]

                if no_return:
                    current = utils.sample_excluding(candidates, visited)
                else:
                    current = random.choice(candidates)

                # no viable next step
                if current is None:
                    break

            # failed to find a viable walk. Try again.
            if current is None:
                continue

            return tuple(sampled_path), current

        return None, None

    def random_walk_constrained(self, start, path):
        """
        Warning! Can sometimes return None.
        """

        # if start node isn't present we can't take this walk
        if start not in self.neighbors:
            return None

        current = start
        for r in path:
            rels = self.neighbors[current]
            if r not in rels:
                # no viable next steps
                return None
            current = random.choice(rels[r])
        return current

    def random_entity(self):
        if len(self.random_entities) == 0:
            self.random_entities = list(np.random.choice(list(self.neighbors.keys()), size=20000, replace=True))
        return self.random_entities.pop()

    def relation_stats(self):
        stats = defaultdict(dict)
        rel_counts = Counter(r for s, r, t in self.triples)

        for r, args in self.relation_args.items():
            out_degrees, in_degrees = [], []
            for s in args['s']:
                out_degrees.append(len(self.neighbors[s][r]))
            for t in args['t']:
                in_degrees.append(len(self.neighbors[t][invert(r)]))

            domain = float(len(args['s']))
            range = float(len(args['t']))
            out_degree = np.mean(out_degrees)
            in_degree = np.mean(in_degrees)
            stat = {'avg_out_degree': out_degree,
                    'avg_in_degree': in_degree,
                    'min_degree': min(in_degree, out_degree),
                    'in/out': in_degree / out_degree,
                    'domain': domain,
                    'range': range,
                    'r/d': range / domain,
                    'total': rel_counts[r],
                    'log(total)': np.log(rel_counts[r])
                    }

            # include inverted relation
            inv_stat = {'avg_out_degree': in_degree,
                        'avg_in_degree': out_degree,
                        'min_degree': stat['min_degree'],
                        'in/out': out_degree / in_degree,
                        'domain': range,
                        'range': domain,
                        'r/d': domain / range,
                        'total': stat['total'],
                        'log(total)': stat['log(total)']
                        }

            stats[r] = stat
            stats[invert(r)] = inv_stat

        return stats