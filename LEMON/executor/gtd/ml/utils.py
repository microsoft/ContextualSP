import os
from collections import defaultdict, deque, OrderedDict
from contextlib import contextmanager
from os.path import join
import logging

import tensorflow as tf
import time
from keras import backend as K


class TensorDebugger(object):
    """Debug your TensorFlow model.

    EXAMPLE BELOW:

    tf.reset_default_graph()
    tdb = TensorDebugger()

    # define a graph, register some nodes
    x = tf.placeholder(tf.int32, name='x')

    zs = []
    for k in range(10):
        y = tf.constant(k, name='y')
        tdb.register('y', y)  # register with any name you want; we just called it 'y'
        z = x * y
        zs.append(z)

    g = tf.constant(12, name='g')
    h = tf.constant(10, name='h')
    tdb.register('g', g, force_run=True)
    tdb.register('h', h)

    total = tf.reduce_sum(tf.pack(zs))

    sess = tf.InteractiveSession()
    fetches = [total]
    feed_dict = {x: 10}

    # replace your sess.run recall with tdb.debug
    # result = sess.run(fetches, feed_dict)
    result, bp_results = tdb.debug(sess, fetches, feed_dict)

    print 'result:', result
    print 'bp_results:', bp_results

    # result: [450]
    # bp_results: {'y': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'g': 12}

    # notice that we collected every value of 'y' in the for-loop
    # we didn't collect 'h', because it wasn't on the execution path to compute fetches
    # we collected 'g' even though it's not on execution path, because we marked it as force_run

    sess.close()
    """
    DEFAULT = None

    @classmethod
    def default(cls):
        if cls.DEFAULT is None:
            cls.DEFAULT = TensorDebugger()
        return cls.DEFAULT

    def __init__(self, g=None):
        self.name_to_nodes = defaultdict(list)
        self.name_to_placeholders = defaultdict(list)

        if g is None:
            self.g = tf.get_default_graph()

        self.namestack = []

    @property
    def dependency_graph(self):
        """Build a dependency graph.

        Returns:
            a dict. Each key is the name of a node (Tensor or Operation) and each value is a set of
            dependencies (other node names)
        """
        deps = defaultdict(set)
        for op in self.g.get_operations():
            # the op depends on its input tensors
            for input_tensor in op.inputs:
                deps[op.name].add(input_tensor.name)
            # the op depends on the output tensors of its control_dependency ops
            for control_op in op.control_inputs:
                for output_tensor in control_op.outputs:
                    deps[op.name].add(output_tensor.name)
            # the op's output tensors depend on the op
            for output_tensor in op.outputs:
                deps[output_tensor.name].add(op.name)
        return deps

    def ancestors(self, op_name, deps):
        """Get all nodes upstream of the current node."""
        explored = set()
        queue = deque([op_name])
        while len(queue) != 0:
            current = queue.popleft()
            for parent in deps[current]:
                if parent in explored: continue
                explored.add(parent)
                queue.append(parent)
        return explored

    def register(self, name, node, force_run=False):
        """Register a name for a node.

        If multiple nodes are registered to the same name, TensorFlowDebugger
        saves all nodes as a list, in the order that they were registered.

        This is convenient if you want to register all nodes constructed in a for-loop
        under the same name. E.g.

        for k in range(10):
            x = tf.constant(k)
            debugger.register('x', x)
        """
        # TODO(kelvin): deal with SparseTensor
        placeholder = node.op.node_def.op == 'Placeholder'

        # TODO(kelvin): remove this hack
        node.force_run = force_run
        name_tuple = tuple(self.namestack + [name])

        if placeholder:  # deal with placeholders separately, because they can't be directly fetched
            lookup = self.name_to_placeholders
        else:
            lookup = self.name_to_nodes
        lookup[name_tuple].append(node)

        # TODO(kelvin): somehow make it optional to specify a name?
        # We could use introspection to get the name of the variable...
        return node

    @contextmanager
    def namescope(self, name):
        self.namestack.append(name)
        yield
        self.namestack.pop()

    def debug(self, sess, fetches, feed_dict):
        """Like Session.run, but also returns debug values.

        Args:
            sess: Session object
            fetches: same as Session.run
            feed_dict: same as Session.run

            Will ONLY compute values of breakpoints that are on the execution path defined by fetches.

        Returns:
            results: same as what's returned by Session.run
            bp_results: a dictionary mapping breakpoints to their values

            If a single breakpoint maps to multiple nodes, the "value" for that breakpoint
            will be a list of the values of all nodes mapping to that breakpoint,
            in the order that the nodes were registered.
        """
        single_fetch = not isinstance(fetches, list)

        # as a new list
        if single_fetch:
            fetches = [fetches]
        else:
            fetches = list(fetches)

        # compute all ancestors of the fetches
        deps = self.dependency_graph  # compute dependencies
        ancestors = set()
        for fetch in fetches:
            name = fetch if isinstance(fetch, str) else fetch.name
            anc = self.ancestors(name, deps)
            ancestors.update(anc)

        # give each fetch a name
        orig_fetch = '__orig_fetch__'
        names = [orig_fetch] * len(fetches)

        # add debug nodes to fetches
        for name, cand_nodes in self.name_to_nodes.items():
            # filter nodes by those on execution path
            nodes = []
            for cand in cand_nodes:
                if cand.name in ancestors or cand.force_run:
                    nodes.append(cand)

            fetches.extend(nodes)
            names.extend([name] * len(nodes))

        # get all values
        all_results = sess.run(fetches, feed_dict)

        # extract out results
        results = []  # original results
        bp_results = defaultdict(list)  # breakpoint results

        for name, result in zip(names, all_results):
            if name == orig_fetch:
                results.append(result)
            else:
                bp_results[name].append(result)

        # get placeholder values directly from feed_dict
        for name, placeholders in self.name_to_placeholders.items():
            for placeholder in placeholders:
                if placeholder in feed_dict:
                    key = placeholder
                elif placeholder.name in feed_dict:
                    key = placeholder.name
                else:
                    if placeholder.force_run:
                        raise ValueError("Tried to force-run {}, but no value provided.".format(placeholder.name))
                    continue  # placeholder wasn't fed
                bp_results[name].append(feed_dict[key])

        if single_fetch:
            results = results[0]

        # unwrap single-item lists and single-item name tuples
        unwrap = lambda l: l[0] if len(l) == 1 else l
        bp_results = {unwrap(k): unwrap(v) for k, v in bp_results.items()}

        return results, bp_results


class Saver(object):
    """A light wrapper around the TensorFlow Saver.

    This object is different in a few ways:
        - it has its save directory specified up front.
        - it is able to identify the latest checkpoint even if the directory was moved
            between the last save and reload.
        - it always uses the default session
    """
    def __init__(self, save_dir, *args, **kwargs):
        """Create a Saver.

        Args:
            save_dir (str): directory to save checkpoints
            args (list): args to pass to the tf.train.Saver constructor
            kwargs (dict): kwargs to pass to the tf.train.Saver constructor
        """
        self._save_dir = save_dir
        self._saver = tf.train.Saver(*args, **kwargs)
        self._prev_save_time = time.time()

    def save(self, step):
        """Save.

        Args:
            step (int): train step number
        """
        path = join(self._save_dir, 'weights')
        self._saver.save(tf.get_default_session(), path, step)
        self._prev_save_time = time.time()

    def interval_save(self, step, interval):
        """If more than specified interval of time has elapsed since last save, then save.

        Args:
            step (int): train step number
            interval (int): interval of time, in seconds.
        """
        if time.time() - self._prev_save_time >= interval:
            self.save(step)

    def restore(self, step_or_ckpt=None):
        """Restore variables.

        Args:
            step_or_ckpt (int|str): if int, restores the checkpoint associated with that step. If str, restores
                checkpoint at that path. If None, restores the latest checkpoint. Default is None.
        """
        if step_or_ckpt is None:
            ckpt = self.latest_checkpoint
        elif isinstance(step_or_ckpt, int):
            ckpt = self.checkpoint_paths[step_or_ckpt]
        elif isinstance(step_or_ckpt, str):
            ckpt = step_or_ckpt
        else:
            raise TypeError(step_or_ckpt)
        sess = tf.get_default_session()
        self._saver.restore(sess, ckpt)

    @property
    def checkpoint_paths(self):
        """A map from step number to checkpoint path.

        Returns:
            OrderedDict[int, str]
        """
        log_path = join(self._save_dir, 'checkpoint')
        if not os.path.exists(log_path):
            logging.warn('No checkpoint log found at {}'.format(log_path))
            return OrderedDict()  # without any checkpoint log, we assume that there are no checkpoints

        # load the checkpoints log
        with open(log_path, 'r') as f:
            logs = list(l.strip() for l in f)

        d = OrderedDict()
        for i, line in enumerate(logs):
            key, val = line.split(': ')
            if i == 0:
                assert key == 'model_checkpoint_path'
                continue  # this one is redundant
            else:
                assert key == 'all_model_checkpoint_paths'
            orig_path = val[1:-1]  # strip quotation marks
            _, f_name = os.path.split(orig_path)
            correct_path = join(self._save_dir, f_name)
            step = int(correct_path.split('-')[-1])
            d[step] = correct_path
        return d

    @property
    def latest_checkpoint(self):
        ckpts = self.checkpoint_paths
        if len(ckpts) == 0:  # what if there are no checkpoints
            raise IOError("No checkpoint to restore.")
        latest_step = max(ckpts.keys())
        return ckpts[latest_step]


class TensorBoardLogger(object):

    def __init__(self, log_dir):
        self.g = tf.Graph()
        self.summaries = {}
        self.sess = tf.Session(graph=self.g)
        self.summ_writer = tf.summary.FileWriter(log_dir, flush_secs=5)

    def log_proto(self, proto, step_num):
        """Log a Summary protobuf to the event file.
        Args:
            proto:  a Summary protobuf
            step_num: the iteration number at which this value was logged
        """
        self.summ_writer.add_summary(proto, step_num)
        return proto

    def log(self, key, val, step_num):
        """Directly log a scalar value to the event file.

        Args:
            key (string): a name for the value
            val: a float
            step_num: the iteration number at which this value was logged
        """
        try:
            ph, summ = self.summaries[key]
        except KeyError:
            # if we haven't defined a variable for this key, define one
            with self.g.as_default():
                ph = tf.placeholder(tf.float32, (), name=key)  # scalar
                summ = tf.summary.scalar(key, ph)
            self.summaries[key] = (ph, summ)

        summary_str = self.sess.run(summ, {ph: val})
        self.summ_writer.add_summary(summary_str, step_num)
        return val


def assert_shape(variable, shape):
    """Assert that a TensorFlow Variable has a particular shape.

    Args:
        variable: TF Variable
        shape: a TensorShape, Dimension or tuple
    """
    variable.get_shape().assert_is_compatible_with(shape)


def guarantee_initialized_variables(session, variables=None):
    """Guarantee that all the specified variables are initialized.

    If a variable is already initialized, leave it alone. Otherwise, initialize it.

    If no variables are specified, checks all variables in the default graph.

    Args:
        variables (list[tf.Variable])
    """
    name_to_var = {v.op.name: v for v in tf.global_variables() + tf.local_variables()}
    uninitialized_variables = list(name_to_var[name] for name in
                                   session.run(tf.report_uninitialized_variables(variables)))
    init_op = tf.variables_initializer(uninitialized_variables)
    session.run(init_op)
    return uninitialized_variables


def assert_broadcastable(low_tensor, high_tensor):
    low_shape = tf.shape(low_tensor)
    high_shape = tf.shape(high_tensor)

    low_rank = tf.rank(low_tensor)

    # assert that shapes are compatible
    high_shape_prefix = tf.slice(high_shape, [0], [low_rank])
    assert_op = tf.assert_equal(high_shape_prefix, low_shape, name="assert_shape_prefix")
    return assert_op


def expand_dims_for_broadcast(low_tensor, high_tensor):
    """Expand the dimensions of a lower-rank tensor, so that its rank matches that of a higher-rank tensor.

    This makes it possible to perform broadcast operations between low_tensor and high_tensor.

    Args:
        low_tensor (Tensor): lower-rank Tensor with shape [s_0, ..., s_p]
        high_tensor (Tensor): higher-rank Tensor with shape [s_0, ..., s_p, ..., s_n]

    Note that the shape of low_tensor must be a prefix of the shape of high_tensor.

    Returns:
        Tensor: the lower-rank tensor, but with shape expanded to be [s_0, ..., s_p, 1, 1, ..., 1]
    """
    orig_shape = tf.shape(low_tensor)
    orig_rank = tf.rank(low_tensor)
    target_rank = tf.rank(high_tensor)

    # assert that shapes are compatible
    assert_op = assert_broadcastable(low_tensor, high_tensor)

    with tf.control_dependencies([assert_op]):
        pad_shape = tf.tile([1], [target_rank - orig_rank])
        new_shape = tf.concat(0, [orig_shape, pad_shape])
        result = tf.reshape(low_tensor, new_shape)

    # add static shape information
    high_shape_static = high_tensor.get_shape()
    low_shape_static = low_tensor.get_shape()
    extra_rank = high_shape_static.ndims - low_shape_static.ndims

    result_dims = list(low_shape_static.dims) + [tf.Dimension(1)] * extra_rank
    result_shape = tf.TensorShape(result_dims)
    result.set_shape(result_shape)

    return result


def broadcast(tensor, target_tensor):
    """Broadcast a tensor to match the shape of a target tensor.

    Args:
        tensor (Tensor): tensor to be tiled
        target_tensor (Tensor): tensor whose shape is to be matched
    """
    rank = lambda t: t.get_shape().ndims
    assert rank(tensor) == rank(target_tensor)  # TODO: assert that tensors have no overlapping non-unity dimensions

    orig_shape = tf.shape(tensor)
    target_shape = tf.shape(target_tensor)

    # if dim == 1, set it to target_dim
    # else, set it to 1
    tiling_factor = tf.select(tf.equal(orig_shape, 1), target_shape, tf.ones([rank(tensor)], dtype=tf.int32))
    broadcasted = tf.tile(tensor, tiling_factor)

    # Add static shape information
    broadcasted.set_shape(target_tensor.get_shape())

    return broadcasted


def gather_2d(tensor, i, j):
    """2D version of tf.gather.

    The equivalent in Numpy would be tensor[i, j, :]

    Args:
        tensor (Tensor): a Tensor of rank at least 2
        i (Tensor): row indices
        j (Tensor): column indices of same shape as i, or broadcastable to the same shape
    """
    rank = lambda t: t.get_shape().ndims

    # get static shape info
    assert rank(tensor) >= 2
    assert rank(i) == rank(j)
    dims_static = tensor.get_shape()

    # get dynamic shape info
    shape = tf.shape(tensor)
    dims = tf.split(0, rank(tensor), shape)
    rows, cols = dims[:2]

    new_dims = [rows * cols] + dims[2:]
    new_shape = tf.concat(0, new_dims)
    tensor_flat = tf.reshape(tensor, new_shape)

    # annotate with static shape
    new_shape_static = tf.TensorShape([dims_static[0] * dims_static[1]] + list(dims_static[2:]))
    tensor_flat.set_shape(new_shape_static)

    k = i * cols + j
    vals = tf.gather(tensor_flat, k)
    return vals


@contextmanager
def clean_session():
    """Create a new Graph, bind the graph to a new Session, and make that session the default."""
    graph = tf.Graph()  # create a fresh graph
    with tf.Session(graph=graph) as sess:
        K.set_session(sess)  # bind Keras
        yield sess