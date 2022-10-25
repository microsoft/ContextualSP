import glob
import os
from abc import ABCMeta, abstractproperty, abstractmethod
from collections import Sequence
from os.path import join

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.engine
from tensorflow import Tensor

from gtd.io import JSONPicklable, makedirs


class Batch(Sequence, metaclass=ABCMeta):
    """An immutable Sequence of Example objects."""

    @abstractproperty
    def uid(self):
        """An integer that uniquely identifies this batch."""
        pass

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return self.uid == other.uid


class Model(object):
    """A Model encapsulates a network of TensorFlow operations.

    Each Model typically implements some modular and reusable functionality, e.g. "feed forward network"
    or "LSTM" or "neural attention". A full system is constructed by composing together several Models to form one
    large Model.
    """
    pass


class Feedable(Model, metaclass=ABCMeta):
    """A Model that can be fed plain old Python objects (e.g. a list of strings) as input.

    A Feedable defines a function which converts input objects into numpy arrays, which can then be passed into the
    TensorFlow computation graph.
    """

    @abstractmethod
    def inputs_to_feed_dict(self, *args, **kwargs):
        """Convert inputs into a feed_dict that can be fed into Session.run.

        Args:
            args, kwargs: input arguments to this model.

        Returns:
            dict[Tensor, np.array]: a feed_dict is a dict mapping placeholders to their assignments (numpy arrays).
        """
        pass

    @classmethod
    def inputs_to_feed_dict_union(cls, models, *args, **kwargs):
        """Convenience method for merging the feed_dicts of several models which all take the same inputs.

        Args:
            models (list[Feedable])
        """
        feed_dict = {}
        for model in models:
            feed_dict.update(model.inputs_to_feed_dict(*args, **kwargs))
        return feed_dict

    def compute(self, fetch, *args, **kwargs):
        """Compute outputs, given inputs.

        Uses the current default Session for execution.

        Args:
            fetch: anything that can be fetched by Session.run.
            args, kwargs: input arguments, matching the arguments passed to feed_dict

        Returns:
            the result of Session.run
        """
        sess = tf.get_default_session()
        if sess is None:
            raise ValueError('No default TensorFlow Session registered.')
        feed = self.inputs_to_feed_dict(*args, **kwargs)
        results = sess.run(fetch, feed_dict=feed)
        return results


class Optimizable(Model, metaclass=ABCMeta):
    """A Model with a differentiable objective function."""

    @abstractproperty
    def objective_tensor(self):
        """A scalar Tensor that we will take gradients with respect to."""
        pass

    @property
    def gradients(self):
        """A map from Variable Tensors to their gradient Tensors."""
        try:
            return self._var_to_grad
        except AttributeError:
            optimizer = tf.train.GradientDescentOptimizer(0.01)  # we merely use this optimizer to identify gradients
            self._var_to_grad = {v: g for g, v in optimizer.compute_gradients(self.objective_tensor) if g is not None}

        return self._var_to_grad

    @property
    def variables(self):
        """The set of variables which affect the objective_tensor."""
        return set(self.gradients.keys())


class KerasModel(Feedable):
    """A Model that can be trained with Keras.

    A KerasModel explicitly declares its `output_tensors` and input `placeholders`.

    Using Keras:
    - Setup
        - Remember to configure Keras to use the TensorFlow backend
        - If you use Keras layers, you MUST bind Keras to a TensorFlow session before constructing layers.
        - see [this](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html) for more info.
    - Note that Keras Input layers return plain old TensorFlow placeholders
    - When initializing variables, do NOT use tf.initialize_all_variables(). This will overwrite the initialization
      performed by Keras. Instead, use the `gtd.ml.utils.guarantee_initialized_variables` function.
    - If you plan to use the KerasTrainer, your ENTIRE model must use Keras Layers from beginning to end. You cannot
      intersperse with TF Operations (Keras needs to propagate its own metadata).
    """
    @abstractproperty
    def placeholders(self):
        """Placeholders owned by this Model.

        Returns:
            list[Tensor]
        """
        pass

    @classmethod
    def placeholders_union(cls, models):
        """Convenience method for merging the placeholders of several models.

        Args:
            models (list[KerasModel])
        """
        phs = []
        for model in models:
            phs.extend(model.placeholders)
        return phs

    @abstractproperty
    def output_tensors(self):
        """Outputs of this model.

        Returns:
            list[Tensor]: a list of Tensors.
        """
        pass


class KerasObjective(KerasModel, metaclass=ABCMeta):
    """Specifies the loss functions for training a model, as well as how to assign values to label Placeholders."""

    @abstractproperty
    def losses(self):
        """List of losses.

        Returns:
            list[(Tensor, Tensor, Tensor)]: a list of (label, objective, metric) triples.
                e.g. (some_tensor, 'sparse_categorical_crossentropy', 'accuracy')
        """
        pass


class KerasTrainer(object):
    def __init__(self, model, objective, optimizer, batch_size, save_dir):
        """Create a KerasTrainer.

        Responsible for training, checkpointing weights, and restoring weights from disk.

        Args:
            model (KerasModel)
            objective (KerasObjective)
            optimizer: optimizer for Keras
            batch_size (int)
            save_dir (str)
        """
        self.model = model
        self.objective = objective
        self._batch_size = batch_size
        self._save_dir = save_dir

        labels, objectives, metrics = [list(seq) for seq in zip(*objective.losses)]

        self.inputs = model.placeholders
        self.outputs = labels

        with tf.name_scope('keras_trainer'):
            keras_model = keras.engine.Model(input=self.inputs, output=self.outputs)
            keras_model.compile(optimizer=optimizer, loss=objectives, metrics=metrics)

        self.keras_model = keras_model

    @property
    def batch_size(self):
        return self._batch_size

    def _vectorized_batches(self, batches):
        """Convert iterable of Batches into iterable of vectorized batches.

        Args:
            batches (Iterable[Batch])

        Returns:
            Iterable: iterable of feed_dicts.
        """
        for batch in batches:
            feed_x = self.model.inputs_to_feed_dict(batch)
            feed_y = self.objective.inputs_to_feed_dict(batch)
            X = [feed_x[i] for i in self.inputs]
            Y = [feed_y[o] for o in self.outputs]
            yield X, Y

    def train(self, train_batches, valid_batches, samples_per_epoch, nb_epoch, nb_val_samples, extra_callbacks=None):
        """Train the model.

        Automatically adds the following Keras callbacks:
            - ModelCheckpoint
            - EarlyStopping
            - TensorBoard

        Args:
            train_batches (Iterable[Batch]): an iterable of training Batches
            valid_batches (Iterable[Batch]): an iterable of validation Batches
            samples_per_epoch (int)
            nb_epoch (int): max number of epochs to train for
            nb_val_samples (int): number of samples for validation
            extra_callbacks (list): a list of additional Keras callbacks to run
        """
        checkpoint_path = join(self.checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)
        early_stopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        tboard = TensorBoard(self.tensorboard_dir, write_graph=False)

        callbacks = [checkpointer, early_stopper, tboard]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        train = self._vectorized_batches(train_batches)
        valid = self._vectorized_batches(valid_batches)

        self.keras_model.fit_generator(train, samples_per_epoch, nb_epoch,
                                       callbacks=callbacks,
                                       validation_data=valid, nb_val_samples=nb_val_samples
                                       )

    @property
    def save_dir(self):
        return self._save_dir

    @classmethod
    def get_checkpoint_dir(cls, save_dir):
        return join(save_dir, 'checkpoints')

    @classmethod
    def get_tensorboard_dir(cls, save_dir):
        return join(save_dir, 'tensorboard')

    @property
    def checkpoint_dir(self):
        p = self.get_checkpoint_dir(self.save_dir)
        makedirs(p)
        return p

    @property
    def tensorboard_dir(self):
        p = self.get_tensorboard_dir(self.save_dir)
        makedirs(p)
        return p

    @classmethod
    def get_checkpoint_paths(cls, save_dir):
        checkpoint_dir = cls.get_checkpoint_dir(save_dir)
        pattern = join(checkpoint_dir, '*.hdf5')
        return list(glob.iglob(pattern))

    @property
    def latest_checkpoint_path(self):
        checkpoint_paths = self.get_checkpoint_paths(self.save_dir)
        latest = max(checkpoint_paths, key=os.path.getctime)
        return latest

    def load_weights(self, path):
        self.keras_model.load_weights(path)