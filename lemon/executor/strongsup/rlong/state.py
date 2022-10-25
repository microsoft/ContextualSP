from abc import ABCMeta, abstractmethod


class RLongState(object, metaclass=ABCMeta):
    """Represents a row of objects, each of which has various properties.

    Used in:
    - RLongWorld as the initial state
    - RLongDenotation as the current state during execution
    - RLongValue as the final state
    """
    __slots__ = ['_objects']

    def __init__(self, objects):
        """Create a new RLongState.

        Args:
            objects (list).
        """
        self._objects = objects

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self._objects == other._objects)

    def __hash__(self):
        return hash(self._objects)

    def __repr__(self):
        return ' '.join(repr(x) for x in self._objects)

    def __getitem__(self, i):
        return self._objects[i]

    def __len__(self):
        return len(self._objects)

    def dump_human_readable(self, fout):
        """Dump a human-readable representation to a file object.
        By default, print repr(self).
        """
        print(self, file=fout)

    @property
    def objects(self):
        return self._objects

    @property
    def all_objects(self):
        return self._objects

    @classmethod
    def from_raw_string(cls, raw_string):
        """Create a new RLongState from dataset string.
        This is a CLASS METHOD.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_join(self, value, prop):
        """Return the result of joining the property with the value.

        Args:
            value: Property value
            prop (str): Property name
        Returns:
            A result (object)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_double_join(self, value1, value2, prop):
        """Return the result of joining the property with 2 values.

        Args:
            value1: Property value
            value2: Property value
            prop (str): Property name
        Returns:
            A result (object)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action, stack):
        """Apply an action and return the new state.
        Relevant arguments should be popped from the stack.

        Args:
            action (str)
            stack (list)
        Returns:
            (new_state, history_entry)
            new_state (RLongState): State after the action is applied
            history_entry (tuple): An entry to be added to history
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_argument(self, argument):
        """Return a RLongObject that corresponds to the history argument
        but with the current properties.

        Args:
            argument (RLongObject)
        Returns:
            RLongObject
        """
        raise NotImplementedError

    def reverse_action(self, action):
        """(Optional method) Return the reversed action.

        Args:
            action (str)
        Returns:
            reversed action (str)
        """
        raise NotImplementedError


class RLongObject(object):
    __slots__ = ()
    # just a marker class


################################
# Helper methods

def get_single_object(stack_entry):
    if isinstance(stack_entry, list):
        assert len(stack_entry) == 1, 'Cannot operate on > 1 objects'
        return stack_entry[0]
    return stack_entry


################################
# Alchemy domain

class RLongAlchemyObject(tuple, RLongObject):
    __slots__ = ()

    def __new__(self, position, chemicals):
        """Create a new RLongAlchemyObject.

        Args:
            position (int): Position of the beaker (starting with 1)
            chemicals (str): The beaker's content.
                Each character represents 1 unit of chemical of that color.
                An empty string represents an empty beaker.
        """
        color = (None if not chemicals
                or any(x != chemicals[0] for x in chemicals)
                else chemicals[0])
        return tuple.__new__(RLongAlchemyObject, (position, chemicals, color))

    @property
    def position(self):
        """Return the beaker's position (int)."""
        return self[0]

    @property
    def chemicals(self):
        """Return the beaker's content (str).
        Each character represents 1 unit of chemical of that color.
        An empty string represents an empty beaker.
        """
        return self[1]

    @property
    def color(self):
        """If the beaker is not empty and has homogeneous content,
        return the beaker's chemical color (1-character str).
        Otherwise, return None.
        """
        return self[2]

    @property
    def amount(self):
        """Return the amount of chemical (int)."""
        return len(self[1])

    def __repr__(self):
        return '{}:{}'.format(self.position, self.chemicals or '_')


class RLongAlchemyState(RLongState):
    """State for alchemy domain.
    Properties: position, color, amount
    Actions: pour, mix, drain
    """
    __slots__ = ()

    @classmethod
    def from_raw_string(cls, raw_string):
        """Create a new RLongAlchemyState from dataset string.

        Format for each object: {position}:{chemicals}
        """
        objects = []
        for raw_object in raw_string.split():
            raw_position, raw_chemicals = raw_object.split(':')
            objects.append(RLongAlchemyObject(
                int(raw_position),
                '' if raw_chemicals == '_' else raw_chemicals))
        return cls(objects)

    def apply_join(self, value, prop):
        if prop == 'Color':
            return [x for x in self._objects if x.color == value]
        else:
            raise ValueError('Unknown property {}'.format(prop))

    def apply_double_join(self, value1, value2, prop):
        raise ValueError('Unknown property {}'.format(prop))

    def apply_action(self, action, stack):
        if action == 'Pour':
            # Object Object Pour
            target_pos = get_single_object(stack.pop()).position
            source_pos = get_single_object(stack.pop()).position
            assert source_pos != target_pos, \
                    'Cannot pour: Source and target are the same'
            target = self._objects[target_pos - 1]
            source = self._objects[source_pos - 1]
            assert source.color is not None, \
                    'Cannot pour: Source does not have a pourable content'
            assert source.amount + target.amount <= 4, \
                    'Cannot pour: Overflow'
            new_objects = self._objects[:]
            new_objects[target_pos - 1] = RLongAlchemyObject(
                    target_pos, target.chemicals + source.chemicals)
            new_objects[source_pos - 1] = RLongAlchemyObject(source_pos, '')
            return type(self)(new_objects), ('Pour', source, target)
        elif action == 'Mix':
            # Object Mix; the chemical becomes brown
            target_pos = get_single_object(stack.pop()).position
            target = self._objects[target_pos - 1]
            assert target.amount, \
                    'Cannot mix: No content'
            assert target.color is None, \
                    'Cannot mix: The content is already homogeneous'
            new_objects = self._objects[:]
            new_objects[target_pos - 1] = RLongAlchemyObject(
                    target_pos, 'b' * target.amount)
            return type(self)(new_objects), ('Mix', target)
        elif action == 'Drain':
            # Object Number Drain
            drain_amount = stack.pop()
            target_pos = get_single_object(stack.pop()).position
            target = self._objects[target_pos - 1]
            assert target.amount, \
                    'Cannot drain: No content'
            new_objects = self._objects[:]
            if isinstance(drain_amount, str) and drain_amount[0] == 'X':
                # Fraction
                numer, denom = int(drain_amount[1]), int(drain_amount[3])
                assert target.amount % denom == 0, \
                        'Cannot drain: Invalid fraction'
                drain_amount = int(target.amount * numer / denom)
            assert (isinstance(drain_amount, int)
                    and 0 < drain_amount <= target.amount), \
                    'Cannot drain: Invalid drain amount'
            remaining = target.amount - drain_amount
            new_objects[target_pos - 1] = RLongAlchemyObject(
                    target_pos, target.chemicals[:remaining])
            return type(self)(new_objects), ('Drain', target, drain_amount)
        else:
            raise ValueError('Unknown action {}'.format(action))

    def resolve_argument(self, argument):
        # Beaker is uniquely determined by position
        return self._objects[argument.position - 1]


################################
# Scene Domain

class RLongSceneObject(tuple, RLongObject):
    __slots__ = ()

    def __new__(self, position, shirt, hat, id_):
        """Create a new RLongSceneObject.
        An empty space is not an object.

        Args:
            position (int): Position of the person (starting with 1)
            shirt (str): The shirt color.
            hat (str): The hat color. Special color `e` means no hat.
            id_ (int): The hidden ID used when retrieving with H1, H2, ...
        """
        return tuple.__new__(RLongSceneObject, (position, shirt, hat, id_))

    @property
    def position(self):
        """Return the person's position (int)."""
        return self[0]

    @property
    def shirt(self):
        """Return the shirt color (str)."""
        return self[1]

    @property
    def hat(self):
        """Return the hat color (str)."""
        return self[2]

    @property
    def apparent(self):
        """Return the non-ID part."""
        return self[:3]

    @property
    def id_(self):
        """Return the ID (int)."""
        return self[3]

    def __repr__(self):
        return '{}:{}{}'.format(self.position,
                self.shirt or '_', self.hat or '_')


class RLongSceneState(RLongState):
    """State for the scene domain.
    Properties: position, shirt, hat
    Actions: create, delete, move, swaphat
    """
    STAGE_LENGTH = 10
    __slots__ = ['_next_id']

    def __init__(self, objects, next_id):
        """Create a new RLongSceneState.

        Args:
            objects (list).
            next_id (int): The next available object ID.
        """
        RLongState.__init__(self, objects)
        self._next_id = next_id

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and len(self._objects) == len(other._objects)
                and all(self._objects[i].apparent == other._objects[i].apparent
                    for i in range(len(self._objects))))

    @classmethod
    def from_raw_string(cls, raw_string):
        """Create a new RLongSceneState from dataset string.

        Format for each object: {position}:{shirt}{hat}
        """
        objects = []
        id_ = 0
        for raw_object in raw_string.split():
            raw_position, raw_colors = raw_object.split(':')
            if raw_colors != '__':
                objects.append(RLongSceneObject(
                    int(raw_position), raw_colors[0],
                    'e' if raw_colors[1] == '_' else raw_colors[1],
                    id_))
                id_ += 1
        return cls(objects, id_)

    def get_object_with_id(self, id_):
        target = [x for x in self._objects if x.id_ == id_]
        assert target, 'No object matching ID'
        assert len(target) == 1, 'Multiple objects matching ID'
        return target[0]

    def apply_join(self, value, prop):
        if prop == 'Shirt':
            return [x for x in self._objects if x.shirt == value]
        elif prop == 'Hat':
            return [x for x in self._objects if x.hat == value]
        elif prop == 'Left':
            target_id = get_single_object(value).id_
            target = self.get_object_with_id(target_id)
            assert target.position > 1, \
                    'Cannot call left on leftmost person'
            return target.position - 1
        elif prop == 'Right':
            target_id = get_single_object(value).id_
            target = self.get_object_with_id(target_id)
            assert target.position < self.STAGE_LENGTH, \
                    'Cannot call right on rightmost person'
            return target.position + 1
        else:
            raise ValueError('Unknown property {}'.format(prop))

    def apply_double_join(self, value1, value2, prop):
        if prop == 'ShirtHat':
            return [x for x in self._objects if x.shirt == value1
                    and x.hat == value2]
        else:
            raise ValueError('Unknown property {}'.format(prop))

    def apply_action(self, action, stack):
        if action == 'Leave':
            # Object Leave
            target_id = get_single_object(stack.pop()).id_
            target = self.get_object_with_id(target_id)
            new_objects = [x for x in self._objects if x.id_ != target_id]
            return type(self)(new_objects, self._next_id), \
                    ('Leave', target)
        elif action == 'SwapHats':
            # Object Object SwapHats
            target1_id = get_single_object(stack.pop()).id_
            target2_id = get_single_object(stack.pop()).id_
            assert target1_id != target2_id, \
                    'Cannot swap hats: Two targets are the same'
            target1 = self.get_object_with_id(target1_id)
            target2 = self.get_object_with_id(target2_id)
            new_objects = []
            for x in self._objects:
                if x.id_ == target1_id:
                    new_objects.append(RLongSceneObject(
                        x.position, x.shirt, target2.hat, x.id_))
                elif x.id_ == target2_id:
                    new_objects.append(RLongSceneObject(
                        x.position, x.shirt, target1.hat, x.id_))
                else:
                    new_objects.append(x)
            return type(self)(new_objects, self._next_id), \
                    ('SwapHats', target1, target2)
        elif action == 'Move':
            # Object Number Move
            new_pos = stack.pop()
            assert isinstance(new_pos, int), \
                    'Cannot move: Position is not an integer'
            if new_pos < 0:
                new_pos = self.STAGE_LENGTH + 1 + new_pos
            assert all(x.position != new_pos for x in self._objects), \
                    'Cannot move: Target position is occupied'
            target_id = get_single_object(stack.pop()).id_
            target = self.get_object_with_id(target_id)
            assert target.position != new_pos, \
                    'Cannot move: Target and source positions are the same'
            new_objects = []
            for x in self._objects:
                if x == target:
                    new_objects.append(RLongSceneObject(
                        new_pos, x.shirt, x.hat, x.id_))
                else:
                    new_objects.append(x)
            new_objects.sort(key=lambda x: x.position)
            return type(self)(new_objects, self._next_id), \
                    ('Move', target, new_pos)
        elif action == 'Create':
            # Number Color Color|e Create
            hat = stack.pop()
            shirt = stack.pop()
            new_pos = stack.pop()
            assert isinstance(hat, str) and len(hat) == 1, \
                    'Cannot create: Invalid hat color'
            assert isinstance(shirt, str) and len(shirt) == 1, \
                    'Cannot create: Invalid hat color'
            assert isinstance(new_pos, int), \
                    'Cannot create: Position is not an integer'
            if new_pos < 0:
                new_pos = self.STAGE_LENGTH + 1 + new_pos
            assert all(x.position != new_pos for x in self._objects), \
                    'Cannot create: Target position is occupied'
            new_objects = self._objects[:]
            new_person = RLongSceneObject(
                new_pos, shirt, hat, self._next_id)
            new_objects.append(new_person)
            new_objects.sort(key=lambda x: x.position)
            return type(self)(new_objects, self._next_id + 1), \
                    ('Create', new_person)
        else:
            raise ValueError('Unknown action {}'.format(action))

    def resolve_argument(self, argument):
        # Person is uniquely determined by ID
        # If the person is on the stage, get its identity.
        for x in self._objects:
            if x.id_ == argument.id_:
                return x
        # The object is not in the scene
        return RLongSceneObject(0, argument.shirt, argument.hat, argument.id_)


################################
# Tangrams Domain

class RLongTangramsObject(tuple, RLongObject):
    __slots__ = ()

    def __new__(self, position, shape):
        """Create a new RLongTangramsObject.

        Args:
            position (int): Position of the tangram (starting with 1)
            shape (str): Shape ID.
        """
        return tuple.__new__(RLongTangramsObject, (position, shape))

    @property
    def position(self):
        """Return the person's position (int)."""
        return self[0]

    @property
    def shape(self):
        """Return the shape ID (str)."""
        return self[1]

    def __repr__(self):
        return '{}:{}'.format(self.position, self.shape)


class RLongTangramsState(RLongState):
    """State for the tangrams domain.
    Properties: position, shape
    Actions: add, delete, swap
    """
    __slots__ = ()

    @classmethod
    def from_raw_string(cls, raw_string):
        """Create a new RLongTangramsState from dataset string.

        Format for each object: {position}:{shape}
        """
        objects = []
        for raw_object in raw_string.split():
            raw_position, raw_shape = raw_object.split(':')
            objects.append(RLongTangramsObject(
                int(raw_position), raw_shape))
        return cls(objects)

    def get_object_with_shape(self, shape):
        target = [x for x in self._objects if x.shape == shape]
        assert target, 'No object matching shape'
        return target[0]

    def apply_join(self, value, prop):
        # Can only use indexing.
        raise ValueError('Unknown property {}'.format(prop))

    def apply_double_join(self, value1, value2, prop):
        raise ValueError('Unknown property {}'.format(prop))

    def apply_action(self, action, stack):
        if action == 'Add':
            # Number Object Add
            target_shape = get_single_object(stack.pop()).shape
            new_pos = stack.pop()
            assert isinstance(new_pos, int), \
                    'Cannot add: Position is not an integer'
            if new_pos < 0:
                new_pos = len(self._objects) + 2 + new_pos
            assert new_pos <= len(self._objects) + 1, \
                    'Cannot add: Position out of bound'
            new_tangram = RLongTangramsObject(new_pos, target_shape)
            new_objects = [new_tangram]
            for x in self._objects:
                assert x.shape != target_shape, \
                        'Cannot add: Repeated shape'
                if x.position < new_pos:
                    new_objects.append(x)
                else:
                    new_objects.append(RLongTangramsObject(
                        x.position + 1, x.shape))
            new_objects.sort(key=lambda x: x.position)
            return type(self)(new_objects), ('Add', new_pos, new_tangram)
        elif action == 'Swap':
            # Object Object Swap
            target1_shape = get_single_object(stack.pop()).shape
            target2_shape = get_single_object(stack.pop()).shape
            assert target1_shape != target2_shape, \
                    'Cannot swap: Two targets are the same'
            target1 = self.get_object_with_shape(target1_shape)
            target2 = self.get_object_with_shape(target2_shape)
            new_objects = []
            for x in self._objects:
                if x.shape == target1_shape:
                    new_objects.append(RLongTangramsObject(
                        x.position, target2.shape))
                elif x.shape == target2_shape:
                    new_objects.append(RLongTangramsObject(
                        x.position, target1.shape))
                else:
                    new_objects.append(x)
            return type(self)(new_objects), ('Swap', target1, target2)
        elif action == 'Remove':
            # Object Leave
            target_shape = get_single_object(stack.pop()).shape
            target = self.get_object_with_shape(target_shape)
            new_objects = []
            for x in self._objects:
                if x.position < target.position:
                    new_objects.append(x)
                elif x.position > target.position:
                    new_objects.append(RLongTangramsObject(
                        x.position - 1, x.shape))
            return type(self)(new_objects), ('Remove', target)
        else:
            raise ValueError('Unknown action {}'.format(action))

    def resolve_argument(self, argument):
        # Tangram is uniquely determined by shape
        # If the tangram is on the stage, get its identity.
        for x in self._objects:
            if x.shape == argument.shape:
                return x
        # The object is not in the scene
        return RLongTangramsObject(0, argument.shape)


class RLongUndogramsState(RLongTangramsState):
    """State for the tangrams domain, but supports HUndo.
    Properties: position, shape
    Actions: add, delete, swap
    """
    __slots__ = ()

    def apply_action(self, action, stack):
        new_state, command = RLongTangramsState.apply_action(self, action, stack)
        if action == 'Remove':
            # We also add position to the arguments to make it parallel to AAdd
            command = (command[0], command[1].position, command[1])
        return new_state, command

    def reverse_action(self, action):
        if action == 'Swap':
            return 'Swap'
        elif action == 'Remove':
            return 'Add'
        elif action == 'Add':
            return 'Remove'
        else:
            raise ValueError('Unknown action {}'.format(action))
