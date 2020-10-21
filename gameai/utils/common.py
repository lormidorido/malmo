# variables and class taken from "malmo ai-challenge"
ENV_AGENT_NAMES = ['Agent_1', 'Agent_2']
ENV_TARGET_NAMES = ['Pig']
ENV_ENTITIES_NAME = ENV_AGENT_NAMES + ENV_TARGET_NAMES
ENV_ACTIONS = ["move 1", "turn -1", "turn 1"]
ENV_ENTITIES = 'entities'
ENV_BOARD = 'board'
ENV_BOARD_SHAPE = (9, 9)
ENV_INDIVIDUAL_REWARD = 5
ENV_CAUGHT_REWARD = 25

class ENV_AGENT_TYPES:
    RANDOM, FOCUSED, TABQ, DEEPQ, HUMAN, OTHER = range(0, 6)

class Entity(object):
    """ Wrap entity attributes """

    def __init__(self, x, y, z, yaw, pitch, name=''):
        self._name = name
        self._x = int(x)
        self._y = int(y)
        self._z = int(z)
        self._yaw = int(yaw) % 360
        self._pitch = int(pitch)

    @property
    def name(self):
        return self._name

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = int(value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = int(value)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        self._z = int(value)

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        self._yaw = int(value) % 360

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = int(value)

    @property
    def position(self):
        return self._x, self._y, self._z

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.position == other

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def create(cls, obj):
        return cls(obj['x'], obj['y'], obj['z'], obj['yaw'], obj['pitch'])