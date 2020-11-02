# this wrapper is based on the state builder from malmo-challenge
# https://github.com/microsoft/malmo-challenge/
from gameai.utils.common import Entity
from malmoenv.core import Env
from gym.spaces import Box
import cv2, gym
import numpy as np
import json

# Each entity in the environment requires a colour mapping to use with the wrapper
RGB_PALETTE = {
    'sand': [194, 178, 128],
    'clay': [127, 95, 63],
    'grass': [132, 192, 17],
    'lapis_block': [190, 190, 255],
    'Agent0': [255, 0, 0],
    'Agent1': [0, 0, 255],
    'Chicken': [246, 236, 133],
    'egg': [255, 245, 195],
    'Pig': [155, 192, 203],
    'brick_block': [132, 31, 39]
}

GRAY_PALETTE = {
    'sand': 255,
    'clay': 230,
    'grass': 200,
    'lapis_block': 150,
    'Agent0': 100,
    'Agent1': 50,
    'Chicken': 0,
    'egg': 15,
    'Pig': 30,
    'brick_block': 120
}

# wrapper to extract symbolic representation from malmo
class SymbolicObs(Env):
    def __init__(self, env, shape=(11, 11), gray=True):
        # Env specific variables
        super(SymbolicObs, self).__init__(env)
        self.env = env
        self.shape = shape
        self.channels = 1 if bool(gray) else 3
        self.observation_space = Box(low=0, high=255, shape=shape + (self.channels, ), dtype=np.uint8)
        self.action_space = env.action_space

        # wrapper specific variables
        self.palette = GRAY_PALETTE if bool(gray) else RGB_PALETTE


        self.last_obs = np.zeros(shape)


    def step(self, action):
        print("stepping, action = {}".format(action))
        obs, r, done, info = self.env.step(action)
        obs = self.extract_representation(info)

        self.last_obs = obs
        return obs, r, done, info

    def reset(self):
        # info is not received on reset, so an empty observation is sent instead to the agent
        obs = self.env.reset()
        self.last_obs = np.zeros(self.shape)
        return self.last_obs
        # return self.extract_representation(obs)

    def extract_representation(self, info):
        # occasionally MC returns empty info
        if len(info['raw_info']) <= 1:
            return self.last_obs
        symb_state = json.loads(info["raw_info"])
        board = np.reshape(symb_state["board"], newshape=self.shape)
        entities = symb_state["entities"]

        obs_shape = board.shape + (self.channels, )
        obs = np.zeros(obs_shape, dtype=np.float32)

        # convert Minecraft yaw to 0=north, 1=west etc.
        agent = symb_state["entities"][0]
        agent_direction = ((((int(agent['yaw']) - 45) % 360) // 90) - 1) % 4

        # iterate over the whole board and map entries to colours
        it = np.nditer(board, flags=["multi_index"])
        while not it.finished:
            board_entity = it.value
            mapped_value = self.palette[str(board_entity)]
            obs[it.multi_index[0], it.multi_index[1]] = mapped_value
            it.iternext()

        # in this version of the representation just add a small value to the agent, no information about the tile below
        for agent in entities:
            agent_x = int(agent['x'])
            agent_z = int(agent['z']) + 1

            # make sure that entity is within the given range
            if agent_x < 0 or agent_x >= self.shape[0]:
                break
            if agent_z < 0 or agent_z >= self.shape[1]:
                break

            original_tile = board[agent_z, agent_x]
            agent_value = self.palette[agent["name"]]
            if agent_direction == 0:
                # facing north
                agent_value += 10
            elif agent_direction == 1:
                # west
                agent_value += 20
            elif agent_direction == 2:
                # south
                agent_value += 30
            else:
                # east
                agent_value += 40

            obs[agent_z, agent_x] = agent_value

        obs = obs / 255.

        return obs

# logic is adopted from the malmo-challenge repo
# https://github.com/microsoft/malmo-challenge/blob/a1dec75d0eb4cccc91f9d818a4ecae5fa1ac906f/ai_challenge/pig_chase/environment.py
class MultiEntrySymbolicObs(Env):
    def __init__(self, env, shape=(11, 11), gray=False):
        # Env specific variables
        super(MultiEntrySymbolicObs, self).__init__(env)
        self.env = env
        self.raw_shape = shape
        self.channels = 1 if bool(gray) else 3
        self.observation_space = Box(low=0, high=255, shape=(shape[0]*2, shape[1]*2, self.channels), dtype=np.uint8)
        self.action_space = env.action_space

        # wrapper specific variables
        self.palette = GRAY_PALETTE if bool(gray) else RGB_PALETTE
        self._gray = gray

        # store last observation to avoid failures
        self.last_obs


    def step(self, action):
        print("stepping, action = {}".format(action))
        obs, r, done, info = self.env.step(action)
        obs = self.extract_representation(info)

        return obs, r, done, info

    def reset(self):
        # info is not received on reset, so an empty observation is sent instead to the agent
        obs = self.env.reset()
        return np.zeros(self.observation_space.shape)
        # return self.extract_representation(obs)

    def extract_representation(self, info):
        # occasionally MC returns empty info
        if len(info['raw_info']) <= 1:
            return self.last_obs
        symb_state = json.loads(info["raw_info"])
        board = np.reshape(symb_state["board"], newshape=self.raw_shape)
        entities = symb_state["entities"]

        buffer_shape = (board.shape[0] * 2, board.shape[1] * 2)
        if not self._gray:
            buffer_shape = buffer_shape + (self.channels,)
        buffer = np.zeros(buffer_shape, dtype=np.float32)


        it = np.nditer(board, flags=['multi_index', 'refs_ok'])

        # malmo challenge used 4 entries per pixel
        while not it.finished:
            entities_on_cell = str.split(str(board[it.multi_index]), '/')
            mapped_value = self.palette[entities_on_cell[0]]
            # draw 4 pixels per block
            buffer[it.multi_index[0] * 2:it.multi_index[0] * 2 + 2,
                   it.multi_index[1] * 2:it.multi_index[1] * 2 + 2] = mapped_value
            it.iternext()

        for agent in entities:
            agent_x = int(agent['x'])
            agent_z = int(agent['z']) + 1

            # make sure that entity is within the given range
            if agent_x < 0 or agent_x >= self.raw_shape[0]:
                break
            if agent_z < 0 or agent_z >= self.raw_shape[1]:
                break

            agent_pattern = buffer[agent_z * 2:agent_z * 2 + 2,
                                   agent_x * 2:agent_x * 2 + 2]

            # convert Minecraft yaw to 0=north, 1=west etc.
            agent_direction = ((((int(agent['yaw']) - 45) % 360) // 90) - 1) % 4

            if agent_direction == 0:
                # facing north
                agent_pattern[1, 0:2] = self.palette[agent['name']]
                agent_pattern[0, 0:2] += self.palette[agent['name']]
                agent_pattern[0, 0:2] /= 2.
            elif agent_direction == 1:
                # west
                agent_pattern[0:2, 1] = self.palette[agent['name']]
                agent_pattern[0:2, 0] += self.palette[agent['name']]
                agent_pattern[0:2, 0] /= 2.
            elif agent_direction == 2:
                # south
                agent_pattern[0, 0:2] = self.palette[agent['name']]
                agent_pattern[1, 0:2] += self.palette[agent['name']]
                agent_pattern[1, 0:2] /= 2.
            else:
                # east
                agent_pattern[0:2, 0] = self.palette[agent['name']]
                agent_pattern[0:2, 1] += self.palette[agent['name']]
                agent_pattern[0:2, 1] /= 2.

            buffer[agent_z * 2:agent_z * 2 + 2,
                   agent_x * 2:agent_x * 2 + 2] = agent_pattern

        # normalise the buffer
        obs = buffer / 255.

        return obs