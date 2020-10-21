# this wrapper is based on the state builder from malmo-challenge
# https://github.com/microsoft/malmo-challenge/
from gameai.utils.common import Entity
from malmoenv.core import Env
from gym.spaces import Box
import cv2, gym
import numpy as np
import json

#
RGB_PALETTE = {
    'sand': [255, 225, 150],
    'clay': [99, 66, 33],
    'grass': [44, 176, 55],
    'lapis_block': [190, 190, 255],
    'Agent_1': [255, 0, 0],
    'Agent_2': [0, 0, 255],
    'Pig': [185, 126, 131]
}

GRAY_PALETTE = {
    'sand': 255,
    'clay': 230,
    'grass': 200,
    'lapis_block': 150,
    'Agent_1': 100,
    'Agent_2': 50,
    'Pig': 0,
    'brick_block': 2
}

# wrapper to extract symbolic representation from malmo
class SymbolicObs(Env):
    def __init__(self, env, shape=(11, 11), gray=True):
        # Env specific variables
        super(SymbolicObs, self).__init__(env)
        self.env = env
        self.shape = shape
        self.observation_space = Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = env.action_space

        # wrapper specific variables
        self.palette = GRAY_PALETTE if bool(gray) else RGB_PALETTE


    def step(self, action):
        # todo https://github.com/microsoft/malmo-challenge/blob/a1dec75d0eb4cccc91f9d818a4ecae5fa1ac906f/ai_challenge/pig_chase/environment.py
        # link above seem to be useful for this wrapper
        print("stepping, action = {}".format(action))
        obs, r, done, info = self.env.step(action)
        obs = self.extract_representation(info)

        return obs, r, done, info

    def reset(self):
        # todo reset won't work in current form has to get info on reset
        obs = self.env.reset()
        return np.zeros(self.shape)
        # return self.extract_representation(obs)

    def extract_representation(self, info):
        # todo convert from json to array
        symb_state = json.loads(info["raw_info"])
        board = np.reshape(symb_state["board"], newshape=self.shape)
        entities = symb_state["entities"]

        # todo check if this is needed or not - can overlap objects
        # for entity in entities:
        #     board[int(entity['z']+1), int(entity['x'])] += f"/{entity['name']}"

        obs_shape = board.shape + (3, )
        obs = np.zeros(obs_shape, dtype=np.float32)

        # todo in malmo challenge they used 4 entries per pixel

        # convert Minecraft yaw to 0=north, 1=west etc.
        agent = symb_state["entities"][0]
        agent_direction = ((((int(agent['yaw']) - 45) % 360) // 90) - 1) % 4

        # todo iterate over the whole board and map entries to colours
        it = np.nditer(board, flags=["multi_index"])
        while not it.finished:
            board_entity = it.value
            mapped_value = self.palette[str(board_entity)]
            obs[it.multi_index[0], it.multi_index[1]] = mapped_value
            it.iternext()

        # todo add the entities on top of the board

        # agent = symb_state["entities"][0]
        # chicken = symb_state["entities"][1]

        return obs