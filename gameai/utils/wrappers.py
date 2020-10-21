from malmoenv.core import Env
from gym.spaces import Box
import cv2, gym
import numpy as np
import json

# Template for creating a Malmo wrapper
class DummyWrapper(Env):
    def __init__(self, env):
        super(DummyWrapper, self).__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, r, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, obs):
        return obs

# Wrapper to convert observations to a desired shape
class DownsampleObs(Env):
    def __init__(self, env, shape):
        super(DownsampleObs, self).__init__(env)
        self.env = env
        self.shape = shape # env.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=self.shape + env.observation_space.shape[2:], dtype=np.uint8)
        self.action_space = env.action_space

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.score += r
        self.steps += 1
        obs = self.observation(obs)
        info = {"raw_info": info}
        if done:
            print(f"episode finished in {self.steps} and with score {self.score}")
        return obs, r, done, info

    def reset(self):
        obs = self.env.reset()
        self.steps = 0
        self.score = 0
        return self.observation(obs)

    def observation(self, obs):
        if obs is None:
            print("obs is None")
        else:
            obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
            if obs.ndim == 2:
                obs = np.expand_dims(obs, -1)
            obs = obs / 255.0
        return obs