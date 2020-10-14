from malmoenv.core import Env
from gym.spaces import Box
import cv2, gym, os
import numpy as np
import json
from PIL import Image
import subprocess
import matplotlib.pyplot as plt

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

# wrapper to extract symbolic representation from malmo
class SymbolicObs(Env):
    def __init__(self, env, shape):
        super(SymbolicObs, self).__init__(env)
        self.env = env
        self.shape = shape # env.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=self.shape + env.observation_space.shape[2:], dtype=np.uint8)
        self.action_space = env.action_space
        self.board_shape = (11, 11)

    def step(self, action):
        # todo https://github.com/microsoft/malmo-challenge/blob/a1dec75d0eb4cccc91f9d818a4ecae5fa1ac906f/ai_challenge/pig_chase/environment.py
        # link above seem to be useful for this wrapper
        print("stepping, action = {}".format(action))
        obs, r, done, info = self.env.step(action)
        # print("step obs = {}".format(obs))
        obs = self.observation(obs)
        # info = {"raw_info": info}
        symb_state = json.loads(info["raw_info"])
        board = np.reshape(symb_state["board"], newshape=(11, 11))
        agent = symb_state["entities"][0]
        chicken = symb_state["entities"][1]
        return obs, r, done, info

    def reset(self):
        obs = self.env.reset()
        return self.env.reset()

    def observation(self, obs):
        if obs is None:
            print("obs is None")
        else:
            obs = cv2.resize(obs, self.shape[::-1], interpolation=cv2.INTER_AREA)
            if obs.ndim == 2:
                obs = np.expand_dims(obs, -1)
            # print("observation shape = {}".format(obs.shape))
            obs = obs / 255.0
        return obs

# Records a video and saves is as a gif
class VideoRecorder(Env):
    def __init__(self, env, savepath=""):
        super(VideoRecorder, self).__init__(env)
        self.env = env
        # self.shape = shape # env.observation_space.shape[:2]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.savepath = savepath
        self.fps = 30

        # store each frame in an array
        self.frames = []
        self.episode = 0
        self.reward = 0
        self.length = 0

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.reward += r
        self.length += 1
        self.frames.append(obs)
        if done:
            self.saveVideo()
            # Image.save(f"mob_chase_ep_{self.episode}_rew_{self.reward}_len_{self.length}.gif", save_all=True, append_images=self.frames)
        return obs, r, done, info

    def reset(self):
        self.episode += 1
        self.reward = 0
        self.length = 0
        obs = self.env.reset()
        self.frames.append(obs)
        return obs

    def saveVideo(self):
        # images are upside down, but they are correct
        # imgplot = plt.imshow(self.frames[0])
        # plt.show()
        filename = f"mob_chase_{self.episode}_{self.length}_{self.reward}.mp4"
        self.cmdline = ("ffmpeg",
                        '-nostats',
                        '-loglevel', 'error',  # suppress warnings
                        '-y',

                        # input
                        '-f', 'rawvideo',
                        '-s:v', '{}x{}'.format(*self.observation_space.shape[:2]),
                        '-pix_fmt', 'rgb24',
                        # '-framerate', '%d' % self.fps,
                        '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly

                        # output
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-r', '%d' % self.fps,
                        filename
                        )

        print('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

        for frame in self.frames:
            self.proc.stdin.write(frame.tobytes())

        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            print("VideoRecorder encoder exited with status {}".format(ret))

        # print(f"current working dir in save video {os.getcwd()}")
        # fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        # fps = 30
        # filename = f"mob_chase_{self.episode}_{self.length}_{self.reward}"
        # out = cv2.VideoWriter(filename, fourcc, fps, (self.env.observation_space.shape[:2]))
        # for frame in self.frames:
        #     out.write(frame)
        # out.release()