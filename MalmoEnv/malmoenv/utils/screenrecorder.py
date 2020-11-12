import ffmpeg
from malmoenv.core import Env
import numpy as np
import cv2

class ScreenCapturer(Env):
    """
    Should use this before modifying the observation space with other wrappers
    - uses python-ffmpeg in the background

    """
    def __init__(self, env, format="gif", savepath="", name="malmo", size=None, accumulate_episodes=1):
        super(ScreenCapturer, self).__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.savepath = savepath
        self.fps = 6
        self.name = name
        self.in_size = env.observation_space.shape[:2]
        self.out_size = size if (size != None) else env.observation_space.shape[:2]
        if format not in ["gif", "mp4"]:
            raise Exception(f"Provided format: [{format}] is currently not supported")
        self.format = format
        self.accumulate_episodes = accumulate_episodes # number of episodes to collect before creating a frame

        # store each frame in an array
        self.frames = []
        self.episode = 0
        self.total_reward = 0
        self.total_length = 0

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.total_reward += r
        self.total_length += 1
        self.frames.append(self.process_frame(np.copy(obs)))
        if done and self.episode % self.accumulate_episodes == 0:
            self.save_gif()
        return obs, r, done, info

    def reset(self):
        self.episode += 1
        obs = self.env.reset()
        self.frames.append(self.process_frame(obs))
        return obs

    def process_frame(self, frame):
        # convert to np array
        # frame = np.array(frame*255, dtype=np.uint8)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def save_gif(self):
        reward = self.total_reward / self.accumulate_episodes
        length = self.total_length / self.accumulate_episodes
        filename = f"{self.name}_e{self.episode}_l{length:.2f}_r{reward:.2f}.{self.format}"
        process = (
            ffmpeg
            .input("pipe:", format='rawvideo', pix_fmt='rgb24', s=f"{self.in_size[1]}x{self.in_size[0]}")
            .output(filename, s=f"{self.out_size[1]}x{self.out_size[0]}", sws_flags="neighbor", r=self.fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        for frame in self.frames:
            process.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )
        process.stdin.close()
        process.wait()

        self.frames = []
        self.total_reward = 0
        self.total_length = 0