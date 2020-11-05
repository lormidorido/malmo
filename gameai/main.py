import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from MalmoEnv.utils.launcher import launch_minecraft
from gameai.utils.utils import parse_args, create_env
from gameai.utils.symbolic_wrappers import MultiEntrySymbolicObs, SymbolicObs
# from gameai.utils.wrappers import ScreenCapturer, DownsampleObs
import matplotlib.pyplot as plt


if __name__ == "__main__":
    args = parse_args()
    args.port = 10001
    # args.mission = '../MalmoEnv/missions/pig_chase.xml'
    args.mission = '../MalmoEnv/missions/gameai_task.xml'
    NUM_ENVS = 1
    EPISODES = 5

    # launch minecraft instances, that will be used later
    # launch_minecraft blocks until all instances are set up
    GAME_INSTANCE_PORTS = [args.port + i for i in range(NUM_ENVS)]
    launch_script = "./launchClient_quiet.sh" #"./launchClient_headless.sh" #
    # instances = launch_minecraft(GAME_INSTANCE_PORTS, launch_script=launch_script)

    # connects to the previously created instances
    env = create_env(args)
    # env = DownsampleObs(env, shape=(84, 84))
    # env = ScreenCapturer(env, size=(200, 200))
    # env = SymbolicObs(env, gray=False)
    env = MultiEntrySymbolicObs(env, shape=(7, 7))

    for i in range(EPISODES):
        obs = env.reset()
        steps = 0
        total_rewards = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            plt.imshow(obs)
            plt.show()
            steps += 1
            total_rewards += reward

            if done:
                print(f"Episode finished in {steps} with reward: {total_rewards} ")

            time.sleep(.05)

    # close envs
    env.close()
    for instance in instances:
        instance.communicate()

