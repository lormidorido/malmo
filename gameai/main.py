import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from MalmoEnv.utils.launcher import launch_minecraft
from gameai.utils.utils import parse_args, create_env


if __name__ == "__main__":
    args = parse_args()
    NUM_ENVS = 1
    EPISODES = 5

    # launch minecraft instances, that will be used later
    # launch_minecraft blocks until all instances are set up
    GAME_INSTANCE_PORTS = [10001 + i for i in range(NUM_ENVS)]
    launch_script = "./launchClient_quiet.sh" #"./launchClient_headless.sh" #
    instances = launch_minecraft(GAME_INSTANCE_PORTS, launch_script=launch_script)

    # connects to the previously created instances
    env = create_env(args)

    for i in range(EPISODES):
        obs = env.reset()
        steps = 0
        total_rewards = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
            total_rewards += reward

            if done:
                print(f"Episode finished in {steps} with reward: {total_rewards} ")

            time.sleep(.05)

    # close envs
    env.close()
    for instance in instances:
        instance.communicate()

