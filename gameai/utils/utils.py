import argparse, os
import malmoenv
from pathlib import Path
from gameai.utils.wrappers import DownsampleObs

def parse_args():
    parser = argparse.ArgumentParser(description='malmoenv arguments')
    parser.add_argument('--mission', type=str, default='../MalmoEnv/missions/mobchase_single_agent.xml',
                        help='the mission xml')
    parser.add_argument('--port', type=int, default=10000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    # todo next 2 arguments can be removed
    # https://github.com/elpollouk/malmo/blob/elpollouk/MultiAgentEnv/MalmoEnv/rllib_train.py
    parser.add_argument('--port2', type=int, default=None,
                        help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    # parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()

    if args.server2 is None:
        args.server2 = args.server
    # Better to use absolute path for XML files in case working directory would change
    args.mission = os.path.realpath(args.mission)

    return args

def create_env(args):
    xml = Path(args.mission).read_text()
    env = malmoenv.make()
    print(f"create env listening on port {args.port}")
    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync,
             reshape=True)
    env.reward_range = (-float('inf'), float('inf'))
    # env = DownsampleObs(env, shape=tuple((84, 84)))
    # env = MultiEntrySymbolicObs(env)
    return env