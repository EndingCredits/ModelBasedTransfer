import argparse
import copy
import os
import tensorflow as tf
from main import run_agent

import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', type=str, default=None,
                       help='Location of rom file')
    parser.add_argument('--env', type=str, default=None,
                       help='Gym environment to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Leave None to automatically detect')
                       
    parser.add_argument('--unity_test', type=int, default=0,
                       help='Run unity test')

    parser.add_argument('--seed', type=int, default=123,
                       help='Seed to initialise the agent with')

    parser.add_argument('--training_iters', type=int, default=10000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=2500,
                       help='Number of iterations between parameter prints')
    parser.add_argument('--test_step', type=int, default=50000,
                       help='Number of iterations between tests')
    parser.add_argument('--test_count', type=int, default=5,
                       help='Number of test episodes per test')
    parser.add_argument('--do_tests', type=int, default=0,
                       help='Set 0 to skip tests')

    parser.add_argument('--learning_rate', type=float, default=0.00001,
                       help='Learning rate for TD updates')
    parser.add_argument('--model_learning_rate', type=float, default=0.00025,
                       help='Learning rate for forward model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                       help='Size of replay memory')
    parser.add_argument('--learn_step', type=int, default=4,
                       help='Number of steps in between learning updates')

    parser.add_argument('--memory_size', type=int, default=50000,
                       help='Size of DND dictionary')
    parser.add_argument('--num_neighbours', type=int, default=50,
                       help='Number of nearest neighbours to sample from the DND each time')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Alpha parameter for updating stored values')
    parser.add_argument('--delta', type=float, default=0.001,
                       help='Delta parameter for thresholding closeness of neighbours')

    parser.add_argument('--n_step', type=int, default=100,
                       help='Initial epsilon')
    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_final', type=float, default=None,
                       help='Final epsilon')
    parser.add_argument('--epsilon_anneal', type=int, default=None,
                       help='Epsilon anneal steps')

    args = parser.parse_args()

    args.env_type = 'gym'
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    
    if args.unity_test != 0:
        args.env_type = 'Unity'
        args.display_step = 5000
        args.do_tests = 0

    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters
    
    envs = [ 'CartPole-v0',
             'vgdl_aliens-v0',
             'vgdl_aliens_objects-v0',
             'vgdl_boulderdash-v0',
             'vgdl_boulderdash_objects-v0',
             'vgdl_missilecommand-v0',
             'vgdl_missilecommand_objects-v0',
             'vgdl_zelda-v0',
             'vgdl_zelda_objects-v0',
           ]
           
    random.shuffle(envs)
    
    for i in range(10):
      for env in envs:
        args_copy = copy.deepcopy(args)
        args_copy.env = env
        print(args_copy)
        run_agent(args_copy)
        tf.reset_default_graph()

