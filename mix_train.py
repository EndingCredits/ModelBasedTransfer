from __future__ import division
import argparse
import copy
import os
import random
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import NECAgent



#TODO: Split this into a separate agent initiation of agent and env and training
def run_agent(args, games):
  training=True
  training_iters = args.training_iters
  display_step = args.display_step

  # Launch the graph
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  with tf.Session(config=config) as sess:

    # Set precision for printing numpy arrays, useful for debugging
    #np.set_printoptions(threshold='nan', precision=3, suppress=True)

    # Create environments
    import gym
    try:
        import gym_vgdl #This can be found on my github if you want to use it.
    except:
        pass
    pairs = []
    
    for game in games:
        env = gym.make(game)
        agent = build_agent(sess, env, game, args)
        
        pairs.append((env, agent))
        
    # Initialize all tensorflow variables
    sess.run(tf.global_variables_initializer())
    
    discrim_losses = {}
    
    for env, agent in pairs:
        try:
            agent.Load('chk/'+ agent.name, True, False, True)
        except:
            pass
        
        state = env.reset()
        agent.Reset(state, training)
        
        for _, a in pairs:
            discrim_losses[(agent,a)] = []
        
    try:
        agent.Load('chk', False, True, False)
    except:
        pass
    
    #for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #    print v

    for step in tqdm(range(training_iters), ncols=80):

        env, agent = random.choice(pairs)
        
        # Act, and add 
        action, value = agent.GetAction()
        state, reward, terminal, info = env.step(action)
        agent.Update(action, reward, state, terminal)
        
        if agent.started_training:
          for i in range(1):
            a = agent ; t = 1.0
            if np.random.rand() > 0.5:
              while a == agent:
                _, a = random.choice(pairs)
              t = 0.0
            batch_size = args.batch_size
            if len(a.visited_states) > batch_size:
              idx_batch = set(random.sample(range(
                  len(list(a.visited_states)[-1000:])), batch_size))
              embs = [val for i, val in enumerate(list(a.visited_states)[-1000:]) if i in idx_batch] 
              l = agent._train_discrim(embs, t)
              discrim_losses[(agent, a)].append(np.mean(l))

        if terminal:
            # Reset agent and environment
            state = env.reset()
            agent.Reset(state, training)


        # Display Statistics
        if step % display_step == 0 and step != 0:
            tqdm.write("{}, {:>7}/{}it"\
                .format(time.strftime("%H:%M:%S"), step, training_iters))
            for _, a in pairs:
                ep_rewards = a.ep_rewards
                av_loss = np.mean(a.losses[-1000:])
                av_g_loss = np.mean(a.gen_losses[-1000:])
                av_d_loss = np.mean(a.discrim_losses[-1000:])
                num_eps = len(ep_rewards)
                if num_eps > 0:
                  avr_ep_reward = np.mean(ep_rewards[-25:])
                  max_ep_reward = np.max(ep_rewards[-25:])
                else:
                  max_ep_reward = avr_ep_reward = 0
                tqdm.write("{:>40}: tot_eps: {:4.1f}, "\
                .format(a.name, num_eps)
                +"25_ep_avr: {:4.1f}, 25_ep_max: {:4.1f}, "\
                .format(avr_ep_reward, max_ep_reward)
                +"m_loss: {:4.3f}, g_loss: {:4.3f}, d_loss: {:4.3f}"\
                .format(av_loss, av_g_loss, av_d_loss))
                
                to_print = " "*42
                for _, a_ in pairs:
                    sim = np.mean(discrim_losses[(a,a_)][-1000:])
                    to_print = to_print + "{:4.3f}, ".format(sim)
                tqdm.write(to_print)
                
                if training:
                    a.Save('chk/'+ a.name)
                np.save('chk/'+ a.name + '/states.npy', a.visited_states)
            a.Save('chk', False, True, False)

                
    
def build_agent(sess, env, name, args):
    agent_args = copy.deepcopy(args)
    agent_args.env = name
    shape = env.observation_space.shape
    if len(shape) is 3: mode = 'image'
    elif shape[0] is None: mode = 'object'
    else: mode = 'vanilla'
    agent_args.num_actions = env.action_space.n #only works with discrete action spaces

    # Set agent variables
    if mode=='DQN':
        agent_args.model = 'CNN'
        agent_args.preprocessor = 'deepmind'
        agent_args.obs_size = [84,84]
        agent_args.history_len = 4
    elif mode=='image':
        agent_args.model = 'CNN'
        agent_args.preprocessor = 'grayscale'
        agent_args.obs_size = shape[:-1]#list(env.observation_space.shape)[:2]
        agent_args.history_len = 2
    elif mode=='object':
        agent_args.model = 'object'
        agent_args.preprocessor = 'default'
        agent_args.obs_size = shape#list(env.observation_space.shape)
        agent_args.history_len = 0
    elif mode=='vanilla':
        agent_args.model = 'nn'
        agent_args.preprocessor = 'default'
        agent_args.obs_size = shape#list(env.observation_space.shape)
        agent_args.history_len = 0

    # Create agent
    return NECAgent.NECAgent(sess, agent_args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', type=str, default='roms/pong.bin',
                       help='Location of rom file')
    parser.add_argument('--env', type=str, default=None,
                       help='Gym environment to use')
    parser.add_argument('--model', type=str, default=None,
                       help='Leave None to automatically detect')
                       
    parser.add_argument('--unity_test', type=int, default=0,
                       help='Run unity test')

    parser.add_argument('--seed', type=int, default=123,
                       help='Seed to initialise the agent with')

    parser.add_argument('--training_iters', type=int, default=5000000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=5000,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--learning_rate', type=float, default=0.00005,
                       help='Learning rate for TD updates')
    parser.add_argument('--model_learning_rate', type=float, default=0.0005,
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
                       

    parser.add_argument('--save_file', type=str, default=None,
                       help='Name of save file (leave None for no saving)')

    args = parser.parse_args()
    
    if args.epsilon_final == None: args.epsilon_final = args.epsilon
    if args.epsilon_anneal == None: args.epsilon_anneal = args.training_iters

    arg_dict = vars(args)
    print(' ' + '_'*33 + ' ')
    print('|' + ' '*16 + '|' + ' '*16  + '|')
    for i in arg_dict:
        print "|{:>15} | {:<15}|".format(i, arg_dict[i])
    print('|' + '_'*16 + '|' + '_'*16  + '|')
    print('')
    
    games = [ #'CartPole-v0',
             'vgdl_aliens-v0',
             'vgdl_aliens_2-v0',
             #'vgdl_aliens_flipped-v0',
             #'vgdl_aliens_objects-v0',
             'vgdl_boulderdash-v0',
             'vgdl_boulderdash_2-v0',
             #'vgdl_boulderdash_flipped-v0',
             #'vgdl_boulderdash_objects-v0',
             #'vgdl_missilecommand-v0',
             #'vgdl_missilecommand_2-v0',
             #'vgdl_missilecommand_objects-v0',
             'vgdl_zelda-v0',
             'vgdl_zelda_2-v0',
             #'vgdl_zelda_flipped-v0',
             #'vgdl_zelda_objects-v0',
           ]
           
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
           
    run_agent(args, games)

