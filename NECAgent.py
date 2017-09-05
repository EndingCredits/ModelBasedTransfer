from __future__ import division

import numpy as np
import tensorflow as tf
import scipy#.misc.imresize
#import cv2

import knn_dictionary


class NECAgent():
    def __init__(self, session, args):

        self.full_train = True


        # Environment details
        self.obs_size = list(args.obs_size)
        self.n_actions = args.num_actions
        self.viewer = None

        # Agent parameters
        self.discount = args.discount
        self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal

        # DND parameters
        self.DND_size = args.memory_size
        self.delta = args.delta
        self.dict_delta = args.delta#0.1
        self.alpha = args.alpha
        self.number_nn = args.num_neighbours

        # Training parameters
        self.model_type = args.model
        self.history_len = args.history_len
        self.memory_size = args.replay_memory_size
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.learn_step = args.learn_step

        # Stored variables
        self.step = 0
        self.started_training = False
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.losses = []
        self.session = session

        # Replay Memory
        self.memory = ReplayMemory(self.memory_size, self.obs_size)

        # Preprocessor:
        if args.preprocessor == 'deepmind':
            self.preproc = deepmind_preprocessor
        elif args.preprocessor == 'grayscale':
            #incorrect spelling in order to not confuse those silly americans
            self.preproc = greyscale_preprocessor
        else:
            self.preproc = default_preprocessor
            #a lambda could be used here, but I think this makes more sense
        

        # Tensorflow variables:
        
        # Model for Embeddings
        
        if self.model_type == 'CNN':
            from networks import deepmind_CNN
            state_dim = [None, self.history_len] + self.obs_size
            model = deepmind_CNN
        elif self.model_type == 'nn':
            from networks import feedforward_network
            state_dim = [None] + self.obs_size
            model = feedforward_network
        elif self.model_type == 'object':
            from networks import object_embedding_network
            state_dim = [None] + self.obs_size
            model = object_embedding_network
            
        self.state = tf.placeholder("float", state_dim)
        self.post_state = tf.placeholder("float", state_dim)
        
        with tf.variable_scope('agent_model'):
            self.state_embeddings = model(self.state)
        with tf.variable_scope('agent_model', reuse=True):
            poststate_embeddings = model(self.post_state)
            
        self.action = tf.placeholder("uint8", [None])
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
        
        
        self.model_truth = tf.placeholder("float", [None])
        with tf.variable_scope('forward_model'):
            from networks import diff_network
            self.model_prediction = diff_network(self.state_embeddings,
                poststate_embeddings, action_one_hot)
        log_pos = tf.log( tf.clip_by_value(self.model_prediction, 1e-10, 1.0) )
        log_neg = tf.log( tf.clip_by_value(1.0 - self.model_prediction, 1e-10, 1.0) )
        cross_entropy = self.model_truth * log_pos + (1-self.model_truth) * log_neg
        self.model_loss = -cross_entropy
        

        # DNDs
        self.DND = knn_dictionary.q_dictionary(
          self.DND_size, self.state_embeddings.get_shape()[-1], self.n_actions,
          self.dict_delta, self.alpha)


        # Retrieve info from DND dictionary
        embs_and_values = tf.py_func(self.DND._query,
          [self.state_embeddings, self.action, self.number_nn], [tf.float64, tf.float64])
        self.dnd_embeddings = tf.to_float(embs_and_values[0])
        self.dnd_values = tf.to_float(embs_and_values[1])

        # DND calculation
        # (takes advantage of broadcasting)
        square_diff = tf.square(self.dnd_embeddings - tf.expand_dims(self.state_embeddings, 1))
        distances = tf.reduce_sum(square_diff, axis=2) + [self.delta]
        weightings = 1.0 / distances
        normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True)
        self.pred_q = tf.reduce_sum(self.dnd_values * normalised_weightings, axis=1)
        
        # Loss Function
        self.target_q = tf.placeholder("float", [None])
        self.td_err = self.target_q - self.pred_q
        total_loss = 500 * self.model_loss# + tf.square(self.td_err)
        if self.full_train:
          total_loss += tf.square(self.td_err)
        #total_loss = total_loss + become_skynet_penalty #commenting this out makes code run faster 
        
        self.embedding_weights = tf.get_collection(tf.GraphKeys.VARIABLES,scope='agent_model')    
        self.model_weights = tf.get_collection(tf.GraphKeys.VARIABLES, scope='forward_model')
        
        # Optimiser
        if self.full_train:
          self.optim = tf.train.AdamOptimizer(
            self.learning_rate)\
            .minimize(total_loss)
        else:
          self.optim = tf.train.AdamOptimizer(#, decay=0.9, epsilon=0.01)\
            self.learning_rate)\
            .minimize(total_loss, var_list=self.embedding_weights)
            
          # These are the optimiser settings used by DeepMind
          
        self.saver = tf.train.Saver(self.embedding_weights)
        self.model_saver = tf.train.Saver(self.model_weights)


    def _get_state(self, t=-1):
        # Returns the compiled state from stored observations
        if t==-1: t = self.trajectory_t-1

        if self.history_len == 0:
            state = self.trajectory_observations[t]
        else:
            if self.obs_size[0] == None:
                state = []
                for i in range(self.history_len):
                    state.append(self.trajectory_observations[t-i])
            else:
                state = np.zeros([self.history_len]+self.obs_size)
                for i in range(self.history_len):
                  if (t-i) >= 0:
                    state[i] = self.trajectory_observations[t-i]
        return state


    def _get_state_embeddings(self, states):
        # Returns the DND hashes for the given states
        if self.obs_size[0] == None:
            states_, masks = batch_objects(states)
            embeddings = self.session.run(self.state_embeddings,
              feed_dict={self.state: states_})#, self.masks: masks})
        else:    
            embeddings = self.session.run(self.state_embeddings, feed_dict={self.state: states})
        return embeddings


    def _predict(self, embedding):
        # Return action values for given embedding

        # calculate Q-values
        qs = []
        for a in xrange(self.n_actions):
            if self.DND.dicts[a].queryable(self.number_nn):
                q = self.session.run(self.pred_q, feed_dict={
                  self.state_embeddings: [embedding], self.action: [a]})[0]
            else:
                q = 0.0
            qs.append(q)

        # Return Q values
        return qs


    def _train(self, states, actions, Q_targets, poststates, correct):

        if not self.DND.queryable(self.number_nn):
            return False

        self.started_training = True

        if self.obs_size[0] == None:
            states, _ = batch_objects(states)
            poststates, _ = batch_objects(poststates)
        
        feed_dict = {
            self.state: states,
            self.target_q: Q_targets,
            self.action: actions,
            self.post_state: poststates,
            self.model_truth: [correct]
        }

        loss, _ = self.session.run([self.model_loss, self.optim], feed_dict=feed_dict)
        
        self.losses.append(loss)
        
        return True


    def Reset(self, obs, train=True):
        self.training = train

        #TODO: turn these lists into a proper trajectory object
        self.trajectory_observations = [self.preproc(obs)]
        self.trajectory_embeddings = []
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_t = 0
        return True

    def GetAction(self):
        # TODO: Perform calculations on Update, then use aaved values to select actions
        
        # Get state embedding of last stored state
        state = self._get_state()
        embedding = self._get_state_embeddings([state])[0]
        
        # Rendering code for displaying raw state
        if False:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            im = state[-1]
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=np.uint8)
            ret[:, :, :] = im[:, :, np.newaxis]*255
            self.viewer.imshow(ret)

        # Get Q-values
        Qs = self._predict(embedding)
        action = np.argmax(Qs) ; value = Qs[action]

        # Get action via epsilon-greedy
        if True: #self.training:
          if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.n_actions)
            #value = Qs[action] # Paper uses maxQ, uncomment for on-policy updates

        self.trajectory_embeddings.append(embedding)
        self.trajectory_values.append(value)
        return action, value


    def Update(self, action, reward, obs, terminal=False):

        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_t += 1
        self.trajectory_observations.append(self.preproc(obs))

        self.step += 1

        if self.training:

            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final

            if self.memory.count > self.batch_size*2 and (self.step % self.learn_step) == 0:
                # Get transition sample from memory
                s, a, R, p = self.memory.sample(self.batch_size, self.history_len)
                threshold = 0.5
                c=1.0
                if np.random.rand() > threshold:
                    p, _, _, _ = self.memory.sample(self.batch_size, self.history_len)
                    c=0.0
                # Run optimization op (backprop)
                self._train(s, a, R, p, c)


            # Add to replay memory and DND
            if terminal:
                # Calculate returns
                returns = []
                for t in xrange(self.trajectory_t):
                    if self.trajectory_t - t > self.n_steps:
                        #Get truncated return
                        start_t = t + self.n_steps
                        R_t = self.trajectory_values[start_t]
                    else:
                        start_t = self.trajectory_t
                        R_t = 0
                        
                    for i in xrange(start_t-1, t, -1):
                        R_t = R_t * self.discount + self.trajectory_rewards[i]
                    returns.append(R_t)
                    self.memory.add(self.trajectory_observations[t], self.trajectory_actions[t], R_t, (t==(self.trajectory_t-1)))
                if self.full_train:
                  self.DND.add(self.trajectory_embeddings, self.trajectory_actions, returns)
        return True
        
        
    def Save(self, save_dir):
        self.saver.save(self.session, save_dir + '/embedding.ckpt')
        self.model_saver.save(self.session, save_dir + '/model.ckpt')
        self.DND.save(save_dir + '/DNDdict')

    def Load(self, save_dir):
        #ckpt = tf.train.get_checkpoint_state(save_dir)
        #print("Loading model from {}".format(ckpt.model_checkpoint_path))
        #self.saver.restore(self.session, save_dir + '/embedding.ckpt')
        self.model_saver.restore(self.session, save_dir + '/model.ckpt')
        self.DND.load(save_dir + '/DNDdict')


def batch_objects(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of the longest list,
    #   compiles the list into a single n x m x d array, and returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, masks


# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, memory_size, obs_size):
    self.memory_size = memory_size
    self.obs_size = obs_size

    if self.obs_size[0] == None:
        self.observations = [None]*self.memory_size
    else:
        self.observations = np.empty([self.memory_size]+self.obs_size, dtype = np.float16)
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.returns = np.empty(self.memory_size, dtype = np.float16)
    self.terminal = np.empty(self.memory_size, dtype = np.bool_)

    self.count = 0
    self.current = 0

  def add(self, obs, action, returns, terminal):
    self.observations[self.current] = obs
    self.actions[self.current] = action
    self.returns[self.current] = returns
    self.terminal[self.current] = terminal

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _get_state(self, index, seq_len):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    if seq_len == 0:
      state = self.observations[index]
    else:
      if self.obs_size[0] == None:
        state = []
        for i in range(seq_len):
          state.append(self.observations[index-i])
      else:
        state = np.zeros([seq_len]+self.obs_size)
        for i in range(seq_len):
          state[i] = self.observations[index-i]
    return state

  def _uninterrupted(self, start, final):
    if self.current in range(start+1, final):
        return False
    for i in range(start, final-1):
        if self.terminal[i] == True: return False
    return True

  def sample(self, batch_size, seq_len=0):
    # sample random indexes
    indexes = [] ; states = [] ; poststates = []
    watchdog = 0
    while len(indexes) < batch_size:
      while True:
        # find random index
        index = np.random.randint(1, self.count - 1)
        if seq_len is not 0:
          start = index-seq_len
          if not self._uninterrupted(start, index):
            continue
        break

      indexes.append(index)
      states.append(self._get_state(index, seq_len))
      poststates.append(self._get_state(index+1, seq_len))

    return states, self.actions[indexes], self.returns[indexes], poststates


class trajectory:
# Get_Action requires last 4 obs to make a prediction
# Observations need to be stored to make prediction
# Actions and rewards need to be stored to calculate returns
# Values can be stored to prevent need from computing them twice
# Embeddings can be stored to prevent need from computing them twice (only needed for NEC)
    def __init__(self, obs_size):
        self.obs_size = obs_size

        self.trajectory = []
        self.current_entry = trajectory_entry()
        self.t = 0

        self.trajectory_observations = [obs]
        self.trajectory_embeddings = []
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []

    def _get_entry(self, t):
        if t==-1 or t==self.t: return self.current_entry
        return self.trajectory[t]

    def step(self):
        self.trajectory.append(self.current_entry)
        self.current_entry = trajectory_entry()
        self.t += 1

    def get_state(self, t=-1, his_len=1):
        if t==-1: t = self.t

        state = np.zeros([self.history_len]+self.obs_size)
        for i in range(his_len):
            state[i] = self._get_entry(t-i).observation
        return state

    def get_returns(self, n_step):

        for t in xrange(self.trajectory_t):
            if self.trajectory_t - t > self.n_steps:
                #Get truncated return
                start_t = t + self.n_steps
                R_t = self.trajectory_values[start_t]
            else:
                start_t = self.trajectory_t 
                R_t = 0
                        
            for i in xrange(start_t-1, t, -1):
                R_t = R_t * self.discount + self.trajectory_rewards[i]

        return obs, embeddings, actions, returns


class trajectory_entry:
  observation = None
  embedding = None
  action = None
  reward = None
  value = None


# Preprocessors:
def default_preprocessor(state):
    return state

def greyscale_preprocessor(state):
    #state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)/255.
    state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return state

def deepmind_preprocessor(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    state = scipy.misc.imresize(state, (84,84))
    return state

