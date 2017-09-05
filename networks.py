import numpy as np
import tensorflow as tf
import ops
from ops import linear, conv2d, flatten
from ops import invariant_layer, mask_and_pool


def deepmind_CNN(state, output_size=128):
    initializer = tf.truncated_normal_initializer(0, 0.1)
    activation_fn = tf.nn.relu
    
    state = tf.transpose(state, [0, 2, 3, 1])
    
    l1 = conv2d(state, 32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
    l2 = conv2d(l1, 64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
    l3 = conv2d(l2, 64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')

    shape = l3.get_shape().as_list()
    l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
      
    embedding = linear(l3_flat, output_size, activation_fn=activation_fn, name='l4')

    # Returns the network output, parameters
    return embedding


def feedforward_network(state, out_size=128):
    initializer = tf.truncated_normal_initializer(0, 0.1)
    activation_fn = tf.nn.relu

    l1 = linear(state, 64,
      activation_fn=activation_fn, name='l1')
    l2 = linear(state, 64,
      activation_fn=activation_fn, name='l2')

    embedding = linear(l2, out_size,
      activation_fn=activation_fn, name='l3')

    # Returns the network output, parameters
    return embedding


def embedding_network(state, mask):
    # Placeholder layer sizes
    d_e = [[128, 256]]
    d_o = [128]

    # Build graph:
    initial_elems = state

    # Embedding Part
    for i, block in enumerate(d_e):
        el = initial_elems
        for j, layer in enumerate(block):
            context = c if j==0 and not i==0 else None
            el = invariant_layer(el, layer, context=context, name='l'+str(i)+'_'+str(j))

        c = mask_and_pool(el, mask) # pool to get context for next block
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(d_o):
        fc = ops.linear(fc, layer, name='lO_' + str(i))
    
    # Output
    embedding = fc

    return embedding
    
    
def object_embedding_network(state, n_actions=128):
    mask = ops.get_mask(state)
    net = embedding_network(state, mask)
    out = ops.linear(net, n_actions, name='outs')
    return out

    
def diff_network(state_a, state_b, action):
    activation_fn = tf.nn.relu
    with tf.variable_scope('difference_network'):

        a = linear(state_a, 256,
          activation_fn=activation_fn, name='a')
          
        b = linear(state_b, 256,
          activation_fn=activation_fn, name='b')
        
        cont = a - b #state_a - state_b # a - b
        
        l1 = linear(cont, 256,
          activation_fn=activation_fn, name='hidden') + linear(action, 256,
          activation_fn=activation_fn, name='hidden_b')
        out = linear(l1, 1, name='out')
        out = tf.nn.sigmoid(out)
        
    return out
