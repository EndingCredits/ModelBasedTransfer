import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def linear(x,
           output_size,
           initializer=tf.truncated_normal_initializer(0, 0.02),
           #tf.truncated_normal_initializer(0.0, 0.02),
           activation_fn=None,
           name='linear'):
           
  shape = x.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('weights', [shape[1], output_size], tf.float32,
        initializer=initializer)
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(0.0))

    out = tf.nn.bias_add(tf.matmul(x, w), b)

    if activation_fn != None:
      return activation_fn(out)
    else:
      return out
      
      
      
def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out

     

# Permutation invariant layer
def invariant_layer(x,
                    out_size,
                    context=None,
                    initializer=None,
                    activation_fn=None,
                    name='linear_set'):

    in_size = x.get_shape().as_list()[-1]
    if context is not None:
      context_size = context.get_shape().as_list()[-1]

    with tf.variable_scope(name) as vs:
      w_e = tf.get_variable('w_e', (in_size,out_size), tf.float32, initializer=initializer)
      if context is not None:
        w_c = tf.get_variable('w_c', (context_size,out_size), tf.float32, initializer=initializer)
      b = tf.get_variable('b', (out_size), tf.float32, initializer=tf.constant_initializer(0.0))
      
      element_part = tf.nn.conv1d(x, [w_e], stride=1, padding="SAME")
      if context is not None:
         context_part = tf.expand_dims(tf.matmul(context, w_c), 1)
      else:
         context_part = 0
      
      out = element_part + context_part + b
      
      if activation_fn != None:
          out = activation_fn(out)

    return out


def get_mask(x):
    # Returns a matrix with values set to 1 where elements aren't padding
    emb_sum = tf.reduce_sum(tf.abs(x), axis=-1, keep_dims=True)
    return 0.99 - tf.to_float(tf.equal(emb_sum, 0.0))


def mask_and_pool(embeds, mask, pool_type='max'):
    # Use broadcasting to multiply
    masked_embeds = tf.multiply(embeds, mask)

    # Pool using max pooling
    embed = tf.reduce_max(masked_embeds, axis=1)
    return embed
    
    
    
def flatten(input_):
    in_list = [x for x in input_ if x is not None]
    if type(in_list[0]) is list:
      in_list = [flatten(elem) for elem in in_list ]
    
    return tf.concat([ tf.reshape(elem, [-1]) for elem in in_list], axis=0)


