## this code is developed from https://github.com/mingzhaochina/unet_cea
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import numpy as np

def conv(inputs, kernel_size, num_outputs, name, stride_size = 1, padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: filter size
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)
    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.compat.v1.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1]
        kernel_shape   = [kernel_size, num_filters_in, num_outputs]
        stride_shape   = stride_size

        weights = tf.compat.v1.get_variable('weights', kernel_shape, tf.float32, tf.compat.v1.glorot_uniform_initializer())
        bias    = tf.compat.v1.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv1d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
    
def conv_btn(inputs, kernel_size, num_outputs, name, is_training = True, stride_size=1 , padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, length, channels]
        kernel_size: filter size
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        is_training: Boolean, in training mode or not
        stride_size: convolution stide
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)
    Outputs:
        outputs: Tensor, [batch_size, length+-,  num_outputs]
    """
    with tf.compat.v1.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1]
        kernel_shape   = [kernel_size, num_filters_in, num_outputs]
        stride_shape   = stride_size
        weights = tf.compat.v1.get_variable('weights', kernel_shape, tf.float32, tf.compat.v1.glorot_uniform_initializer())
        bias    = tf.compat.v1.get_variable('bias', num_outputs, tf.float32, tf.compat.v1.constant_initializer(0.0))
        conv    = tf.nn.conv1d(inputs, weights, stride_shape, padding = padding, dilations=1)
        outputs = tf.nn.bias_add(conv, bias)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs
    
def deconv_upsample(inputs, factor, name, padding = 'SAME', activation_fn = None):
    """
    Convolution Transpose upsampling layer with bilinear interpolation weights:
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        factor: Integer, upsampling factor
        name: String, scope name
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height * factor, width * factor, num_filters_in]
    """

    with tf.compat.v1.variable_scope(name):
        stride_shape   = factor
        input_shape    = tf.shape(inputs)
        num_filters_in = inputs.get_shape().as_list()[-1]
        output_shape   = tf.stack([input_shape[0], input_shape[1] * factor, num_filters_in])
        weights = bilinear_upsample_weights(factor, num_filters_in)
        outputs = tf.nn.conv1d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def bilinear_upsample_weights(factor, num_outputs):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization:
    ----------
    Args:
        factor: Integer, upsampling factor
        num_outputs: Integer, number of convolution filters

    Returns:
        outputs: Tensor, [kernel_size, kernel_size, num_outputs]
    """

    kernel_size = 2 * factor - factor % 2

    weights_kernel = np.zeros((kernel_size,
                               num_outputs,
                               num_outputs), dtype = np.float32)

    rfactor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = rfactor - 1
    else:
        center = rfactor - 0.5

    og = np.ogrid[:kernel_size]
    upsample_kernel = (1 - abs(og - center) / rfactor)

    for i in range(num_outputs):
        weights_kernel[:, i, i] = upsample_kernel

    init = tf.compat.v1.constant_initializer(value = weights_kernel, dtype= tf.float32)
    weights = tf.compat.v1.get_variable('weights', shape = weights_kernel.shape, dtype =tf.float32, initializer = init)
    return weights

def maxpool(inputs, kernel_size, name, padding = 'SAME'):
    """
    Max pooling layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding

    Returns:
        outputs: Tensor, [batch_size, height / kernelsize[0], width/kernelsize[1], channels]
    """

    kernel_shape = kernel_size
    outputs = tf.keras.layers.MaxPool1D(pool_size = kernel_shape,
            strides = kernel_shape, padding = padding, name = name)(inputs)
          
    return outputs

def dropout(inputs, keep_prob, name):
    """
    Dropout layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        keep_prob: Float, probability of keeping this layer
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    return tf.nn.dropout(inputs, rate= keep_prob, name = name)

def concat(inputs1, inputs2, name):
    """
    Concatente two tensors in channels
    """
    return tf.concat(values=[inputs1, inputs2],axis =2, name = name)
