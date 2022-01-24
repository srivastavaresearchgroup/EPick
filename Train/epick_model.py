"""
This code is based on https://github.com/mingzhaochina/unet_cea
"""
import os
import sys
import tensorflow as tf
import layers
import tensorflow_addons as tfa
import numpy as np

def model (inputs, num_classes, is_training):

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)

    ###========== Encoder Section ==============================================================
    conv0_1 = layers.conv_btn1(inputs, 3, 32, 'conv0_1', is_training  = is_training)
    conv0_2 = layers.conv_btn1(conv0_1, 3, 32, 'conv0_2', is_training = is_training)
    pool0   = layers.maxpool(conv0_2, 2,  'pool0') ## [4, 3000, 32]

    conv1_1 = layers.conv_btn1(pool0, 3, 32, 'conv1_1', is_training  = is_training)
    conv1_2 = layers.conv_btn1(conv1_1, 3, 32, 'conv1_2', is_training = is_training)
    pool1   = layers.maxpool(conv1_2, 4,  'pool1') # [4, 750, 32]

    conv2_1 = layers.conv_btn1(pool1,   3, 32, 'conv2_1', is_training = is_training)
    conv2_2 = layers.conv_btn1(conv2_1, 3, 32, 'conv2_2', is_training = is_training)
    pool2   = layers.maxpool(conv2_2, 4,   'pool2')
    drop2   = layers.dropout(pool2, dropout_keep_prob, 'drop2')  ## [4,188, 32]

    conv3_1 = layers.conv_btn1(drop2,   3, 64, 'conv3_1', is_training = is_training)
    conv3_2 = layers.conv_btn1(conv3_1, 3, 64, 'conv3_2', is_training = is_training)
    pool3   = layers.maxpool(conv3_2, 4,   'pool3')
    drop3   = layers.dropout(pool3, dropout_keep_prob, 'drop3') # [4, 47, 64]

    conv4_1 = layers.conv_btn1(drop3,   3, 64, 'conv4_1', is_training = is_training)
    conv4_2 = layers.conv_btn1(conv4_1, 3, 64, 'conv4_2', is_training = is_training)
    pool4   = layers.maxpool(conv4_2, 4,   'pool4')
    drop4   = layers.dropout(pool4, dropout_keep_prob, 'drop4') #[4, 12, 64]

    conv5_1 = layers.conv_btn1(drop4,   3, 128, 'conv5_1', is_training = is_training)
    conv5_2 = layers.conv_btn1(conv5_1, 3, 128, 'conv5_2', is_training = is_training)
    drop5   = layers.dropout(conv5_2, dropout_keep_prob, 'drop5') # [4, 12, 128]

    ### ======================================== Decoder part ======================================
    upsample61      = layers.deconv_upsample(drop5, 4,  'upsample6')
    upsample61      = tf.keras.layers.Cropping1D(cropping=((0, 1)))(upsample61)

    concat6_1 = tf.concat(values=[layers.maxpool(conv0_2, 128, 'down_0_2'), 
                                layers.maxpool(conv1_2,  64, 'down_1_2'), 
                                layers.maxpool(conv2_2,  16, 'down_2_2'), 
                                layers.maxpool(conv3_2,   4, 'down_3_2')], axis =2, name ='concat6_1') # [4, 47, 160]

    attention_layer_1 = tfa.layers.MultiHeadAttention(head_size = 2, num_heads= 2)
    attention_conv6_1 = attention_layer_1([concat6_1, conv4_2])

    concat6         = layers.concat(upsample61, attention_conv6_1, 'concat6')
    conv6_1   = layers.conv_btn1(concat6, 3, 128, 'conv6_1', is_training = is_training)
    drop6     = layers.dropout(conv6_1, dropout_keep_prob, 'drop6') # [4, 47, 128]

    upsample7     = layers.deconv_upsample(drop6, 4,  'upsample7')

    concat7_1 = tf.concat(values=[layers.maxpool(conv0_2, 32, 'down_0_2'), 
                                layers.maxpool(conv1_2,  16, 'down_1_2'), 
                                layers.maxpool(conv2_2,  4, 'down_2_2')], axis =2, name ='concat7_1') ## [4, 188, 96]
    attention_layer_2 = tfa.layers.MultiHeadAttention(head_size = 2, num_heads= 2)
    attention_conv7_1 = attention_layer_2([concat7_1, conv3_2])

    concat7         = layers.concat(upsample7, attention_conv7_1, 'concat7')
    conv7_1   = layers.conv_btn1(concat7,       3, 64, 'conv7_1', is_training = is_training)
    drop7     = layers.dropout(conv7_1, dropout_keep_prob, 'drop7') # (4, 188, 64)

    upsample81     = layers.deconv_upsample(drop7, 4,  'upsample8')
    upsample81    = tf.keras.layers.Cropping1D(cropping=((0, 2)))(upsample81)

    concat8_1 = tf.concat(values=[layers.maxpool(conv0_2, 8, 'down_0_2'), 
                                layers.maxpool(conv1_2,  4, 'down_1_2')], axis =2, name ='concat8_1') # (4, 750, 64)
    attention_layer_3 = tfa.layers.MultiHeadAttention(head_size = 2, num_heads= 2)
    attention_conv8_1 = attention_layer_3([concat8_1, conv2_2])

    concat8       = layers.concat(upsample81, attention_conv8_1, 'concat8')
    conv8_1 = layers.conv_btn1(concat8, 3, 32, 'conv8_1', is_training = is_training) # (4, 750, 32)

    upsample91     = layers.deconv_upsample(conv8_1, 4, 'upsample9')
    attention_layer_4 = tfa.layers.MultiHeadAttention(head_size = 2, num_heads= 2)
    attention_conv9_1 = attention_layer_4([layers.maxpool(conv0_2, 2, 'down_0_2'), conv1_2])
    
    concat9       = layers.concat(upsample91, attention_conv9_1,  'concat9')
    conv9_1 = layers.conv_btn1(concat9, 3, 32, 'conv9_1', is_training = is_training) # [4, 3000, 32]

    upsample101          = layers.deconv_upsample(conv9_1, 2, 'upsample10')
    attention_layer_5 = tfa.layers.MultiHeadAttention(head_size = 2, num_heads= 1)
    attention_conv10_1 = attention_layer_5([conv0_2, conv0_2])

    concat10       = layers.concat(upsample101, attention_conv10_1, 'concat10')
    conv10_1 = layers.conv_btn1(concat10, 3, 32, 'conv10_1', is_training = is_training) # [4, 6000, 32]

    score  = layers.conv(conv10_1, 1, num_classes, 'score', activation_fn = None)
    logits = tf.reshape(score, (-1, num_classes))
    return logits

def segmentation_loss(logits, labels, class_weights = None):
    label = tf.reshape(labels, [-1,3])
    labels=tf.argmax(label, 1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels = labels, logits = logits, name = 'segment_cross_entropy_per_example')
    if class_weights is not None:
        weights = tf.matmul(label, class_weights, a_is_sparse = True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)
    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')
    tf.compat.v1.summary.scalar("loss/segmentation", segment_loss)
    return segment_loss

def l2_loss():
    weights = [var for var in tf.compat.v1.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
    tf.compat.v1.summary.scalar("loss/weights", l2_loss)
    return l2_loss

def loss(logits, labels, weight_decay_factor, batch_size, image_size, class_weights = None):
    segment_loss = segmentation_loss(logits, labels, class_weights)
    total_loss   = segment_loss + weight_decay_factor * l2_loss()
    tf.compat.v1.summary.scalar("loss/total", total_loss)
    return total_loss

def accuracy(logits, labels):
    labels = tf.compat.v1.to_int64(labels)
    labels = tf.reshape(labels, [-1, 3])
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
    predicted_labels = tf.reshape(tf.argmax(labels, axis=1), [-1, 1])
    precision = tf.compat.v1.metrics.mean_per_class_accuracy(predicted_labels,predicted_annots,3)
    return precision

def recall(logits, labels):
        labels = tf.compat.v1.to_int64(labels)
        labels = tf.reshape(labels, [-1, 3])
                        # tf.argmax: Returns the index with the largest value across axes of a tensor
        predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
        predicted_labels = tf.reshape(tf.argmax(labels, axis=1), [-1, 1])
        recall=tf.metrics.recall(predicted_labels,predicted_annots,3)
                                        #precision, recall, f1 = score(predicted_annots, predicted_labels)
                                            #precision = score(predicted_annots, predicted_labels)
        return recall

def train(loss, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, global_step):
    decayed_learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, global_step, 
                            learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)
    # execute update_ops to update batch_norm weights
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer   = tf.compat.v1.train.AdamOptimizer(decayed_learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)
    tf.compat.v1.summary.scalar("learning_rate", decayed_learning_rate)
    return train_op

def predict(logits, batch_size, image_size):
    predicted_images = tf.reshape(tf.argmax(logits, axis = 1), [batch_size, image_size])
    return predicted_images
  
def predict2(logits, batch_size, image_size):
    logits = tf.nn.softmax(logits)

    predicted_images = tf.reshape(logits, [batch_size, image_size, 3])
    return predicted_images
