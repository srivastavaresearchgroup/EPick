"""
This code is based on https://github.com/mingzhaochina/unet_cea
"""
import os
import sys
from Dataprocess import data_pipeline as dp
import numpy as np
import tensorflow as tf
from Dataprocess import config as config
import argparse
import time
import glob
import epick_model
import matplotlib.pyplot as plt

args = None

def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """
    tf_record_pattern = os.path.join(args.tfrecords_dir, '*.%s' % type)
    data_files =tf.io.gfile.glob(tf_record_pattern)
    data_size = 0
    for fn in data_files:
        for record in tf.compat.v1.python_io.tf_record_iterator(fn):
            data_size += 1
    return data_files, data_size

def train():
    tf.compat.v1.set_random_seed(24)

    cfg = config.Config()
    cfg.batch_size = args.batch_size
    cfg.add = 1
    cfg.n_clusters = args.num_classes
    cfg.n_clusters += 1
    cfg.n_epochs = FLAGS.num_epochs

    data_files, data_size = load_datafiles(args.tfrecords_prefix)
    pos_pipeline = dp.DataPipeline(args.tfrecords_dir, cfg, True)

    waveforms = pos_pipeline.samples 
    labels = pos_pipeline.labels

    logits = epick_model.model(waveforms, args.num_classes, True) 
    accuarcy = epick_model.accuracy(logits, labels)

    #load class weights if available
    if args.class_weights :
        weights = [1,50,50 ] ## e.g [1,50,50]
        class_weight_tensor = tf.constant(weights, dtype=tf.float32, shape=[args.num_classes, 1])
    else:
        class_weight_tensor = None
    loss = epick_model.loss(logits, labels, args.weight_decay_rate, arga.batch_size, args.waveform_length, class_weight_tensor)
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = epick_model.train(loss,args.learning_rate,args.learning_rate_decay_steps,args.learning_rate_decay_rate, global_step)
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    session_manager = tf.compat.v1.train.SessionManager(local_init_op = tf.compat.v1.local_variables_initializer())
    sess = session_manager.prepare_session("", init_op = init_op, saver = saver, checkpoint_dir = args.checkpoint_dir)
    writer = tf.compat.v1.summary.FileWriter(args.checkpoint_dir + "/train_logs", sess.graph)
    merged = tf.compat.v1.summary.merge_all()
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess = sess, coord = coord)
    start_time = time.time()
    
    p_acc = []
    s_acc = []
    loss_list =[]
    try:
        while not coord.should_stop():
            step = tf.compat.v1.train.global_step(sess, global_step)
            _, loss_value, summary = sess.run([train_op, loss, merged])
            writer.add_summary(summary, step)

            if step % 5000 == 0:
                acc_seg_value = sess.run([accuarcy])
                epoch = step * args.batch_size / data_size
                duration = time.time() - start_time
                start_time = time.time()

                print('[PROGRESS]\t Epoch %d | Step %d | loss = %.2f | P. acc. =  %.3f \
                      | S. acc. =  %.3f | N. acc. =  %.3f | dur. = (%.3f sec)'\
                      % (epoch, step, loss_value, acc_seg_value[0][1][1],acc_seg_value[0][1][2], acc_seg_value[0][1][0],\
                         duration))
                p_acc.append(acc_seg_value[0][1][1])
                s_acc.append(acc_seg_value[0][1][2])
                loss_list.append(loss_value)
                
                print('[PROGRESS]\tSaving checkpoint')
                checkpoint_path = os.path.join(args.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)
            
    except tf.errors.OutOfRangeError:
        print('Stop training at %d epochs, %d steps.' % (args.num_epochs, step))

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        print('Checkpoint directory does not exist, creating directory: ' + os.path.abspath(args.checkpoint_dir))
        os.makedirs(args.checkpoint_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'training.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory', default='../')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'tfrecords')
    parser.add_argument('--checkpoint_dir', help = 'Checkpoints directory', default ='../')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 3)
    parser.add_argument('--class_weights', help = 'Weight per class for weighted loss.  [num_classes]',type=bool,default=False)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 1e-4)
    parser.add_argument('--learning_rate_decay_steps', help = 'Learning rate decay steps', type = int, default = 10000)
    parser.add_argument('--learning_rate_decay_rate', help = 'Learning rate decay rate', type = float, default = 0.9)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)
    parser.add_argument('--waveform_length', help = 'Waveform length', type = int, default = 6000)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 1)
    parser.add_argument('--num_epochs', help = 'Number of epochs', type = int, default = 1)
    args, unparsed = parser.parse_known_args()

    tf.compat.v1.app.run()
