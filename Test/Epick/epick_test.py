"""
This code is based on https://github.com/mingzhaochina/unet_cea
"""

import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import time
import glob
import statistics
from Dataprocess import data_pipeline as dp
from Dataprocess import config as config
from epick_model import model, accuracy, loss, predict2
import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import fnmatch, math
import itertools
from matplotlib.ticker import FuncFormatter
import math

# initial setting
args = None

def result(predict_images, t_labels):
    TE = FE = 0     ## TE: true earthquake and FE: false earthquake
    TN = FN = 0     ## TN: true pure noise and FN: false noise
    Final_true = []
    Final_pre = []
    Final_p = [] # P index difference between true and predict
    Final_s = [] # S index difference between true and predict
    false_p = []
    false_s = []

    batch_size = predict_images.shape[0]
    for i in range(batch_size):
        '''sletect the one with largest probability given multipe P phases or S phases'''
        d_pp = []
        d_ss = []
        image_array = predict_images[i, :, :]  ## [image_size, 3]
        p1 = np.argmax(list(image_array[:, 1]))
        for j in range(image_array.shape[0]):
            if j!= p1:
                image_array[j][1] = 0
        s1 = np.argmax(list(image_array[:, 2]))
        for j in range(image_array.shape[0]):
            if j!= s1:
                image_array[j][2] = 0
        image_array = np.argmax(image_array, axis = 1)
        pre_labels = list(image_array) ## n =0, P = 1, S= 2
        
        true_n = list(t_labels[i, :, 0])        ## ture noise label
        true_p = list(t_labels[i, :, 1])        ## true p label
        true_s = list(t_labels[i, :, 2])        ## true s label
        
        ## true label
        f_true = [0] * 6000
        if true_n != [1]*6000:
            tp1 = true_p.index(1)
            ts1 = true_s.index(1)
            f_true[tp1] = 1
            f_true[ts1] = 2
            
        ## earthquake signal and pure noise (non-earthquake signal)
        if true_n == [1]*6000 and pre_labels ==  [0]*6000:
            TN += 1
        if true_n == [1]*6000 and pre_labels != [0]*6000:
            FE += 1
        if true_n != [1]*6000 and pre_labels == [0]*6000:
            FN += 1
            false_p.append(true_p.index(1))
            false_s.append(true_s.index(1))
        if true_n != [1]*6000 and pre_labels != [0]*6000 and pre_labels.count(1)==1 and pre_labels.count(2)==1:
            TE += 1     
            pp = []
            ss = []
            ## p-arrival
            if 1 not in pre_labels:
                pre_p = 10000
                pp.append(pre_p)
                d_pp.append(pre_p-tp1)
            else:
                pre_p = pre_labels.index(1)
                pp.append(pre_p)
                d_pp.append(pre_p-tp1)
                if abs(pre_p-tp1)<= 10: ## uncertainty interval: 10 samples
                    pre_labels[pre_p] = 0
                    pre_labels[tp1] = 1
                
            ## s-arrival
            if 2 not in pre_labels:
                pre_s = 10000
                ss.append(pre_s)
                d_ss.append(pre_s-ts1)
            else:
                pre_s = pre_labels.index(2)
                ss.append(pre_s)
                d_ss.append(pre_s-ts1)
                if abs(pre_s-ts1)<= 20:  ## uncertainty interval: 20 samples
                    pre_labels[pre_s] = 0
                    pre_labels[ts1] = 2

        Final_pre += pre_labels
        Final_true += f_true
        Final_p += d_pp
        Final_s += d_ss
            
    return TE, FE, TN, FN, Final_pre, Final_true, Final_p, Final_s, false_p, false_s

def test():
    tf.compat.v1.set_random_seed(24)
    cfg = config.Config()
    cfg.batch_size = args.batch_size
    cfg.add = 1
    cfg.n_clusters = args.num_classes
    cfg.n_clusters += 1
    cfg.n_epochs = 1

    pos_pipeline = dp.DataPipeline(args.tfrecords_dir, cfg, False)
    waveforms = pos_pipeline.samples 
    labels = pos_pipeline.labels   
    logits = model(waveforms, args.num_classes, False) 
    
    accuracy = accuracy(logits, labels)
    loss = loss(logits, labels, args.weight_decay_rate)
    prediction = predict2(logits, args.batch_size, args.waveform_length) 

    ## session initialization
    init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), 
                                 tf.compat.v1.local_variables_initializer())           
    sess = tf.compat.v1.Session()
    sess.run(init_op)
    saver = tf.compat.v1.train.Saver() 
    
    ## loading savd model
    if not tf.io.gfile.exists(args.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        saver.restore(sess, args.checkpoint_path)  # reload the trained model

    coord = tf.compat.v1.train.Coordinator() 
    threads = tf.compat.v1.train.start_queue_runners(sess = sess, coord = coord) 

    step = 0

    total_TE = total_FE = total_TN = total_FN = 0
    total_TP = total_FP = total_TNP = total_FNP = 0
    total_TS = total_FS = total_TNS = total_FNS = 0

    F_t =[]  ## for true_label
    F_p =[]  ## for predict_label

    Final_p_error = []
    Final_s_error = []
    False_p_error = []
    False_s_error = []

    try:
        while not coord.should_stop():
            loss_value, predicted_images_value, images_value,  ss_labels= sess.run([loss, predicted_images, images, labels])
            TE, FE, TN, FN, Final_pre, Final_true, Final_p, Final_s, false_p, false_s = result(predicted_images_value, ss_labels)
            total_TE += TE
            total_FE += FE
            total_TN += TN
            total_FN += FN

            F_p += Final_pre
            F_t += Final_true

            Final_p_error += Final_p
            Final_s_error += Final_s
            False_p_error += false_p
            False_s_error += false_s

            step+= 1

    except tf.errors.OutOfRangeError:
        print('Finishing in %d steps.' % step)
    finally:
        """When done, ask the threads to stop."""
        coord.request_stop()

    cm1= confusion_matrix([0, 0, 0, 0], [1, 1, 1, 1])  ## initialize
    cm1[0][0] = total_TN
    cm1[0][1] = total_FE
    cm1[1][0] = total_FN
    cm1[1][1] = total_TE
    multi_label1 = [ 'Non-earthquake', 'Earthquake']
    tick_marks = np.arange(len(multi_label1))
    plt.xticks(tick_marks, multi_label1, rotation=45)
    plt.yticks(tick_marks, multi_label1)
    thresh = cm1.max() / 2
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, "{:,}".format(cm1[i, j]),horizontalalignment="center", color="white" if cm1[i, j] > thresh else "black", fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Result of seismic event detection')
    plt.colorbar()
    plt.savefig('./earthquake_detection.jpg')
    plt.close()
    
    p_error = [int(i)/100.0 for i in Final_p_error] ## converting samples to second
    p_error_mean = sum(p_error)/len(p_error)
    p_error_std = statistics.stdev(p_error)
    
    s_error = [int(i)/100.0 for i in Final_s_error]
    s_error_mean = sum(s_error)/len(s_error)
    s_error_std = statistics.stdev(s_error)

    print("P-pick mean = ", p_error)
    print("P-pick standard deviation = ", p_error_std)
    print("S-pick mean = ", s_error , file = doc)
    print("S-pick standard deviation = ", s_error_std)
    
    ## Wait for thread 
    coord.join(threads) 
    sess.close()

def main(_):
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print('[INFO   ]\t Output directory does not exist, creating directory: ' + os.path.abspath(args.output_dir))
            os.makedirs(args.output_dir)
    test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Model testing')
    parser.add_argument('--tfrecords_dir', help ='Tfrecords directory',default = './')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'tfrecords')
    parser.add_argument('--checkpoint_path', help ='Path of checkpoint to restore', default = './model.ckpt')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 3)
    parser.add_argument('--waveform_length', help = 'Waveform length)', type = int, default = 6000)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 4)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)       
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files.', default = './output/')
    args, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run()
