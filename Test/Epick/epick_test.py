"""
This code is based on https://github.com/mingzhaochina/unet_cea
"""

mport os, sys
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

def result(prediction, true_labels):
  TE = FE = 0     ## TE: true earthquake and FE: false earthquake
  TN = FN = 0     ## TN: true pure noise and FN: false noise
  Final_true = []
  Final_pre = []
  Final_p = [] # P index difference between true and predict
  Final_s = [] # S index difference between true and predict
  false_p = []
  false_s = []
  d_pp = []
  d_ss = []
  batch_size = prediction.shape[0]
  for i in range(batch_size):
      pred_array = prediction[i, :, :]
      for j in range(pred_array.shape[0]):
          if j!= np.argmax(pred_array[:, 1], axis = 0):
              pred_array[j][1] = 0
      for j in range(pred_array.shape[0]):
          if j!=  np.argmax(pred_array[:, 2], axis = 0):
              pred_array[j][2] = 0
      pred_array = np.argmax(pred_array, axis = 1)
      pre_labels = list(pred_array) 
      true_n = list(true_labels[i, :, 0])        
      true_p = list(true_labels[i, :, 1])       
      true_s = list(true_labels[i, :, 2])   
      if true_n == [1]*6000 and pre_labels ==  [0]*6000:
          TN += 1
      if true_n == [1]*6000 and pre_labels != [0]*6000:
          FE += 1
      if true_n != [1]*6000 and pre_labels == [0]*6000:
          FN += 1
      false_p.append(true_p.index(1))
      false_s.append(true_s.index(1))
      if true_n != [1]*6000 and pre_labels != [0]*6000:
          TE +=1
          tp1 = true_p.index(1)
          ts1 = true_s.index(1)
          f_true[tp1] = 1
          f_true[ts1] = 2
          Final_true = Final_true + f_true
          '''calculate the difference between true index and predicted index'''
          pp = []
          ss = []

          pp.append(pred_label.index(1))
          d_pp.append(pp[-1] - tp1)
          if abs(tp1 - pp[-1]) < 10 or abs(tp1-pp[-1]) == 10
              pre_labels[pp[-1]] = 0
              pre_labels[tp1] = 1

          ss.append(pred_labels.index(2))
          d_ss.append(ss[-1] - ts1)
          if abs(ts1 - ss[-1])< 20 or abs(ts1 - ss[-1]) == 20:
              pre_labels[ss[-1]] = 0
              pre_labels[ts1] = 2

    Final_pre = Final_pre + pre_labels
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
    logits = model(waveforms, args.num_classes, True) 
    
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

    ''' Metrics of earthquake detection '''
    E_accuracy = (total_TE + total_TN)/(step*args.batch_size)
    E_precision = total_TE/(total_TE + total_FE)
    E_recall =  total_TE/(total_TE + total_FN)
    E_f1 = 2 * E_precision * E_recall/(E_precision + E_recall)
    print("earthquake_detection_accuracy: ", E_accuracy)
    print("earthquake_detection_precision: ", E_precisionc)
    print("earthquake_detection_recall: ",  E_recall)
    print("earthquake_detection_f1: ", E_f1)

    cm2 = confusion_matrix(F_t, F_p) 
    Final_accuracy = (cm2[0][0]+cm2[1][1]+cm2[2][2])/(cm1[1][1]*6000)
    Final_precision = precision_score(F_t, F_p, average = 'weighted')
    Final_recall = recall_score(F_t, F_p, average = 'weighted')
    Final_f1 = f1_score(F_t, F_p, average = 'weighted')
    print("Final_detection_accuracy: ",Final_accuracy)
    print("Final_detection_precision: ", Final_precision)
    print("Final_detection_recall: ",  Final_recall)
    print("Final_detection_f1: ", Final_f1)
    
    print('classification report: ', classification_report(F_t, F_p, target_names=['Non-arrival', 'P phase', 'S phase'], digits=5))
   
    p_error = [int(Final_p_error[i])/100.0 for i in range(len(Final_p_error))] ## converting smaples to time (s)
    p_error_mean = sum(p_error)/len(p_error)
    p_error_std = statistics.stdev(p_error)
    
    s_error = [int(Final_s_error[i])/100.0 for i in range(len(Final_s_error))]
    s_error_mean = sum(s_error)/len(s_error)
    s_error_std = statistics.stdev(s_error)

    print("P pick mean = ", p_error, file=doc)
    print("P pick standard deviation = ", p_error_std)
    print("S pick mean = ", s_error , file = doc)
    print("S pick standard deviation = ", s_error_std)
    
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
