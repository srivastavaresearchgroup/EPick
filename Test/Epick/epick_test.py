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
import fnmatch,math
import itertools
from matplotlib.ticker import FuncFormatter
import math

# initial setting
FLAGS = None
def result_static_fun3(predict_images, t_labels):
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
  batch_size = predict_images.shape[0]
  for i in range(batch_size):
    image_array = predict_images[i, :, :]  ## [image_size, 3]
    p1 = np.argmax(image_array[:, 1], axis = 0)
    for j in range(image_array.shape[0]):
      if j!= p1:
        image_array[j][1] = 0
        s1 = np.argmax(image_array[:, 2], axis = 0)
    for j in range(image_array.shape[0]):
      if j!= s1:
        image_array[j][2] = 0
    image_array = np.argmax(image_array, axis = 1)
    pre_labels = list(image_array) ## n =0, P = 1, S= 
    true_n = list(t_labels[i, :, 0])        ## ture noise label
    true_p = list(t_labels[i, :, 1])        ## true p label
    true_s = list(t_labels[i, :, 2])        ## true s label
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
      for i in range(len(pre_labels)):
        if pre_labels[i] == 1:
          pp.append(i)
          if len(pp)>1:
            d_pp.append(pp[-1] - tp1)
            if abs(tp1 - pp[-1]) < 10 or abs(tp1-pp[-1]) == 10
              pre_labels[pp[-1]] = 0
              pre_labels[tp1] = 1
            elif len(pp) == 1:
              d_pp.append(pp[0] - tp1)
              if abs(tp1 - pp[0]) < 10 or abs(tp1-pp[0]) == 10
                pre_labels[pp[0]] = 0
                pre_labels[tp1] = 1
      for j in range(len(pre_labels)):
        if pre_labels[j] == 2:
          ss.append(j)
          if len(ss)>1:
            d_ss.append(ss[-1] - ts1)
            if abs(ts1 - ss[-1])< 20 or abs(ts1 - ss[-1]) == 20:
              pre_labels[ss[-1]] = 0
              pre_labels[ts1] = 2
          elif len(ss) == 1:
            d_ss.append(ss[0] - ts1)
            if abs(ts1 - ss[0])< 20 or abs(ts1-ss[0])==20:
              pre_labels[ss[0]] = 0
              pre_labels[ts1] = 2
  Final_pre = Final_pre + pre_labels
  Final_p += d_pp
  Final_s += d_ss       
  return TE, FE, TN, FN, Final_pre, Final_true, Final_p, Final_s, false_p, false_s
    
def test():
    tf.compat.v1.set_random_seed(24)
    cfg = config.Config()
    cfg.batch_size = FLAGS.batch_size
    cfg.add = 1
    cfg.n_clusters = FLAGS.num_classes
    cfg.n_clusters += 1
    cfg.n_epochs = 1

    pos_pipeline = dp.DataPipeline(FLAGS.tfrecords_dir, cfg, False)
    images = pos_pipeline.samples 
    labels = pos_pipeline.labels   
    logits = model(images, FLAGS.num_classes, True) 
    
    accuracy = accuracy(logits, labels)
    loss = loss(logits, labels, FLAGS.weight_decay_rate)
    predicted_images = predict2(logits, FLAGS.batch_size, FLAGS.image_size) 

    ## session initialization
    init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), 
                                 tf.compat.v1.local_variables_initializer())           
    sess = tf.compat.v1.Session()
    sess.run(init_op)
    saver = tf.compat.v1.train.Saver() 
    
    ## restore modeltf.io.gfile
    if not tf.io.gfile.exists(FLAGS.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO   ]\t Found checkpoint file, restoring model.') 
        saver.restore(sess, FLAGS.checkpoint_path)  # reload the trained model

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
    
            TE, FE, TN, FN, Final_pre, Final_true, Final_p, Final_s, false_p, false_s = result_static_fun3(predicted_images_value, ss_labels)
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
        print('[INFO    ]\t Done evaluating in %d steps.' % step)
    finally:
        """When done, ask the threads to stop."""
        coord.request_stop()

    doc = open('./test.txt', 'w')

    ''' metric of earthquake '''
    E_accuracy = (total_TE + total_TN)/(step* FLAGS.batch_size)
    E_precision = total_TE/(total_TE + total_FE)
    E_recall =  total_TE/(total_TE + total_FN)
    E_f1 = 2 * E_precision * E_recall/(E_precision + E_recall)
    print("earthquake_detection_accuracy: ", E_accuracy, file = doc)
    print("earthquake_detection_precision: ", E_precision, file = doc)
    print("earthquake_detection_recall: ",  E_recall, file = doc)
    print("earthquake_detection_f1: ", E_f1, file = doc)


    ''' plot confusion matrix of earthquake and non-earthqule '''
    cm1= confusion_matrix([0, 0, 0, 0], [1, 1, 1, 1])  ## initialize
    cm1[0][0] = total_TN
    cm1[0][1] = total_FE
    cm1[1][0] = total_FN
    cm1[1][1] = total_TE
    multi_label1 = [ 'Noise', 'Earthquake']
    tick_marks = np.arange(len(multi_label1))
    plt.xticks(tick_marks, multi_label1, rotation=45)
    plt.yticks(tick_marks, multi_label1)
    thresh = cm1.max() / 2
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, "{:,}".format(cm1[i, j]),horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black",
                 fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('./earthquake_detection_confusion_matrix.jpg', dpi =600)
    # plt.show()
    plt.close()

    cm2 = confusion_matrix(F_t, F_p) 
    multi_label2 = ['Non-arrival', 'P-arrival', 'S-arrival' ]
    tick_marks = np.arange(len(multi_label2))
    plt.xticks(tick_marks, multi_label2, rotation=45)
    plt.yticks(tick_marks, multi_label2)
    thresh = cm2.max() / 2
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(j, i, "{:,}".format(cm2[i, j]),horizontalalignment="center",
                 color="white" if cm2[i, j] > thresh else "black",
                 fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig('./phase_detection_confusion_matrix.jpg', dpi =350)
    plt.close()

    Final_accuracy = (cm2[0][0]+cm2[1][1]+cm2[2][2])/(cm1[1][1]*6000)
    Final_precision = precision_score(F_t, F_p, average = 'weighted')
    Final_recall = recall_score(F_t, F_p, average = 'weighted')
    Final_f1 = f1_score(F_t, F_p, average = 'weighted')
    print("Final_detection_accuracy: ",Final_accuracy, file = doc)
    print("Final_detection_precision: ", Final_precision, file = doc)
    print("Final_detection_recall: ",  Final_recall, file = doc)
    print("Final_detection_f1: ", Final_f1, file = doc)
    
    
    t = classification_report(F_t, F_p, target_names=['Neither', 'P phase', 'S phase'], digits=5)
    print(t, file = doc)
    
    ''''statistic'''
    p_error = [int(Final_p_error[i])/100.0 for i in range(len(Final_p_error))] ## converting smaples to time (s)
    p_error_mean = sum(p_error)/len(p_error)
    p_error_std = statistics.stdev(p_error)
    
    s_error = [int(Final_s_error[i])/100.0 for i in range(len(Final_s_error))]
    s_error_mean = sum(s_error)/len(s_error)
    s_error_std = statistics.stdev(s_error)

    print("P pick mean = ", p_error, file=doc)
    print("P pick standard deviation = ", p_error_std, file = doc)
    print("S pick mean = ", s_error , file = doc)
    print("S pick standard deviation = ", s_error_std ,file = doc)

    ## error saving
    df1 = pd.DataFrame(Final_p_error)
    df1.to_csv('./P_pick.csv', index=False)
    df2 = pd.DataFrame(Final_s_error)
    df2.to_csv('./S_pick.csv', index=False)


    ''' plot P-arrival time error histogram '''
    plt.figure()
    plt.hist(p_error, bins= 20, facecolor = 'blue', edgecolor = 'black', linewidth=0.2, log = True)
    plt.xlabel("Time residuals (s)",fontsize = 12)
    plt.ylabel("Number of P picks", fontsize = 12)
    plt.title("Distribution of time residuals of P picks", fontsize = 12)
    plt.tight_layout()
    plt.savefig('./P_histogram.jpg', dpi =600)
    plt.close()
    
    ''' plot S-arrival time error histogram '''
    plt.figure()
    plt.hist(s_error, bins = 20, facecolor = 'blue', edgecolor = 'black', linewidth=0.2, log = True)
    plt.xlabel("Time residuals (s)",fontsize = 12)
    plt.ylabel("Numeber of S picks", fontsize = 12)
    plt.title("Distribution of time residual of S picks", fontsize = 12)
    plt.tight_layout()
    plt.savefig('./S_histogram.jpg', dpi =600)
    plt.close()

    ## Wait for thread 
    coord.join(threads) 
    sess.close()

def main(_):
    if FLAGS.output_dir is not None:
        if not os.path.exists(FLAGS.output_dir):
            print('[INFO   ]\t Output directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            os.makedirs(FLAGS.output_dir)
    test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Test Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir',    help ='Tfrecords directory',default = './stead_test_data/')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'tfrecords')
    parser.add_argument('--checkpoint_path',  help ='Path of checkpoint to restore',
                       default = './model.ckpt')
    parser.add_argument('--num_classes',      help = 'Number of segmentation labels', type = int, default = 3)
    parser.add_argument('--image_size',       help = 'Target image size (resize)', type = int, default = 6000)
    parser.add_argument('--batch_size',       help = 'Batch size', type = int, default = 4)
    parser.add_argument('--output_dir',
                        help = 'Output directory for the prediction files. If this is not set then predictions will not be saved',
                        default = './output/')
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)                    
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run()
