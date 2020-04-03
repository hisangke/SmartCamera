#coding=utf-8
from __future__ import division
import os
import tensorflow as tf
import numpy as np 
import glob
import sys
import h5py 
import time
import random
from tqdm import tqdm
from conv_cell import ConvLSTMCell
from fc_attention import fc_attention_sum,fc_attention
from conv_attention import conv_attention_sum,conv_attention

########################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
conv_img_features = "/data3/hisangke2/DTA-master/dataset/Video/GettingStarted/Python/dataset/"
#fc_img_features = "/your_dir/ucf11_features/image_fc6/"

train_list = "/data3/hisangke2/DTA-master/ucf11TrainTestlist/ucf11_train.txt"
test_list = "/data3/hisangke2/DTA-master/ucf11TrainTestlist/ucf11_test.txt"
classInd = "/data3/hisangke2/DTA-master/ucf11TrainTestlist/classInd.txt"

n_inputs = 4096   
n_steps = 40    
n_hidden_units = filters = 1024
fc_attention_size = 50
n_classes = 11
n_layers = 2
scale = 1
n_num =10

timesteps = num_frames = 40
shape_1 = [7, 7]
shape_2 = [5, 10]
kernel = [3, 3]
channels = 512
attention_kernel = [1,1]
basic_lr = 0.0001

g = open(classInd,"r")
labels = sorted(g.readlines())
nums = []
names = []
for label in labels:
  a = label.split(" ")
  nums.append(int(a[0])-1)
  names.append(a[1][:-1])
label_dict = dict(zip(names,nums))
print(label_dict)

train_lines = []
f = open(train_list,"r")  
lines_ = f.readlines()
for i in range(len(lines_)//scale):
  train_lines.append(lines_[i*scale])
len_train = len(train_lines)
print("successfully,len(train_list)",len_train)
random.shuffle(train_lines)

h = open(test_list,"r")  
test_lines_ = sorted(h.readlines())
test_lines = [] 
for i in range(len(test_lines_)//scale):
  test_lines.append(test_lines_[i*scale])
len_test = len(test_lines)
print("successfully,len(test_list)",len_test)

test_labels = []
ground_labels = []
for test_video in test_lines:
    video_class = str(test_video.split(" ")).split("_")[1]
    ground_label = label_dict[video_class]
    test_labels.append(ground_label)
#print(test_labels)
#########################################################################
def conv_feature(video_name):
    #print(video_name)
    g = h5py.File(conv_img_features+video_name)
    #print(conv_img_features,video_name)
    img_features = g['video_feature']
    img_features = img_features[:]
    g.close()
    return img_features

def accuracy(a,b):
    c = np.equal(a,b).astype(float)
    acc = sum(c)/len(c)
    return acc
def compute_score_loss(batch_conv_img,batch_labels):
    global conv_pred
    global loss
    conv_score,video_loss =  sess.run([conv_pred,loss], feed_dict={
            conv_img: batch_conv_img,
            ys : batch_labels
            })
    return conv_score,video_loss
#################################################################################
conv_img = tf.placeholder(tf.float32, [None, timesteps] + shape_1 + [channels]) 

ys = tf.placeholder(tf.float32, [None, n_classes])
Lr = tf.placeholder(tf.float32) 

def CONV_LSTM(conv_img,shape,attention):
    img_cell = ConvLSTMCell(shape, filters, kernel)
    img_outputs, img_state = tf.nn.dynamic_rnn(img_cell, conv_img, dtype=conv_img.dtype, time_major=True)
    if attention == True: 
        img_attention_output = conv_attention_sum(img_outputs, attention_kernel)
        img_attention_output = tf.layers.batch_normalization(img_attention_output)
        return img_attention_output
    else:
        img_outputs = tf.layers.batch_normalization(img_outputs)
        img_outputs = tf.reduce_sum(img_outputs,axis = 1)
        return img_outputs  

def FC_layer(inputs):
    weights2 =  tf.get_variable("weights2", [n_hidden_units, n_classes],
        initializer=tf.truncated_normal_initializer())
    biases2 = tf.get_variable("biases2", [n_classes],
        initializer=tf.truncated_normal_initializer())
    result = tf.nn.dropout((tf.matmul(inputs, weights2) + biases2), 0.5)
    return result

conv_img_out = CONV_LSTM(conv_img, shape_1,True)
conv_img_drop = tf.nn.dropout(conv_img_out, 0.5)
conv_img_drop = tf.nn.max_pool(conv_img_drop,[1,7,7,1],[1,1,1,1],padding='VALID')
conv_img_drop = tf.reshape(conv_img_drop,[-1,filters])

with tf.variable_scope("FC"):
    conv_result = FC_layer(conv_img_drop)

conv_pred = tf.nn.softmax(conv_result)
conv_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=conv_result, labels=ys))
loss =conv_loss
print(loss)
train_op = tf.train.AdamOptimizer(Lr).minimize(loss)
config = tf.ConfigProto() 
saver = tf.train.Saver()
if not os.path.exists('fc+conv_lstm_tmp_attention'):
    os.mkdir('fc+conv_lstm_tmp_attention/')
sess = tf.Session(config=config)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
init_op = tf.global_variables_initializer() 
sess.run(init_op)
############################################start_training###########################################
batch_size = 12
k = 0
u = 0

full_loss = []
epochs=1000
all_train_loss=[]
all_test_loss=[]
for epoch in range(epochs):
    train_loss=[]
    test_loss=[]
    lr = basic_lr
    pbar = tqdm(total=((len_train//batch_size)+1))
    for i in range((len_train//batch_size)+1):
        batch_conv_img = np.zeros((10*batch_size,40,7,7,512),dtype = np.float32)
        batch_labels = np.zeros((10*batch_size,n_classes),dtype = np.int8)
        time1 = time.time()
        k= 0
        for j in range(i*batch_size,(i+1)*batch_size):
            if j>=len_train:
                j = random.randint(0,len_train-1)
            one_hot=np.zeros((n_classes),dtype = np.int8)
            line = train_lines[j]
            video_name = line.split(" ")[0]
            video_class = str(line.split(" ")[0]).split("_")[1]
            feature_name = video_name+"_.h5"
            one_hot[label_dict[video_class]] = 1
            label = np.expand_dims(one_hot,axis = 0)
            label = label.repeat(10,axis=0)
            conv_img_fea = conv_feature(feature_name)
            #print(conv_img_fea.shape)#(40, 512, 7, 7)
            conv_img_fea = np.transpose(conv_img_fea[np.newaxis,:],(0,1,3,4,2))
            batch_conv_img[n_num*k:n_num*k+n_num] = conv_img_fea
            batch_labels[n_num*k:n_num*k+n_num] = label
            k = k+1
        time2 = time.time()
        print("batch_read_time:", '{0:.2f}'.format(time2-time1),"s")
        k = k+1
        loss_,_ = sess.run([loss,train_op], feed_dict={
            conv_img : batch_conv_img,
            ys : batch_labels,
            Lr : lr
        })
        print("batch_loss:",loss_)
        f=open("logs.txt","a+")
        logs="train_loss:"+str(loss_)
        print(logs)
        f.write(logs)
        f.write("\n")
        f.close()
        train_loss.append(loss_)
        pbar.update(1)
    all_train_loss.append(train_loss)
    pbar.close()
    model_name = "score_attention/"+str(epoch)+"epoch_model.ckpt"
    saver.save(sess, model_name)
    print("successfully,start_test!")
############################################################################################
    pbar = tqdm(total=((len_test//batch_size)+1))
    conv_scores = []
    for i in range((len_test//batch_size)+1):
        batch_conv_img = np.zeros((n_num*batch_size,40,7,7,512),dtype = np.float32)
        batch_labels = np.zeros((n_num*batch_size,n_classes),dtype = np.int8)
        time1 = time.time()
        k= 0
        for j in range(i*batch_size,(i+1)*batch_size):
            if j>=len_test:
                j = random.randint(0,len_test-1)
            one_hot=np.zeros((n_classes),dtype = np.int8)
            line = test_lines[j]
            video_class = str(line.split(" ")[0]).split("_")[1]
            feature_name = line.split(" ")[0]+"_.h5"

            one_hot[label_dict[video_class]] = 1
            label = np.expand_dims(one_hot,axis = 0)
            label = label.repeat(10,axis=0)

            conv_img_fea = conv_feature(feature_name)
            conv_img_fea = np.transpose(conv_img_fea[np.newaxis,:],(0,1,3,4,2))

            batch_conv_img[n_num*k:n_num*k+n_num] = conv_img_fea
            batch_labels[n_num*k:n_num*k+n_num] = label
            k = k+1
        test_conv_score, test_loss_= compute_score_loss(batch_conv_img,batch_labels)
        #print("test_loss:",test_loss_)
        f=open("logs.txt","a+")
        logs="test_loss:"+str(test_loss_)
        print(logs)
        f.write(logs)
        f.write("\n")
        f.close()
        test_loss.append(test_loss_)
        test_conv_score = np.sum(np.reshape(np.array(test_conv_score),(batch_size,n_num,n_classes)),axis=1) 
        conv_scores.append(test_conv_score)
        pbar.update(1)
    all_test_loss.append(test_loss)
    conv_scores = np.reshape(np.array(conv_scores),(-1,n_classes))[:len_test]

    pbar.close()
    num_test_score = conv_scores
    test_label_pred = np.argmax(num_test_score,axis = 1)
    #print(test_label_pred.shape)
    #print(epoch,"epoch:, attention，Conv_lstm_ACC:",accuracy(test_label_pred,test_labels))
    f=open("logs.txt","a+")
    logs=str(epoch)+"epoch:, attention，Conv_lstm_ACC: "+str(accuracy(test_label_pred,test_labels))
    print(logs)
    f.write(logs)
    f.write("\n")
    f.close()
    test_info = []
    for i in range(len_test):
        video_info = []
        video_info.append(num_test_score[i])
        video_info.append(test_labels[i])
        test_info.append(video_info)
    npz_name = "score_attention/"+str(epoch)+"_epoch_conv_score.npz"
    np.savez(npz_name, test_info=test_info)
    import h5py
    trainloss= h5py.File('./score_attention/trainloss.h5', 'w')
    testloss=h5py.File('./score_attention/testloss.h5', 'w')
    trainloss.create_dataset(str(epoch), data=np.array(train_loss))
    testloss.create_dataset(str(epoch), data=np.array(test_loss))
    trainloss.close()
    testloss.close()
