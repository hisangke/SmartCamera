from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint
from torchvision import transforms
from PIL import Image
import imageio
import tensorflow as tf

def select_features(video_file):
    def read_frame(data):
        width=112
        height= 112
        if (width >= 171) or (height >= 128):
            raise ValueError("Target width need to be less than 171 and target height need to be less than 128.")
        image = Image.fromarray(data)
        image.thumbnail((171, 128), Image.ANTIALIAS)
        center_w = image.size[0] / 2
        center_h = image.size[1] / 2
        image = image.crop((center_w - width/ 2,center_h - height / 2,center_w + width/ 2,center_h + height/ 2))
        norm_image = np.array(image, dtype=np.float32)/255
        return np.ascontiguousarray(np.transpose(norm_image,(2, 0, 1)))
    width=112
    height= 112
    sequence_length = 40
    is_training=True
    video_reader = imageio.get_reader(video_file, 'ffmpeg')
    num_frames   = 0 
    for num,im in enumerate(video_reader):
        num_frames=+num
    num_frames+=1
    if sequence_length > num_frames:
        raise ValueError('Sequence length {} is larger then the total number of frames {} in {}.'.format(sequence_length, num_frames, video_file))
    step = 1
    expanded_sequence = sequence_length
    if num_frames > 2*sequence_length:
        step = 2
        expanded_sequence = 2*sequence_length
    seq_start = int(num_frames/2)-int(expanded_sequence/2)
    if is_training:
        seq_start = randint(0, num_frames - expanded_sequence)
    frame_range = [seq_start + step*i for i in range(sequence_length)]            
    video_frames = []
    for frame_index in frame_range:
        video_frames.append(read_frame(video_reader.get_data(frame_index)))
    return np.stack(video_frames, axis=1)
def read_frame(data):
    width=112
    height= 112
    if (width >= 171) or (height >= 128):
        raise ValueError("Target width need to be less than 171 and target height need to be less than 128.")
    image = Image.fromarray(data)
    image.thumbnail((171, 128), Image.ANTIALIAS)
    center_w = image.size[0] / 2
    center_h = image.size[1] / 2
    image = image.crop((center_w - width  / 2,center_h - height / 2,center_w + width  / 2,center_h + height / 2))
    norm_image = np.array(image, dtype=np.float32)/255
    return np.ascontiguousarray(np.transpose(norm_image, (2, 0, 1)))


import h5py
import numpy as np
import time
from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint
from torchvision import transforms
from PIL import Image
import imageio
import tensorflow as tf

ucf11category=["shooting","biking","diving","golf","riding","juggle","swing","tennis","jumping","spiking","walk"]
start=time.time()
def conv_feature(video_name="v_biking_02_03_"):
    video_name+=".h5"
    conv_img_features="F:\\data3\\hisangke2\\MyDataset\\UCF11\\GettingStarted\\Python\\dataset_vgg19\\"
    g = h5py.File(conv_img_features+video_name)
    #print(conv_img_features,video_name)
    img_features = g['video_feature']
    img_features = img_features[:]
    g.close()
    return img_features
feature_name="v_walk_dog_05_05_"
conv_img_fea = conv_feature(feature_name)
batch_xs = np.transpose(conv_img_fea[np.newaxis,:],(0,1,3,4,2))
#print(conv_img_fea)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "C:\\Users\\user\\Desktop\\Contest\\model\\20190807\\testvgg19_0730\\49")
    graph = tf.get_default_graph()
    input_x = sess.graph.get_tensor_by_name("conv_img:0")
    print(input_x)
    output = sess.graph.get_tensor_by_name("conv_pred:0")
    scores=sess.run(output,feed_dict={input_x: batch_xs})
    re="%d"%(np.argmax(scores, 1))
    print("predict: %d" % (np.argmax(scores, 1)),ucf11category[int(re)])
print("Time: ",time.time()-start)


import h5py
import numpy as np
import time
from __future__ import print_function
import sys
import os
import csv
import numpy as np
from random import randint
from torchvision import transforms
from PIL import Image
import imageio
import tensorflow as tf

ucf11category=["shooting","biking","diving","golf","riding","juggle","swing","tennis","jumping","spiking","walk"]
start=time.time()
def conv_feature(video_name="v_biking_02_03_"):
    video_name+=".h5"
    conv_img_features="F:\\data3\\hisangke2\\MyDataset\\UCF11\\GettingStarted\\Python\\dataset_vgg19\\"
    g = h5py.File(conv_img_features+video_name)
    img_features = g['video_feature']
    img_features = img_features[:]
    g.close()
    return img_features
feature_name="v_walk_dog_05_05_"
conv_img_fea = conv_feature(feature_name)
batch_xs = np.transpose(conv_img_fea[np.newaxis,:],(0,1,3,4,2))
#print(conv_img_fea)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "C:\\Users\\user\\Desktop\\Contest\\model\\20190807\\testvgg19_0730\\49")
    graph = tf.get_default_graph()
    input_x = sess.graph.get_tensor_by_name("conv_img:0")
    print(input_x)
    output = sess.graph.get_tensor_by_name("conv_pred:0")
    scores=sess.run(output,feed_dict={input_x: batch_xs})
    re="%d"%(np.argmax(scores, 1))
    print("predict: %d" % (np.argmax(scores, 1)),ucf11category[int(re)])
print("Time: ",time.time()-start)
