import tensorflow as tf
from tensorflow.keras import layers, Model
import model
import matplotlib.pyplot as plt
import datetime
import os
import time

from define import *

import cv2

# build model
discriminator = model.make_discriminator_model()
conv2d_1 = Model(inputs = discriminator.input, outputs = discriminator.get_layer('conv2d_1').output)
conv2d_2 = Model(inputs = discriminator.input, outputs = discriminator.get_layer('conv2d_2').output)
conv2d_3 = Model(inputs = discriminator.input, outputs = discriminator.get_layer('conv2d_3').output)
conv2d_4 = Model(inputs = discriminator.input, outputs = discriminator.get_layer('conv2d_4').output)

# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# make image
BATCH_SIZE = 2
image_path = "C:\\Users\\UfXpri\\Desktop\\DCGAN_Model_face_try3\\CelebA\\"
image_path_list = os.listdir(image_path)
image_path_list = [ image_path+path for path in image_path_list]

batch_iamge_list=[]
for i in range(0, len(image_path_list), BATCH_SIZE):
    batch_iamge_list.append(image_path_list[i:i+BATCH_SIZE])

def get_batch(index):
    paths = batch_iamge_list[index]
    buffer = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - 127.5) / 127.5
        buffer.append(image)
    images = tf.convert_to_tensor(buffer, dtype=tf.float32)
    return images


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/visualizing-discriminator/')


def make_metrix(tensor):
    tensor = tf.transpose(tensor, [0,3,1,2])
    tensor = tf.expand_dims(tensor, 4) # (None, C, W, H, 1)
    batches = []

    for batch_index in range(len(tensor)):
        metrix = []
        for x_index in range(10):
            line = []
            for y_index in range(10):
                index = x_index*10+y_index
                line.append(tensor[batch_index][index])
            line = tf.concat(line, 1)
            metrix.append(line)
        metrix = tf.concat(metrix, 0)
        batches.append(metrix)
    batches = tf.stack(batches, 0)

    return batches

# predict
for i in range(10):
    images = get_batch(i)
    start = time.time()

    conv2d_1_result = conv2d_1(images, training=False)
    conv2d_2_result = conv2d_2(images, training=False)
    conv2d_3_result = conv2d_3(images, training=False)
    conv2d_4_result = conv2d_4(images, training=False)

    slice_1 = make_metrix(conv2d_1_result)
    slice_2 = make_metrix(conv2d_2_result)
    slice_3 = make_metrix(conv2d_3_result)
    slice_4 = make_metrix(conv2d_4_result)

    with test_summary_writer.as_default():
        tf.summary.image("discriminator layers/conv2d_1", slice_1, step=i)
        tf.summary.image("discriminator layers/conv2d_2", slice_2, step=i)
        tf.summary.image("discriminator layers/conv2d_3", slice_3, step=i)
        tf.summary.image("discriminator layers/conv2d_4", slice_4, step=i)

    print("generate time :",time.time()-start)

        