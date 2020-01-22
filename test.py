import tensorflow as tf
import model
import matplotlib.pyplot as plt
import datetime
import os
import time
import numpy as np

from define import *

# build model
generator = model.make_generator_model()

# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# make seed
N = 10
num_examples_to_generate = N*N
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed = np.load("./cherry_pick.npy")[:num_examples_to_generate, :]


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/random_test/')

# predict
start = time.time()
predictions = generator(seed, training=False)
print("generate time :",time.time()-start)

def make_metrix(tensor):

    metrix = []
    for x_index in range(10):
        line = []
        for y_index in range(10):
            index = x_index*10+y_index
            line.append(tensor[index])
        line = tf.concat(line, 1)
        metrix.append(line)
    metrix = tf.concat(metrix, 0)

    return metrix

met = tf.cast([make_metrix(predictions) * 127.5 + 127.5], dtype=tf.uint8).numpy()

with test_summary_writer.as_default():
    tf.summary.image("test", met, step=0)