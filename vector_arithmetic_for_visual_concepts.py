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
seed = np.load("./cherry_pick.npy")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/vector_arithmetic/')

for i in range(10):
    # seed1 = tf.random.normal([noise_dim])
    # seed2 = tf.random.normal([noise_dim])
    # seed3 = tf.random.normal([noise_dim])
    seed1 = seed[np.random.randint(len(seed))]
    seed2 = seed[np.random.randint(len(seed))]
    seed3 = seed[np.random.randint(len(seed))]

    calc = (seed1 - seed2 + seed3) 
    # calc = calc - np.mean(calc) 
    # calc = calc / (tf.math.reduce_std(calc) / 0.1)

    # predict
    seeds = tf.stack([seed1, seed2, seed3, calc])
    start = time.time()
    origin, minus, pluse, output = generator(seeds, training=False)
    print("generate time :",time.time()-start)

    origin = tf.cast([origin * 127.5 + 127.5], dtype=tf.uint8)
    minus = tf.cast([minus * 127.5 + 127.5], dtype=tf.uint8)
    pluse = tf.cast([pluse * 127.5 + 127.5], dtype=tf.uint8)
    output = tf.cast([output * 127.5 + 127.5], dtype=tf.uint8)

    with test_summary_writer.as_default():
        tf.summary.image(f"vactor_arithmetic", tf.concat([origin, minus, pluse, output], axis=2), step=i)

        