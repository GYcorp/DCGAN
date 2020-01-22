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
# seed1 = tf.random.normal([num_examples_to_generate, noise_dim])
# seed2 = tf.random.normal([num_examples_to_generate, noise_dim])
seed1 = seed[0:num_examples_to_generate]
seed2 = seed[num_examples_to_generate:num_examples_to_generate*2]

increase = (seed2 - seed1)/10

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
test_summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/latent_space/')


# predict
for i in range(10):
    seed = seed1 + increase*i
    start = time.time()
    predictions = generator(seed, training=False)
    
    with test_summary_writer.as_default():
        tf.summary.image("latent_space_walking", tf.cast(predictions * 127.5 + 127.5, dtype=tf.uint8), step=i)

    print("generate time :",time.time()-start)
    # fig = plt.figure(figsize=(4,4))

    # for i in range(predictions.shape[0]):
    #     plt.subplot(4, 4, i+1)
    #     plt.imshow(tf.cast(predictions[i] * 127.5 + 127.5, dtype=tf.uint8))
    #     plt.axis('off')

    plt.show()

        