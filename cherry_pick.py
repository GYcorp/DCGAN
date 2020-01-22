import tensorflow as tf
import os
import time
import numpy as np

from define import *
import model

# build model
generator = model.make_generator_model()
discriminator = model.make_discriminator_model()

# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

seeds = []
confs = []

# predict
for epoch in range(100):

    # make seed
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    start = time.time()
    predictions = generator(seed, training=False)
    print("generate time :",time.time()-start)
    confidence = discriminator(predictions, training=False)

    seeds.append(seed)
    confs.append(confidence)

seeds = tf.concat(seeds, axis=0).numpy()
confs = tf.concat(confs, axis=0)#.numpy().tolist()

confs = tf.argsort(confs, axis=0, direction='DESCENDING')
rank = tf.reshape(confs, [-1]).numpy()

best = generator(tf.reshape(seeds[rank[0]], [1,-1]), training=False)
worst = generator(tf.reshape(seeds[rank[-1]], [1,-1]), training=False)

best = tf.cast(best[0] * 127.5 + 127.5, dtype=tf.uint8)
worst = tf.cast(worst[0] * 127.5 + 127.5, dtype=tf.uint8)

# import cv2
# cv2.imshow("best", best.numpy())
# cv2.imshow("worst", worst.numpy())
# cv2.waitKey(0)

def get_cherry_pick(topK=10):
    return seeds[rank[0:topK]]

top_N = get_cherry_pick(14*14)

np.save("./cherry_pick", top_N)
print("cherrypick saved")

