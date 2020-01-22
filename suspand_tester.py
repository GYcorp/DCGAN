import tensorflow as tf
import model
import matplotlib.pyplot as plt
import datetime
import os
import time

from define import *
import data

import cv2

# ready data
train_dataset = data.law()

# build model
generator = model.make_generator_model()
discriminator = model.make_discriminator_model()

# loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
    
generator_optimizer = tf.keras.optimizers.Adam(learning_rate*2, beta_1 = momentum)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = momentum)

# train
# @tf.function
def train_step(images, step):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, real_output)

        tf.print(step*BATCH_SIZE, "gen_loss: ", gen_loss, "disc_loss: ", disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for step, (name_batch, image_batch) in enumerate(dataset):
            print(name_batch)
            train_step(image_batch, step)

        print('Time for each {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_dataset, EPOCHS)

        