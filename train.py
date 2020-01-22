import tensorflow as tf
import model
import matplotlib.pyplot as plt
import datetime
import os
import time

from define import *
import data

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])

# ready data
# train_dataset = data.CelebA_Data()
# train_dataset = data.Flickr_Faces_HQ_Data()
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

# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



seed = tf.random.normal([num_examples_to_generate, noise_dim])

# tf log
generator_loss_metrics = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_loss_metrics = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(f'logs/{current_time}/train_loss/')

# train
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # generator_loss_metrics(gen_loss)
        # discriminator_loss_metrics(disc_loss)
        # tf.print("gen_loss: ", gen_loss, "disc_loss: ", disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs):
    index = 0
    for epoch in range(epochs):
        start = time.time()

        for step, (name_batch, image_batch) in enumerate(data.law()):
            gen_loss, disc_loss = train_step(image_batch)
            print(f"{step*BATCH_SIZE}/{BUFFER_SIZE} gen_loss: {gen_loss.numpy()} disc_loss: {disc_loss.numpy()}")
            with summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=index)
                tf.summary.scalar('disc_loss', disc_loss, step=index)
                index+=1
        # with summary_writer.as_default():
        #     tf.summary.scalar('gen_loss', generator_loss_metrics.result(), step=epoch)
        #     tf.summary.scalar('disc_loss', discriminator_loss_metrics.result(), step=epoch)
        generate_and_save_images(generator, epoch, seed)

        if (epoch) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for each {} is {} sec'.format(epoch + 1, time.time()-start))

    generate_and_save_images(generator, epoch, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    
    with summary_writer.as_default():
        tf.summary.image("test_image", tf.cast(predictions * 127.5 + 127.5, dtype=tf.uint8), step=epoch)


train(train_dataset, EPOCHS)

        