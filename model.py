import tensorflow as tf
from tensorflow.keras import layers

layers = tf.keras.layers

#     # maxpool 대신 strided conv
#     # fully 안씀
#     # batch normalization

# 모든 가중치는 zero-centered Normal-distribution (with deviation 0.02)
zero_center_nomal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
zero_center_nomal = tf.keras.initializers.he_normal()

def make_generator_model():
    # generator
    # Z fullyconnect 백터인풋 
    # 아웃풋 레이어 BN 제거
    # 모두 relu 아웃풋 tanh
    model = tf.keras.Sequential(name = "generator")
    model.add(layers.Dense(4*4*1024, input_shape=(100,)))
    model.add(layers.Reshape((4, 4, 1024)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    # descriminator
    # 3차원 인풋 fully -> sigmoid 아웃풋
    # 인풋 레이어 BN 제거
    # 모두 leakyrelu
    model = tf.keras.Sequential(name = "discriminator")
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(layers.LeakyReLU(0.2)) # slope 0.2
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid')) 
    
    return model

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Create an instance of the model
    generator = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.imshow(tf.cast(generated_image[0] * 127.5 + 127.5, dtype=tf.uint8))
    # plt.imshow((generated_image[0]*255))
    plt.show()

    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)

    print(generator.summary())
    print(discriminator.summary())