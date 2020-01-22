# https://github.com/fyu/lsun
import os
import tensorflow as tf
import cv2
from define import *
# AUTOTUNE = tf.data.experimental.AUTOTUNE

def Flickr_Faces_HQ_Data():
    with open("./Flickr-Faces-HQ/index.txt") as f:
        paths = f.readlines()
        paths = ['./Flickr-Faces-HQ/'+path[:-1] for path in paths]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
        
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (image - 0.5) * 2
        image = tf.image.resize(image, [64, 64])
        return image

    path_ds = path_ds.shuffle(BUFFER_SIZE)
    images_ds = path_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    images_ds = images_ds.batch(BATCH_SIZE)
    images_ds = images_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return images_ds

def CelebA_Data():
    raw_dataset = tf.data.TFRecordDataset("C:\\Users\\ming\\Downloads\\CelebA\\Img\\CelebA.tfrecord")

    def parse_image(example_proto):
        # tf.print("parse_in")
        parsed_data = tf.io.parse_single_example(example_proto, {
            'image': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string)
        })
        name = parsed_data['name']
        # tf.print(name)
        image = tf.image.decode_image(parsed_data['image'])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (image - 0.5) * 2
        # tf.print("parse_out")
        return name, image

    images_ds = raw_dataset.map(parse_image)
    # images_ds = raw_dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    images_ds = images_ds.batch(BATCH_SIZE)
    # images_ds = images_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return images_ds

    
def law():
    image_path = "C:\\Users\\ming\\Downloads\\CelebA\\Img\\img_align_celeba\\"
    image_path_list = os.listdir(image_path)
    image_path_list = [ image_path+path for path in image_path_list]
    
    batch_iamge_list=[]
    for i in range(0, len(image_path_list), BATCH_SIZE):
        batch_iamge_list.append(image_path_list[i:i+BATCH_SIZE])

    for paths in batch_iamge_list:
        buffer = []
        names = []
        for path in paths:
            image = cv2.imread(path)
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image - 127.5) / 127.5
            buffer.append(image)
            names.append(path)
        names = tf.convert_to_tensor(names, dtype=tf.string)
        buffer = tf.convert_to_tensor(buffer, dtype=tf.float32)
        yield names, buffer


if __name__ == "__main__":

    import cv2

    # ds = CelebA_Data()
    ds = law()

    for names, images in ds:
        for name, image in zip(names, images):
            image = tf.cast(image * 127.5 + 127.5, dtype=tf.uint8)
            cv2.imshow("image", image.numpy())
            print(name)
            if cv2.waitKey(1) == ord("q"):
                break