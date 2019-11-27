import glob
import os
import tensorflow as tf
from PIL import Image

import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 2.0 times the original size.
    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random.uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[0], tf.float32), scale), tf.int32)
    w_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[1], tf.float32), scale), tf.int32)
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=[1])
    img = tf.image.resize(img, new_shape)
    label = tf.image.resize(tf.expand_dims(label, 0), new_shape, method='nearest')
    label = tf.squeeze(label, axis=[0])

    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.
    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.image.random_crop(combined_pad, [crop_h, crop_w, 4])

    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop


class CityscapesDataset:

    def __init__(self, dataset_dir, load_option, img_size, batch_size, use_extra, ignore=19):
        self.h, self.w = img_size
        self.batch_size = batch_size

        self.use_extra = use_extra

        self.ignore_label = ignore
        # Check that the folder exists
        assert os.path.exists(dataset_dir) and os.path.isdir(dataset_dir), f"Dataset {dataset_dir} does not exist !"

        self.load_option = load_option

        # File containing the list of pictures and correspnding GT in leftImg8bit_trainvaltest
        gt_dir = os.path.join(dataset_dir, 'gtFine_trainvaltest')
        image_dir = os.path.join(dataset_dir, 'leftImg8bit_trainvaltest')

        gt_coarse_dir = os.path.join(dataset_dir, 'gtCoarse')
        image_coarse_dir = os.path.join(dataset_dir, 'leftImg8bit_trainextra')

        self.img_paths = sorted(glob.glob(os.path.join(image_dir, f'leftImg8bit/{load_option}/*/*leftImg8bit.png')))
        extra_img_paths = sorted(glob.glob(os.path.join(image_coarse_dir, f'leftImg8bit/*/*/*leftImg8bit.png')))

        # Load selected labels
        self.lbl_paths = sorted(
            glob.glob(os.path.join(gt_dir, f'gtFine/{load_option}/*/*gtFine_labelTrainIds.png')))
        coarse_lbl_paths = sorted(
            glob.glob(os.path.join(gt_coarse_dir, f'gtCoarse/train_extra/*/*gtCoarse_labelTrainIds.png')))

        if use_extra and load_option == 'train':
            assert os.path.isdir(image_coarse_dir), f'{image_coarse_dir} does not exist'
            self.lbl_paths += coarse_lbl_paths
            self.img_paths += extra_img_paths

        assert len(self.lbl_paths) != 0, f'Check data folder {image_dir}'
        assert len(self.img_paths) == len(self.lbl_paths), f'{len(self.img_paths)} & {len(self.lbl_paths)}'
        print(f'Total GT Label for {load_option}: {len(self.lbl_paths)}')
        print(f'Total Images for {load_option}: {len(self.img_paths)}')

        self.num_samples = len(self.img_paths)

    def load_sample(self, example_id):
        image = Image.open(self.img_paths[example_id])
        label = Image.open(self.lbl_paths[example_id])
        return np.array(image), np.array(label)

    def load_tfdataset(self):
        """
        Input pipeline
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.lbl_paths))
        dataset = dataset.shuffle(self.num_samples)

        # Load data
        def load_image(img_file, label_file):
            # load the raw data from the file as a string
            image = tf.io.read_file(img_file)
            label = tf.io.read_file(label_file)

            image = tf.image.decode_png(image)
            label = tf.image.decode_png(label)

            # Augmentations
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(label)
            image, label = image_scaling(image, label)

            image, label = random_crop_and_pad_image_and_labels(image, label, self.h, self.w, self.ignore_label)
            # resize the image to the desired size.
            return image, tf.squeeze(label)

        def final_process(image, label):
            image /= 255
            image.set_shape([None, self.h, self.w, 3])
            label.set_shape([None, self.h, self.w])
            return image, label

        dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(final_process, num_parallel_calls=AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset


class CityscapesDemoDataset:
    def __init__(self, dataset_dir, sequence='stuttgart_01'):
        # Check that the folder exists
        assert os.path.exists(dataset_dir) and os.path.isdir(dataset_dir), f"Dataset {dataset_dir} does not exist !"
        image_dir = os.path.join(dataset_dir, 'leftImg8bit_demoVideo', 'leftImg8bit', 'demoVideo', sequence)

        self.img_paths = sorted(glob.glob(os.path.join(image_dir, f'*leftImg8bit.png')))
        assert len(self.img_paths) != 0, f'Check data folder {self.img_paths}'
        print(f'Total Images for in Demoset: {len(self.img_paths)}')
        self.num_samples = len(self.img_paths)

    def load_tfdataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.img_paths)

        # Load data
        def load_image(img_file):
            # load the raw data from the file as a string
            rgb = tf.io.read_file(img_file)
            rgb = tf.image.decode_png(rgb)

            # Use `convert_image_dtype` to convert to floats in the [0,1] range.
            input = tf.image.convert_image_dtype(rgb, tf.float32)

            return rgb, input, img_file

        dataset = dataset.map(load_image)

        dataset = dataset.batch(1)
        return dataset
