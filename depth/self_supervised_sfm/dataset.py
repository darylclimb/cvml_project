import os

import numpy as np
import tensorflow as tf

from depth.self_supervised_sfm.utils import readlines

AUTOTUNE = tf.data.experimental.AUTOTUNE

########################
# Constants
#########################
KITTI_K = np.array([[0.58, 0, 0.5, 0],  # fx/width
                    [0, 1.92, 0.5, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float)


class KittiSFMDataset:
    def __init__(self, dataset_dir, load_option,
                 img_size, batch_size,
                 split='eigen_zhou',
                 frame_idx=(0, -1, 1)):
        self.h, self.w = img_size
        self.split = split
        self.batch_size = batch_size
        self.load_option = load_option
        self.dataset_dir = dataset_dir
        self.frame_idx = frame_idx
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}  # Correspond to image folder

        # Check that the folder exists
        assert os.path.exists(dataset_dir) and os.path.isdir(dataset_dir), f"Dataset {dataset_dir} does not exist !"
        if self.split == 'eigen_zhou':
            filename = os.path.join('splits', f'eigen_zhou/{load_option}_files.txt')
        else:
            raise NotImplementedError
        print(f'Loading from: {filename}')

        data_paths = readlines(filename)
        self.img_paths = []
        for i, line in enumerate(data_paths):
            # Image files
            folder, frame_idx, side = line.split()
            per_sample_imgs = []
            # Load sequence img
            for t in self.frame_idx:
                f_str = f"{int(frame_idx) + t:010d}"
                image_path = os.path.join(dataset_dir, folder, f"image_0{self.side_map[side]}/data", f_str + '.png')

                per_sample_imgs.append(image_path)

            self.img_paths.append(per_sample_imgs)

        print(f'Total Images for {load_option}: {len(self.img_paths)}')
        self.num_samples = len(self.img_paths)

    def load_tfdataset(self):
        inputs = {}
        # Intrinsic
        intrinsic = KITTI_K.copy()
        intrinsic[0, :] *= self.w
        intrinsic[1, :] *= self.h
        inputs['K'] = tf.convert_to_tensor(intrinsic, tf.float32)
        inputs['K_inv'] = tf.linalg.inv(inputs['K'])

        dataset = tf.data.Dataset.from_tensor_slices(self.img_paths)
        dataset = dataset.shuffle(self.num_samples)

        # Load data
        def load_sample(img_paths):
            # load the raw data from the file as a string
            image_cur = tf.io.read_file(img_paths[0])
            image_prev = tf.io.read_file(img_paths[1])
            image_next = tf.io.read_file(img_paths[2])

            image_cur = tf.image.decode_png(image_cur)
            image_prev = tf.image.decode_png(image_prev)
            image_next = tf.image.decode_png(image_next)

            image_cur = tf.cast(tf.image.resize(image_cur, [self.h, self.w]), tf.float32) / 255.
            image_prev = tf.cast(tf.image.resize(image_prev, [self.h, self.w]), tf.float32) / 255.
            image_next = tf.cast(tf.image.resize(image_next, [self.h, self.w]), tf.float32) / 255.

            if self.load_option == "train":
                if tf.random.uniform(()) > 0.5:
                    image_cur = tf.image.flip_left_right(image_cur)
                    image_prev = tf.image.flip_left_right(image_prev)
                    image_next = tf.image.flip_left_right(image_next)
            inputs['img'] = image_cur
            inputs['img-1'] = image_prev
            inputs['img1'] = image_next

            return inputs

        dataset = dataset.map(load_sample, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
