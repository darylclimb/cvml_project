import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from disparitynet import DisparityNet
from utils import readlines, generate_depth_map, visualize_colormap, compute_errors, disp_to_depth

parser = argparse.ArgumentParser(description="Disparity Project")
parser.add_argument('--identifier', default="sfm_resnet34")
parser.add_argument('--dataset_dir', help='path to dataset')
parser.add_argument('--demo_set', default="2011_09_30/2011_09_30_drive_0027_sync")
parser.add_argument("--input_h", default=192)
parser.add_argument("--input_w", default=640)
parser.add_argument('--eval_split', default='eigen',
                    help="which split to run eval on")

PROJECT_DIR = os.getcwd()
HOME = str(pathlib.Path.home())

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


class Evaluator:
    def __init__(self, params, output_dir):
        self.dataset_dir = params.dataset_dir
        self.demo_set = params.demo_set
        self.output_dir = output_dir
        self.params = params
        self.models = {'disparity': DisparityNet(input_shape=(params.input_h, params.input_w, 3))}
        self.load_checkpoint(self.models['disparity'], os.path.join(output_dir, 'disparity_model'))

        # Datasets
        self.data_paths = readlines(os.path.join('splits', params.eval_split, 'test_files.txt'))
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}  # Correspond to image folder
        self.img_paths = []
        self.gt_depths = []
        for i, line in enumerate(self.data_paths):
            folder, frame_idx, side = line.split()
            f_str = f"{int(frame_idx) + 0:010d}"
            image_path = os.path.join(self.dataset_dir, folder, f"image_0{self.side_map[side]}/data", f_str + '.png')
            self.img_paths.append(image_path)

            calib_dir = os.path.join(self.dataset_dir, folder.split("/")[0])
            velo_filename = os.path.join(self.dataset_dir, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(int(frame_idx)))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

            self.gt_depths.append(gt_depth)

        print(f'Total Images: {len(self.img_paths)}')

    def load_checkpoint(self, model, output_dir):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
        manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("No weights to Restores.")
            raise ValueError(f'No Weight to restore in {output_dir}')

    def do_demo(self, folder):
        save_dir = os.path.join(self.output_dir, 'predictions')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        folder_dir = os.path.join(self.dataset_dir, folder, 'image_02', 'data', '*.*')
        images_files = sorted(glob.glob(folder_dir))
        print(f'doing demo on {self.demo_set}... ')
        print(f'saving prediction to {save_dir}...')
        for i, img_path in enumerate(images_files):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.params.input_w, self.params.input_h))
            img_input = tf.expand_dims(tf.convert_to_tensor(img, tf.float32) / 255., 0)
            outputs = self.val_step(img_input)

            disp = np.squeeze(outputs['disparity0'].numpy())
            disp = visualize_colormap(disp)
            save_path = os.path.join(save_dir, f'{i}.png')

            big_image = np.zeros(shape=(self.params.input_h * 2, self.params.input_w, 3))
            big_image[:self.params.input_h, ...] = img
            big_image[self.params.input_h:, ...] = disp
            cv2.imwrite(save_path, cv2.cvtColor(big_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("\n-> Done!\n")

    def eval_depth(self):
        pred_depths = []
        pred_disps = []
        errors = []
        ratios = []

        # Predict
        print('doing evaluation...')
        for i, img_path in enumerate(self.img_paths):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.params.input_w, self.params.input_h))
            img = tf.expand_dims(tf.convert_to_tensor(img, tf.float32) / 255., 0)
            outputs = self.val_step(img)
            _, depth = disp_to_depth(outputs['disparity0'], min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
            depth *= 0.54

            pred_depths.append(depth.numpy())
            pred_disps.append(np.squeeze(outputs['disparity0'].numpy()))

        for i in range(len(pred_depths)):
            gt_depth = self.gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = pred_depths[i][0]
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            # Median scaling
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt_depth, pred_depth))

        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!\n")

    @tf.function
    def val_step(self, inputs):
        return self.models['disparity'](inputs, training=False)


if __name__ == '__main__':
    params = parser.parse_args()
    output_dir = os.path.join(PROJECT_DIR, 'results', params.identifier)

    c = Evaluator(params, output_dir)
    c.eval_depth()
    c.do_demo(params.demo_set)
