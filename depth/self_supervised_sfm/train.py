import argparse
import datetime
import os

import tensorflow as tf

from dataset import KittiSFMDataset
from disparitynet import DisparityNet
from posenet import PoseNet
from utils import pixel_coord, ssim_loss, smooth_loss, bilinear_sampler, forwardproject, backproject, disp_to_depth

parser = argparse.ArgumentParser(description="Disparity Project")
parser.add_argument('--identifier', default="sfm_resnet18")
parser.add_argument('--data_dir')
parser.add_argument("--input_h", default=192)
parser.add_argument("--input_w", default=640)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--epochs", default=50)
parser.add_argument("--num_scales", default=4)
parser.add_argument("--num_input_frames", default=2, help='num of frames as input to posenet')
parser.add_argument("--frame_ids", default=[0, -1, 1], help='frames to load ')
parser.add_argument("--draw_every_iter", default=1000)

PROJECT_DIR = os.getcwd()
MIN_DEPTH = 1e-3
MAX_DEPTH = 80


class Trainer:
    def __init__(self, params, output_dir):
        self.params = params

        # Models
        self.models = {}
        self.models['disparity'] = DisparityNet(input_shape=(params.input_h, params.input_w, 3))

        self.models['pose'] = PoseNet(input_shape=(params.input_h, params.input_w, 3 * params.num_input_frames),
                                      num_input_frames=params.num_input_frames)

        # Datasets
        train_dataset = KittiSFMDataset(params.data_dir, 'train',
                                        (params.input_h, params.input_w),
                                        batch_size=params.batch_size,
                                        frame_idx=params.frame_ids)
        val_dataset = KittiSFMDataset(params.data_dir, 'val',
                                      (params.input_h, params.input_w),
                                      frame_idx=params.frame_ids,
                                      batch_size=params.batch_size)

        self.train_dataset = train_dataset.load_tfdataset()
        self.val_dataset = val_dataset.load_tfdataset()

        # Optimizer
        self.total_iteration = (train_dataset.num_samples // params.batch_size) * params.epochs
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(0.0002, end_learning_rate=0.000001,
                                                                         decay_steps=self.total_iteration,
                                                                         power=0.5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_fn)

        # Tensorboard & Meters
        train_log_dir = os.path.join(output_dir, 'train_logs')
        val_log_dir = os.path.join(output_dir, 'val_logs')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(val_log_dir)

        self.train_meter = {
            'ssim': tf.keras.metrics.Mean(name='ssim'),
            'l1': tf.keras.metrics.Mean(name='l1'),
            'smooth': tf.keras.metrics.Mean(name='smooth'),
        }

        self.val_meter = {
            'ssim': tf.keras.metrics.Mean(name='ssim'),
            'l1': tf.keras.metrics.Mean(name='l1'),
            'smooth': tf.keras.metrics.Mean(name='smooth'),
        }

        self.step = 0
        # Load states from optimiser and model if available
        self.ckpt_disp, self.manager_disp = self.setup_logger(self.models['disparity'],
                                                              os.path.join(output_dir, 'disparity_model'))
        self.ckpt_pose, self.manager_pose = self.setup_logger(self.models['pose'],
                                                              os.path.join(output_dir, 'pose_model'))
        self.start_epoch = int(self.ckpt_disp.step) + 1 if self.manager_disp.latest_checkpoint else int(
            self.ckpt_disp.step)

        print("Starting training step {}".format(self.ckpt_disp.step.numpy()))

        # Helpers
        self.pix_coords = pixel_coord(params.batch_size, params.input_h, params.input_w, True)  # [b, 3, npoints]

    def setup_logger(self, model, out_dir):
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=model)
        manager = tf.train.CheckpointManager(ckpt, out_dir, max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint)
        return ckpt, manager

    def train(self):
        for epoch in range(self.start_epoch, self.params.epochs):
            [self.train_meter[k].reset_states() for k, v in self.train_meter.items()]
            [self.val_meter[k].reset_states() for k, v in self.val_meter.items()]
            # Train
            for i, inputs in enumerate(self.train_dataset):
                loss, outputs = self.train_step(inputs)
                print(
                    f'\rEpoch: [{epoch}/{self.params.epochs}] | 'f'Iter: [{self.optimizer.iterations.numpy()}/{self.total_iteration}] | '
                    f'Lr: {self.optimizer._decayed_lr(tf.float32):.5f} | '
                    f"ssim: {self.train_meter['ssim'].result():.4f} | ",
                    f"l1: {self.train_meter['l1'].result():.4f} | ",
                    f"smooth: {self.train_meter['smooth'].result():.10f} | ",
                    f"total loss: {loss['loss']:.4f} | ",
                    end="")

                if i % self.params.draw_every_iter == 0:
                    with self.train_summary_writer.as_default():
                        tf.summary.image('disparity', outputs['disparity0'], step=epoch)
                        tf.summary.image('depth', outputs['depth0'], step=epoch)

                        stack_prediction_pred = tf.concat([outputs['pred-10'], inputs['img'], outputs['pred10']],
                                                          axis=1)
                        stack_prediction_gt = tf.concat([inputs['img-1'], inputs['img'], inputs['img1']], axis=1)
                        tf.summary.image('predictions', stack_prediction_pred, step=epoch)
                        tf.summary.image('groundtruth', stack_prediction_gt, step=epoch)

            # Validation
            for i, inputs in enumerate(self.val_dataset):
                self.val_step(inputs)
                print(
                    f'\rEpoch: [{epoch}/{params.epochs}] | '
                    f"ssim: {self.val_meter['ssim'].result():.4f} | ",
                    f"l1: {self.val_meter['l1'].result():.4f} | ",
                    f"smooth: {self.val_meter['smooth'].result():.4f} | ",
                    end="")

            with self.train_summary_writer.as_default():
                tf.summary.scalar('ssim', self.train_meter['ssim'].result(), step=epoch)
                tf.summary.scalar('l1', self.train_meter['l1'].result(), step=epoch)
                tf.summary.scalar('smooth', self.train_meter['smooth'].result(), step=epoch)

            with self.test_summary_writer.as_default():
                tf.summary.scalar('ssim', self.val_meter['ssim'].result(), step=epoch)
                tf.summary.scalar('l1', self.val_meter['l1'].result(), step=epoch)
                tf.summary.scalar('smooth', self.val_meter['smooth'].result(), step=epoch)

            # save and increment
            save_path = self.manager_disp.save()
            save_path = self.manager_pose.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt_disp.step), save_path))
            self.ckpt_disp.step.assign_add(1)
            self.ckpt_pose.step.assign_add(1)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self.models['disparity'](inputs['img'], training=True)
            outputs.update(self.predict_pose(inputs))
            outputs.update(self.view_synthesis(inputs, outputs))
            loss = self.criterions(inputs, outputs)

        trainable_params = self.models['disparity'].trainable_variables + self.models['pose'].trainable_variables
        gradients = tape.gradient(loss['loss'], trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))

        # Update moving average
        [self.train_meter[k](loss[k]) for k, v in self.train_meter.items()]

        return loss, outputs

    @tf.function
    def val_step(self, inputs):
        outputs = self.models['disparity'](inputs['img'], training=False)
        outputs.update(self.predict_pose(inputs))
        outputs.update(self.view_synthesis(inputs, outputs))
        loss = self.criterions(inputs, outputs)

        # Update moving average
        [self.val_meter[k](loss[k]) for k, v in self.val_meter.items()]

    def criterions(self, inputs, outputs):
        loss_dict = {}
        total_l1_loss = 0.
        total_ssim_loss = 0.
        total_smooth_loss = 0.

        for scale in range(self.params.num_scales):
            l1_losses = []
            ssim_losses = []
            for f_i in self.params.frame_ids[1:]:
                target_rgb = inputs['img']
                pred_rgb = outputs[f'pred{f_i}{scale}']

                # L1 Loss
                abs_diff = tf.abs(target_rgb - pred_rgb)
                l1_loss = tf.reduce_mean(abs_diff, axis=-1, keepdims=True)  # [b, h, w, 1]
                l1_losses.append(l1_loss)

                # SSIM Loss
                ssim = tf.reduce_mean(ssim_loss(target_rgb, pred_rgb), axis=-1, keepdims=True)
                ssim_losses.append(ssim)

            ssim_losses = tf.concat(ssim_losses, -1)
            l1_losses = tf.concat(l1_losses, -1)
            if scale == 0:
                outputs['l1_error'] = l1_losses

            # Automasking
            identity_l1_losses = []
            identity_ssim_losses = []
            for f_i in self.params.frame_ids[1:]:
                target_rgb = inputs['img']
                source_rgb = inputs[f'img{f_i}']

                # L1 Loss
                abs_diff = tf.abs(source_rgb - target_rgb)
                l1_loss = tf.reduce_mean(abs_diff, axis=-1, keepdims=True)
                identity_l1_losses.append(l1_loss)

                # SSIM Loss [b, h, w, 1]
                ssim = tf.reduce_mean(ssim_loss(source_rgb, target_rgb), axis=-1, keepdims=True)
                identity_ssim_losses.append(ssim)

            identity_ssim_losses = tf.concat(identity_ssim_losses, -1)
            identity_l1_losses = tf.concat(identity_l1_losses, -1)

            identity_l1_losses += tf.random.normal(identity_l1_losses.shape) * 0.00001  # Break ties
            identity_ssim_losses += tf.random.normal(identity_ssim_losses.shape) * 0.00001  # Break ties

            combined_l1 = tf.concat((identity_l1_losses, l1_losses), axis=-1)
            combined_ssim = tf.concat((identity_ssim_losses, ssim_losses), axis=-1)

            combined_l1 = tf.reduce_min(combined_l1, axis=-1)
            combined_ssim = tf.reduce_min(combined_ssim, axis=-1)

            _ssim_loss = tf.reduce_mean(combined_ssim) * 0.85
            _l1_loss = tf.reduce_mean(combined_l1) * 0.15
            total_l1_loss += _l1_loss
            total_ssim_loss += _ssim_loss

            # Disparity smoothness
            disparity = outputs[f'disparity{scale}']
            mean_disp = tf.reduce_mean(disparity, [1, 2], keepdims=True)
            norm_disp = disparity / (mean_disp + 1e-7)

            h = self.params.input_h // (2 ** scale)
            w = self.params.input_w // (2 ** scale)
            color_resized = tf.image.resize(target_rgb, (h, w))

            smooth = smooth_loss(norm_disp, color_resized) * 1e-3
            total_smooth_loss += smooth

        total_smooth_loss /= self.params.num_scales
        total_ssim_loss /= self.params.num_scales
        total_l1_loss /= self.params.num_scales
        loss_dict['ssim'] = total_ssim_loss
        loss_dict['l1'] = total_l1_loss
        loss_dict['smooth'] = total_smooth_loss
        loss_dict['loss'] = total_smooth_loss + total_ssim_loss + total_l1_loss
        return loss_dict

    def predict_pose(self, inputs):
        """
        Compute pose wrt to each source frame
        """
        output = {}
        for f_i in self.params.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = tf.concat([inputs[f'img{f_i}'], inputs['img']], -1)
            else:
                pose_inputs = tf.concat([inputs['img'], inputs[f'img{f_i}']], -1)
            axisangle, translation, M = self.models['pose'](pose_inputs, invert=(f_i < 0))

            output[f'axisangle{f_i}'] = axisangle
            output[f'translation{f_i}'] = translation
            output[f'M{f_i}'] = M

        return output

    def view_synthesis(self, inputs, outputs):
        """
        Warped prediction based on predicted depth and pose
        Args:
            inputs:
                'disparity':    [b, h, w, 1]
                'img':          [b, h, w, 3]
        """
        for scale in range(self.params.num_scales):
            disp = outputs[f'disparity{scale}']
            disp = tf.image.resize(disp, [self.params.input_h, self.params.input_w])

            _, depth = disp_to_depth(disp, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
            outputs[f'depth{scale}'] = depth

            for i, frame_id in enumerate(self.params.frame_ids[1:]):
                source = inputs[f'img{frame_id}']
                T = outputs[f'M{frame_id}']

                # depth2pcl
                cam_points = backproject(self.pix_coords, depth, inputs['K_inv'])

                # pcl2pix
                proj_mat = tf.matmul(inputs['K'], T)
                pix_coords = forwardproject(cam_points, proj_mat, self.params.input_h,
                                            self.params.input_w)  # [b, h, w, 2]

                # Warped source to target
                projected_img = bilinear_sampler(source, pix_coords)  # [b, h, w, 3]
                outputs[f'pred{frame_id}{scale}'] = projected_img
        return outputs


if __name__ == '__main__':
    params = parser.parse_args()
    output_dir = os.path.join(PROJECT_DIR, 'results', params.identifier)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print(f'Start: {params.identifier}', datetime.datetime.now())
    t = Trainer(params, output_dir)
    t.train()
