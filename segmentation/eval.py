import os
from segmentation.cityscape_reader import CityscapesDemoDataset
import tensorflow as tf
import argparse
import numpy as np
import cv2

from segmentation.labels import cityscapes_mask_colors
from segmentation.model import DeeplabV3

parser = argparse.ArgumentParser(description="Cityscapes")
parser.add_argument('--project_name', default="segmentation_cityscapes")
parser.add_argument('--identifier', default="deeplabv3_densenet121")
parser.add_argument('--data_dir', required=True, help="path data root")


def label2rgb(label, img=None, alpha=0.5, cmap=None):
    label_rgb = cmap[label]
    if img is not None:
        label_rgb = alpha * label_rgb + (1 - alpha) * img
        label_rgb = label_rgb.astype(np.uint8)
    return label_rgb


@tf.function
def predict(model, inputs):
    logits = model(inputs, training=False)
    return logits


def val(model, dataset, save_dir):
    for i, (rgb, inputs, img_path) in enumerate(dataset):
        rgb = tf.squeeze(rgb).numpy()

        # Predict
        logits = predict(model, inputs)
        pred = tf.squeeze(tf.argmax(logits, -1)).numpy().astype(np.uint8)

        # Save Images
        pred_color = label2rgb(pred, img=rgb, cmap=cityscapes_mask_colors)
        mask_path = os.path.join(save_dir, f'{int(i):04d}.png')
        cv2.imwrite(mask_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))



def evaluate(args):
    project_dir = os.getcwd()
    output_dir = os.path.join(project_dir, 'results', args.identifier)
    save_dir = os.path.join(output_dir, 'demo')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model = DeeplabV3(input_shape=None)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
    manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("No weights to Restores.")
        raise

    val_dataset = CityscapesDemoDataset(args.data_dir, sequence='stuttgart_02')
    val_dataset = val_dataset.load_tfdataset()

    val(model, val_dataset, save_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    evaluate(args)
