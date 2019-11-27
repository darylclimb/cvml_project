import argparse
import os
import tensorflow as tf

from segmentation.cityscape_reader import CityscapesDataset
from segmentation.loss import sparse_ce_logits_with_ignore
from segmentation.model import DeeplabV3

parser = argparse.ArgumentParser(description="Cityscapes")
parser.add_argument('--identifier', default="deeplabv3_densenet121")
parser.add_argument('--project_name', default="segmentation_cityscapes")

# Dataset
parser.add_argument('--data_dir', required=True, help="path data root")
parser.add_argument('--use_extra', action='store_true', help="use extra coarse data for training")
parser.add_argument('--ignore', default=255, help="Will be excluded from loss computation")

# Training
parser.add_argument('--lr', default=0.0003, help="learning rate")
parser.add_argument('--batch_size', default=6)
parser.add_argument('--num_classes', default=19+1, help="1 for bg")
parser.add_argument('--input_h', default=512, help="input height")
parser.add_argument('--input_w', default=1024, help="input width")
parser.add_argument('--iteration', default=120000, help="total steps")


def _gather(logits, labels, ignore):
    num_classes = tf.shape(logits)[-1]
    logits = tf.reshape(logits, [-1, num_classes])
    gt = tf.reshape(labels, [-1])

    indices = tf.squeeze(tf.where(tf.not_equal(gt, ignore)), 1)
    gt = tf.cast(tf.gather(gt, indices), tf.int32)
    logits = tf.gather(logits, indices)
    return logits, gt


def train(model, optimizer, dataset, epoch, loss_meter, metric_meter, ignore):
    for i, (inputs, labels) in enumerate(dataset):
        train_step(model, optimizer, inputs, labels, loss_meter, metric_meter, ignore)
        print(
            f'\rEpoch: [{epoch}] | '
            f'Iter: [{optimizer.iterations.numpy()}] | '
            f'Lr: {optimizer._decayed_lr(tf.float32):.5f} | '
            f'Train loss: {loss_meter.result():.4f} | ',
            f'iou: {metric_meter.result(): .4f}',
            end="")

@tf.function
def train_step(model, optimizer, inputs, labels, loss_meter, metric_meter, ignore):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = sparse_ce_logits_with_ignore(labels, logits, ignore)

        # compute metric
        logits, labels = _gather(logits, labels, ignore)
        y_pred = tf.argmax(logits, axis=-1)
        metric_meter.update_state(labels, y_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_meter(loss)


def val(model, dataset, epoch, loss_meter, metric_meter, ignore):
    for i, (inputs, labels) in enumerate(dataset):
        val_step(model, inputs, labels, loss_meter, metric_meter, ignore)
        print(f'\rEpoch {epoch}: {i} val loss = {loss_meter.result():.4f} | iou = {metric_meter.result():.4f}',
              end="")

@tf.function
def val_step(model, inputs, labels, loss_meter, metric_meter, ignore):
    logits = model(inputs, training=False)

    loss = sparse_ce_logits_with_ignore(labels, logits, ignore)
    loss_meter(loss)

    # compute metric
    logits, labels = _gather(logits, labels, ignore)
    y_pred = tf.argmax(logits, axis=-1)
    metric_meter.update_state(labels, y_pred)


def main(args):
    # Model
    model = DeeplabV3(input_shape=(args.input_h, args.input_w, 3),
                      num_classes=20)

    # Datasets
    train_dataset = CityscapesDataset(args.data_dir,
                                      'train',
                                      (args.input_h, args.input_w),
                                      batch_size=args.batch_size,
                                      use_extra=args.use_extra)
    val_dataset = CityscapesDataset(args.data_dir,
                                    'val',
                                    (args.input_h, args.input_w),
                                    batch_size=args.batch_size,
                                    use_extra=args.use_extra)

    train_tf_dataset = train_dataset.load_tfdataset()
    val_tf_dataset = val_dataset.load_tfdataset()

    # Loss, Optimiser, Metric
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(args.lr, end_learning_rate=0.00001,
                                                                     decay_steps=args.iteration,
                                                                     power=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    train_iou_meter = tf.keras.metrics.MeanIoU(args.num_classes)
    val_iou_meter = tf.keras.metrics.MeanIoU(args.num_classes)

    train_loss_meter = tf.keras.metrics.Mean(name='train_loss')
    val_loss_meter = tf.keras.metrics.Mean(name='test_loss')

    # Tensorboard and logging
    project_dir = os.getcwd()
    output_dir = os.path.join(project_dir, 'results', args.identifier)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    train_log_dir = os.path.join(output_dir, 'train_logs')
    val_log_dir = os.path.join(output_dir, 'val_logs')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Load states from optimiser and model if available
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = int(ckpt.step) + 1
    else:
        print("Initializing from scratch.")
        start_epoch = int(ckpt.step)

    # Start Train and Eval
    epochs = args.iteration // (train_dataset.num_samples // args.batch_size)
    for epoch in range(start_epoch, epochs):
        # Reset the metrics for the next epoch
        train_loss_meter.reset_states()
        val_loss_meter.reset_states()
        train_iou_meter.reset_states()
        val_iou_meter.reset_states()

        train(model, optimizer, train_tf_dataset, epoch, train_loss_meter, train_iou_meter, args.ignore)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_meter.result(), step=epoch)
            tf.summary.scalar('miou', train_iou_meter.result(), step=epoch)

        val(model, val_tf_dataset, epoch, val_loss_meter, val_iou_meter, args.ignore)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss_meter.result(), step=epoch)
            tf.summary.scalar('miou', val_iou_meter.result(), step=epoch)

        print(f'\nEpoch: {epoch}, '
              f'Train Loss: {train_loss_meter.result():.4f}, '
              f'Val Loss: {val_loss_meter.result():.4f}',
              f'Val iou: {val_iou_meter.result():.4f}',
              f'Train iou: {train_iou_meter.result():.4f}')

        # save and increment
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
