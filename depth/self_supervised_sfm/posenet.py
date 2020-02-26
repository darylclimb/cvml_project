import tensorflow as tf

import classification_models.tfkeras as ctk

# Without this, I am facing out of memory issue
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def translation_mat(trans_vec):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    batch_size = tf.shape(trans_vec)[0]

    one = tf.ones([batch_size, 1, 1], dtype=tf.float32)
    zero = tf.zeros([batch_size, 1, 1], dtype=tf.float32)

    T = tf.concat([one, zero, zero, trans_vec[:, :, :1],
                   zero, one, zero, trans_vec[:, :, 1:2],
                   zero, zero, one, trans_vec[:, :, 2:3],
                   zero, zero, zero, one], axis=2)

    T = tf.reshape(T, [batch_size, 4, 4])

    return T


def rot_from_axisangle(vec):
    """
    Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)

    Args:
        vec: [b, 1, 3]
    Returns:
        rotation matrix: [b, 4, 4]
    """
    angle = tf.norm(vec, 'euclidean', 2, True)
    axis = vec / (angle + 1e-7)

    ca = tf.cos(angle)
    sa = tf.sin(angle)
    C = 1 - ca

    x = tf.expand_dims(axis[..., 0], 1)
    y = tf.expand_dims(axis[..., 1], 1)
    z = tf.expand_dims(axis[..., 2], 1)

    # [b, 1, 1]
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # [b, 1, 1]
    one = tf.ones_like(zxC, dtype=tf.float32)
    zero = tf.zeros_like(zxC, dtype=tf.float32)

    rot_matrix = tf.concat([
        x * xC + ca, xyC - zs, zxC + ys, zero,
        xyC + zs, y * yC + ca, yzC - xs, zero,
        zxC - ys, yzC + xs, z * zC + ca, zero,
        zero, zero, zero, one
    ], axis=2)

    rot_matrix = tf.reshape(rot_matrix, [-1, 4, 4])

    return rot_matrix


class PoseNet(tf.keras.Model):
    def __init__(self, input_shape, num_input_frames):
        super().__init__()

        self.num_input_frames = num_input_frames

        backbone_fn = ctk.Classifiers.get('resnet18')[0]
        encoder = backbone_fn(input_shape=input_shape, weights=None, include_top=False)
        layer_names = ['relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1']
        layers = [encoder.get_layer(name).output for name in layer_names]
        self.backbone = tf.keras.Model(inputs=encoder.input, outputs=layers)

        self.conv1 = tf.keras.Sequential([tf.keras.layers.Conv2D(256, (1, 1), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.relu)])

        self.conv2 = tf.keras.Sequential([tf.keras.layers.Conv2D(256, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.relu)])

        self.conv3 = tf.keras.Sequential([tf.keras.layers.Conv2D(256, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.relu)])

        self.conv4 = tf.keras.Sequential([tf.keras.layers.Conv2D(6 * num_input_frames, (1, 1), (1, 1), 'same')])

    def call(self, inputs, invert, training=False, **kwargs):
        """
        Args:
            inputs:     Color image [b, h, w, 3*sequence_length]
        Returns:
            axisangle:      [b, 2, 3]
            translation:    [b, 2, 3]
            M:              [b, 4, 4]
        """
        # print(inputs.shape)
        e1, e2, e3, e4, x = self.backbone(inputs, **kwargs)
        batch_size = tf.shape(x)[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.reduce_mean(x, [1, 2], keepdims=True)
        x = tf.reshape(x, [batch_size, 2, 1, 6])
        x = x * 0.01

        axisangle = x[..., :3]
        translation = x[..., 3:]

        R = rot_from_axisangle(axisangle[:, 0])

        # [b, 1, 3]
        t = translation[:, 0]
        if invert:
            R = tf.transpose(R, [0, 2, 1])
            t *= -1
        t = translation_mat(t)

        if invert:
            M = tf.matmul(R, t)
        else:
            M = tf.matmul(t, R)

        return axisangle, translation, M
