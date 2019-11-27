from enum import Enum

import tensorflow as tf


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, dilation_rate=1, kernel_size=(3, 3),
                 data_format='channels_last', padding='same',
                 use_bn=False, activation=tf.nn.relu, name=None):
        name = name if name else ''
        super().__init__(name=name)

        use_bias = False if use_bn else True
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                             dilation_rate=(dilation_rate, dilation_rate),
                                             kernel_initializer='he_uniform', data_format=data_format,
                                             kernel_regularizer=tf.keras.regularizers.l2(0.00004),
                                             use_bias=use_bias, padding=padding, name=name + '_conv')

        self.bn = (tf.keras.layers.BatchNormalization(name=name, momentum=0.9) if use_bn else None)

        self.activation = tf.keras.layers.Activation(activation, name=name + '_activation') if activation else None

    def call(self, input_tensor, **kwargs):
        x = self.conv2d(input_tensor)
        x = self.bn(x) if self.bn else x
        x = self.activation(x) if self.activation else x
        return x


class SeparableConvBlock(tf.keras.Model):
    def __init__(self, filters, dilation_rate=1, kernel_size=(3, 3), stride=1,
                 data_format='channels_last', use_bn=False, activation=None, name=None):
        name = name if name else ''
        super().__init__(name=name)

        use_bias = False if use_bn else True

        self.activation1 = tf.keras.layers.Activation(activation, name=name + '_activation1') if activation else None
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=(stride, stride),
                                                              dilation_rate=(dilation_rate, dilation_rate),
                                                              padding='same', use_bias=use_bias,
                                                              name='sepconv' + '_depthwise')

        self.bn1 = (tf.keras.layers.BatchNormalization(momentum=0.9) if use_bn else None)

        self.activation2 = tf.keras.layers.Activation(activation, name=name + '_activation2') if activation else None
        self.conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                           use_bias=use_bias, name='sepconv' + '_pointwise')

        self.bn2 = (tf.keras.layers.BatchNormalization(momentum=0.9) if use_bn else None)
        self.activation3 = tf.keras.layers.Activation(activation, name=name + '_activation3') if activation else None

    def call(self, input_tensor, **kwargs):
        train_bn = kwargs.get('train_bn', False)
        # Depthwise
        x = self.activation1(input_tensor) if self.activation1 else input_tensor
        x = self.depthwise_conv(x)
        x = self.bn1(x) if self.bn1 else x

        # Pointwise
        x = self.activation2(x) if self.activation2 else x
        x = self.conv(x)
        x = self.bn2(x) if self.bn2 else x
        net = self.activation3(x) if self.activation3 else x

        return net


class SCSEBlock(tf.keras.Model):
    def __init__(self, channels, reduction=16, name="scse_block", data_format="channels_last"):
        super().__init__(name=name)
        mid_channels = channels // reduction
        self.channel_axis = 1 if data_format == 'channels_first' else -1
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=name + "_gap")

        self.channel_excitation = tf.keras.Sequential((
            tf.keras.layers.Dense(mid_channels,
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  activation='relu',
                                  name=name + "_dense1",
                                  use_bias=False),
            tf.keras.layers.Dense(channels,
                                  activation='sigmoid',
                                  kernel_initializer=tf.keras.initializers.glorot_normal,
                                  name=name + "_dense2",
                                  use_bias=False)))

        self.spatial_se = tf.keras.layers.Conv2D(channels, kernel_size=1, activation='sigmoid',
                                                 strides=1, padding='same', use_bias=False, data_format=data_format)

    def call(self, input_tensor, **kwargs):
        shape = tf.shape(input_tensor)
        # Channel Attention
        chn_se = self.gap(input_tensor)
        chn_se = self.channel_excitation(tf.reshape(chn_se, (shape[0], 1, 1, shape[self.channel_axis])))
        chn_se = tf.keras.layers.multiply([input_tensor, chn_se])

        # Spatial Excitation
        spa_se = self.spatial_se(input_tensor)
        spa_se = tf.keras.layers.multiply([input_tensor, spa_se])

        return tf.keras.layers.add([chn_se, spa_se])


# class DenseAsppBlock(tf.keras.Model):
#     def __init__(self):