import tensorflow as tf

import classification_models.tfkeras as ctk

# Without this, I am facing out of memory issue
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class DisparityNet(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()

        backbone_fn = ctk.Classifiers.get('resnet34')[0]
        encoder = backbone_fn(input_shape=input_shape, weights='imagenet', include_top=False)
        layer_names = ['relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1', 'relu1']
        layers = [encoder.get_layer(name).output for name in layer_names]
        self.backbone = tf.keras.Model(inputs=encoder.input, outputs=layers)

        # Decoder network
        self.decoder = Decoder()

    def call(self, inputs, training=False, **kwargs):
        out = self.decoder(self.backbone(inputs, **kwargs))
        return out


class DecoderBlock(tf.keras.Model):
    def __init__(self, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = tf.keras.Sequential([tf.keras.layers.Conv2D(out_channels, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.BatchNormalization(momentum=0.9),
                                          tf.keras.layers.Activation(tf.nn.elu)])

        self.conv2 = tf.keras.Sequential([tf.keras.layers.Conv2D(out_channels, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.BatchNormalization(momentum=0.9),
                                          tf.keras.layers.Activation(tf.nn.elu)])

    def call(self, feat1, feat2=None):
        """
        feat1: deeper layer feat
        feat2: shallow layer feat
        """
        x = tf.image.resize(feat1, tf.shape(feat1)[1:3] * 2)
        x = self.conv1(x)
        if feat2 is not None:
            x = tf.concat([x, feat2], axis=-1)
        x = self.conv2(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__(name='decoder')

        self.layer1 = DecoderBlock(256)
        self.layer2 = DecoderBlock(128)
        self.layer3 = DecoderBlock(64)
        self.layer4 = DecoderBlock(32)
        self.layer5 = DecoderBlock(16)

        self.disp1 = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.sigmoid)])

        self.disp2 = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.sigmoid)])

        self.disp3 = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.sigmoid)])

        self.disp4 = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (3, 3), (1, 1), 'same'),
                                          tf.keras.layers.Activation(tf.nn.sigmoid)])

    def call(self, inputs, **kwargs):
        output = {}
        conv_1, conv_2, conv_3, conv_4, conv_5 = inputs
        x = self.layer1(conv_5, conv_4)
        x = self.layer2(x, conv_3)
        output['disparity3'] = self.disp1(x)
        x = self.layer3(x, conv_2)
        output['disparity2'] = self.disp2(x)
        x = self.layer4(x, conv_1)
        output['disparity1'] = self.disp3(x)
        x = self.layer5(x, None)
        output['disparity0'] = self.disp4(x)
        return output
