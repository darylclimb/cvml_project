import tensorflow as tf

from segmentation.layers import SeparableConvBlock, ConvBlock

NAME = 'deeplabv3plus'


class DeeplabV3(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 num_classes=20,
                 rate_scale=1,
                 depth=256,
                 backbone='resnet34',
                 data_format='channels_last'):
        super().__init__()
        # depth: The depth of the ResNet unit output.
        self.depth = depth

        self.num_classes = num_classes
        channel_axis = 1 if data_format == 'channels_first' else -1

        # Get resnet50 encoder
        layer_names = ['conv1_relu',
                       'conv2_block3_out',
                       'conv3_block4_out',
                       'conv4_block6_out',
                       'conv5_block3_out']

        layer_names = ['conv1/relu',
                       'pool2_conv',
                       'pool3_conv',
                       'pool4_conv',
                       'relu']

        # base_model = tf.keras.applications.ResNet50V2(weights='imagenet',
        #                                             input_shape=input_shape, include_top=False)
        base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        # print(base_model.summary())
        # d
        n_filters_low_level = 48

        layers = [base_model.get_layer(name).output for name in layer_names]
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=layers)

        # Low Level Branch
        self.low_level_features_conv1_1 = SeparableConvBlock(filters=n_filters_low_level,
                                                             kernel_size=(1, 1),
                                                             data_format=data_format,
                                                             use_bn=True,
                                                             activation=tf.nn.relu,
                                                             name=NAME + '_low_level_feature_conv1_1')

        # ASPP Block Branch
        self.aspp = AtrousSpatialPyramidPooling(n_filters=256,
                                                rate_scale=rate_scale,
                                                data_format=data_format, name=NAME + '_aspp')

        # Output
        self.concat = tf.keras.layers.Concatenate(axis=channel_axis)

        self.out_conv0 = SeparableConvBlock(filters=256,
                                            kernel_size=(3, 3),
                                            data_format=data_format,
                                            use_bn=True, activation=tf.nn.relu,
                                            name=NAME + '_out_conv0_3x3')

        self.out_conv1 = SeparableConvBlock(filters=256,
                                            kernel_size=(3, 3),
                                            data_format=data_format,
                                            use_bn=True, activation=tf.nn.relu,
                                            name=NAME + '_out_conv1_3x3')

        self.logits = ConvBlock(filters=num_classes,
                                kernel_size=(1, 1),
                                data_format=data_format,
                                use_bn=False,
                                name=NAME + '_logits')
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, training=False, **kwargs):
        C1, C2, C3, C4, C5 = self.backbone(inputs, training=training)
        if training:
            C5 = self.dropout(C5)

        low_level_features = C3
        low_level_features = self.low_level_features_conv1_1(low_level_features)

        aspp = C5
        aspp = self.aspp(aspp, training=training, **kwargs)
        aspp = tf.image.resize(aspp, tf.shape(C3)[1:3])

        # Concat aspp and low level features
        net = self.concat([aspp, low_level_features])

        # Apply all the conv blocks
        net = self.out_conv0(net)
        net = self.out_conv1(net)
        net = self.logits(net)

        # Final upsampling part
        net = tf.image.resize(net, tf.shape(inputs)[1:3])

        return net


class AtrousSpatialPyramidPooling(tf.keras.Model):
    """
        ASPP consists of
        (a) one 1×1 convolution,
        (b) three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and
        (c) the image-level features as described in the paper (see Readme)
    """

    def __init__(self, n_filters=256, rate_scale=1, data_format='channels_last', name=None):
        super().__init__(name='')

        # If use_bn becomes True, then need to specify normalization_type
        self.conv_block3_3_0 = SeparableConvBlock(filters=n_filters,
                                                  dilation_rate=6 * rate_scale,
                                                  data_format=data_format,
                                                  use_bn=True, activation=tf.nn.relu,
                                                  name=name + '_conv_block3_3_0')
        self.conv_block3_3_1 = SeparableConvBlock(filters=n_filters,
                                                  dilation_rate=12 * rate_scale,
                                                  data_format=data_format,
                                                  activation=tf.nn.relu,
                                                  use_bn=True, name=name + '_conv_block3_3_1')
        self.conv_block3_3_2 = SeparableConvBlock(filters=n_filters,
                                                  dilation_rate=18 * rate_scale,
                                                  data_format=data_format,
                                                  activation=tf.nn.relu,
                                                  use_bn=True, name=name + '_conv_block3_3_2')

        self.conv_block1_1 = SeparableConvBlock(filters=n_filters,
                                                dilation_rate=rate_scale,
                                                kernel_size=(1, 1),
                                                data_format=data_format,
                                                activation=tf.nn.relu,
                                                use_bn=False,
                                                name=name + '_conv_block1_1')

        # Feature Level Layer
        self.feature_level_reduce = tf.keras.layers.Lambda(lambda y: tf.reduce_mean(y, [1, 2], keepdims=True),
                                                           name=name + '_feat_level_pooling')
        self.feature_level_conv = ConvBlock(filters=n_filters, kernel_size=(1, 1),
                                            data_format=data_format,
                                            use_bn=True,
                                            activation=tf.nn.relu,
                                            name=NAME + '_feat_level_conv')

        # Output
        self.output_concatenate = tf.keras.layers.Concatenate(axis=3, name=name + '_output_concat')
        self.output_conv1_1 = ConvBlock(filters=n_filters,
                                        kernel_size=(1, 1),
                                        data_format=data_format,
                                        use_bn=True,
                                        activation=tf.nn.relu,
                                        name=NAME + '_output_conv')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, input_tensor, training=False, **kwargs):
        """
        :param input_tensor: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
        :return: network layer with aspp applied to it.
        """
        # Do we train bn layers
        train_bn = kwargs.get('train_bn', False)

        # 3x3 Conv Blocks
        block0 = self.conv_block3_3_0(input_tensor, **kwargs)
        block1 = self.conv_block3_3_1(input_tensor, **kwargs)
        block2 = self.conv_block3_3_2(input_tensor, **kwargs)

        # 1x1 Conv Block
        conv1_1 = self.conv_block1_1(input_tensor, **kwargs)

        # Feature / Image level layer sequence
        feat_level = self.feature_level_reduce(input_tensor)
        feat_level = self.feature_level_conv(feat_level)
        target_size = input_tensor.shape.as_list()[1:3]
        if None in target_size:
            target_size = tf.shape(input_tensor)[1:3]
        feat_level = tf.image.resize(feat_level, size=target_size)

        # Concatenate all
        net = self.output_concatenate([block0, block1, block2, conv1_1, feat_level])

        # Apply 1x1 conv
        net = self.output_conv1_1(net)
        if training:
            net = self.dropout(net)
        return net
