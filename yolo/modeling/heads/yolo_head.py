import tensorflow as tf
from yolo.modeling.layers import nn_blocks


class YoloHead(tf.keras.layers.Layer):
  """YOLO Prediction Head"""

  def __init__(self,
               min_level,
               max_level,
               classes=80,
               boxes_per_level=3,
               output_extras=0,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation=None,
               **kwargs):
    """Yolo Prediction Head initialization function.

    Args:
      min_level: `int`, the minimum backbone output level.
      max_level: `int`, the maximum backbone output level.
      classes: `int`, number of classes per category.
      boxes_per_level: `int`, number of boxes to predict per level.
      output_extras: `int`, number of additional output channels that the head.
        should predict for non-object detection and non-image classification
        tasks.
      norm_momentum: `float`, normalization momentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      activation: `str`, the activation function to use typically leaky or mish.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level

    self._key_list = [
        str(key) for key in range(self._min_level, self._max_level + 1)
    ]

    self._classes = classes
    self._boxes_per_level = boxes_per_level
    self._output_extras = output_extras

    self._output_conv = (classes + output_extras + 5) * boxes_per_level

    self._base_config = dict(
        activation=activation,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

    self._conv_config = dict(
        filters=self._output_conv,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        use_bn=False,
        **self._base_config)

  def build(self, input_shape):
    self._head = dict()
    for key in self._key_list:
      self._head[key] = nn_blocks.ConvBN(**self._conv_config)

  def call(self, inputs):
    outputs = dict()
    for key in self._key_list:
      outputs[key] = self._head[key](inputs[key])
    return outputs

  @property
  def output_depth(self):
    return (self._classes + self._output_extras + 5) * self._boxes_per_level

  @property
  def num_boxes(self):
    if self._min_level is None or self._max_level is None:
      raise Exception(
          'Model has to be built before number of boxes can be determined.')
    return (self._max_level - self._min_level + 1) * self._boxes_per_level

  def get_config(self):
    config = dict(
        min_level=self._min_level,
        max_level=self._max_level,
        classes=self._classes,
        boxes_per_level=self._boxes_per_level,
        output_extras=self._output_extras,
        **self._base_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
