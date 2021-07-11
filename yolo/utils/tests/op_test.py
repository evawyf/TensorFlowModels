from tensorflow.python.ops.gen_image_ops import image_projective_transform_v3
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common
import dataclasses
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.training import optimizer
from yolo.ops.preprocessing_ops import apply_infos
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from yolo.configs import yolo as exp_cfg

from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder

from yolo.dataloaders import yolo_input
from yolo.ops import mosaic
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.ops.box_ops import xcycwh_to_yxyx, yxyx_to_xcycwh

from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.modeling.layers import detection_generator
from collections import defaultdict

from typing import Optional
from official.core import config_definitions
from official.modeling import optimization
from official.modeling import performance

import matplotlib.pyplot as plt

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig

@dataclasses.dataclass
class Parser(hyperparams.Config):
  max_num_instances: int = 200
  letter_box: Optional[bool] = False
  random_flip: bool = True
  random_pad: float = True
  jitter: float = 0.0
  resize: float = 1.0
  jitter_mosaic: float = 0.0
  resize_mosaic: float = 1.0
  sheer: float = 0.0
  aug_rand_angle: float = 0.0
  aug_rand_translate: float = 0.0
  aug_rand_saturation: float = 0.0 #0.7
  aug_rand_brightness: float = 0.0  #0.4
  aug_rand_hue: float = 0.0 #0.1
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  mosaic_scale_min: float = 1.0
  mosaic_scale_max: float = 1.0
  use_tie_breaker: bool = True
  use_scale_xy: bool = False
  anchor_thresh: float = 0.213
  area_thresh: float = 0.1

# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  global_batch_size: int = 1
  input_path: str = "" #'/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/testing_records/records/val*'
  tfds_data_dir: str = "/media/vbanna/DATA_SHARE/CV/datasets/tensorflow"
  tfds_name: str = "coco"
  tfds_split: str = "validation"
  is_training: bool = False
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True
  cache: bool = False


def get_decoder(params):
  if params.tfds_name:
    if params.tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
      decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[
          params.tfds_name]()
    else:
      raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
  else:
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
      decoder = tf_example_decoder.TfExampleDecoder(
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    elif params.decoder.type == 'label_map_decoder':
      decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
          label_map=decoder_cfg.label_map,
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(
          params.decoder.type))
  return decoder

def build_ds(params, input_context = None):
  decoder = get_decoder(params)

  reader = input_reader.InputReader(
      params,
      dataset_fn=tf.data.TFRecordDataset,
      decoder_fn=decoder.decode)
  dataset = reader.read(input_context=input_context)
  return dataset

from yolo.ops import preprocessing_ops
def write_to_folder(path = "/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/testing_records/"):
  params = DataConfig()

  dataset = build_ds(params).unbatch()

  lim = 5
  for k, sample in enumerate(dataset):
    image_ = sample["image"]
    image, M, Mb = preprocessing_ops.affine_warp_image(
      image = image_, 
      desired_size = [512, 512],  
      seed = 0.0, 
      perspective=0.0,
      scale = 0.9,
      degrees = 0, 
      translate = 0.5, 
      shear = 0.0, 
      random_pad = True, 
    )
    boxes = sample["groundtruth_boxes"] 
    boxes = box_ops.denormalize_boxes(boxes, tf.shape(image_)[:2])
    boxes, _, _ = preprocessing_ops.affine_warp_boxes(Mb[0], boxes, Mb[1])
    boxes = box_ops.normalize_boxes(boxes, tf.shape(image)[:2])
    image = tf.expand_dims(image, axis = 0)
    boxes = tf.expand_dims(boxes, axis = 0)
    image = tf.image.draw_bounding_boxes(
      image/255, 
      boxes, 
      [[1.0, 0, 0]]
    )

    plt.imshow(image[0].numpy())
    plt.show()


  return 


if __name__ == "__main__":
  write_to_folder()
        
         