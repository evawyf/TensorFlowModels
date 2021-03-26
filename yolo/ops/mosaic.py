import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops as bbox_ops
from yolo.ops import preprocessing_ops
from official.vision.beta.ops import box_ops

import tensorflow_datasets as tfds 
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.utils.demos import utils, coco
import matplotlib.pyplot as plt


def _generate_source_id(image_bytes):
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(image_bytes, 2**63 - 1))

class Mosaic(object):
  def __init__(self, output_size, is_training = False, mosaic_frequency = 0.75, random_crop = False):
    self._output_size = output_size
    self._mosaic_frequency = mosaic_frequency
    self._random_crop = random_crop
    self._is_training = is_training
    self._seed = None if is_training else 10
    return 
  
  def _estimate_shape(self, image):
    height, width = preprocessing_ops.get_image_shape(image)
    oheight, owidth = self._output_size[0], self._output_size[1]

    if height > width:
      scale = oheight/height
      nwidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
      if nwidth > owidth:
        scale = owidth/width
        oheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
      else:
        owidth = nwidth
    else:
      scale = owidth/width
      nheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
      if nheight > oheight:
        scale = oheight/height
        owidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
      else:
        oheight = nheight
    return oheight, owidth

  def _letter_box(self, image, boxes = None,  xs = 0.0, ys = 0.0):
    height, width = self._estimate_shape(image)
    hclipper, wclipper = self._output_size[0], self._output_size[1]

    xs = tf.convert_to_tensor(xs)
    ys = tf.convert_to_tensor(ys)
    pad_width_p = wclipper - width
    pad_height_p = hclipper - height
    pad_height = tf.cast(tf.cast(pad_height_p, ys.dtype) * ys, tf.int32)
    pad_width = tf.cast(tf.cast(pad_width_p, xs.dtype) * xs, tf.int32)

    image = tf.image.resize(image, (height, width))
    image = tf.image.pad_to_bounding_box(image, pad_height, pad_width, hclipper, wclipper)
    info = tf.convert_to_tensor([pad_height, pad_width, height, width])

    if boxes is not None:
      boxes = bbox_ops.yxyx_to_xcycwh(boxes)
      x, y, w, h = tf.split(boxes, 4, axis=-1)

      y *= tf.cast(height / hclipper, y.dtype)
      x *= tf.cast(width / wclipper, x.dtype)

      y += tf.cast((pad_height / hclipper), y.dtype)
      x += tf.cast((pad_width / wclipper), x.dtype)

      h *= tf.cast(height / hclipper, h.dtype)
      w *= tf.cast(width / wclipper, w.dtype)

      boxes = tf.concat([x, y, w, h], axis=-1)

      boxes = bbox_ops.xcycwh_to_yxyx(boxes)
      #boxes = tf.where(h == 0, tf.zeros_like(boxes), boxes)
    return image, boxes, info

  def _pad_images(self, sample):
    image = sample['image']
    image, boxes, info = self._letter_box(image, xs = 0.0, ys = 0.0)

    sample['image'] = image #image #tf.image.pad_to_bounding_box(image, 0, 0, self._output_size[0], self._output_size[1])
    sample['info'] = info #tf.convert_to_tensor([0, 0, height, width])
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    return sample

  def _unpad_images(self, image, info, squeeze = True):
    if squeeze:
      image = tf.squeeze(image, axis = 0)
      info = tf.squeeze(info, axis = 0)
    image = tf.image.crop_to_bounding_box(image, info[0], info[1], info[2], info[3])
    return image
  
  def _unpad_gt_comps(self, boxes, classes, is_crowd, area, squeeze = True):
    if squeeze:
      boxes = tf.squeeze(boxes, axis = 0)
      classes = tf.squeeze(classes, axis = 0)
      is_crowd = tf.squeeze(is_crowd, axis = 0)
      area = tf.squeeze(area, axis = 0)

    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(classes, indices)
    area = tf.gather(area, indices)
    return boxes, classes, is_crowd, area
  
  def _mapped(self, sample):
    if self._mosaic_frequency > 0.0:
      domo = tf.random.uniform([], 0.0, 1.0, dtype = tf.float32, seed = self._seed)
      if domo >= (1 - self._mosaic_frequency):
        image = sample['image']
        boxes = sample['groundtruth_boxes']
        classes = sample['groundtruth_classes']
        is_crowd = sample['groundtruth_is_crowd']
        area = sample['groundtruth_area']
        info = sample['info']

        images = tf.split(image, 4, axis = 0)
        box_list = tf.split(boxes, 4, axis = 0)
        class_list = tf.split(classes, 4, axis = 0)
        is_crowds = tf.split(is_crowd, 4, axis = 0)
        areas = tf.split(area, 4, axis = 0)
        infos = tf.split(info, 4, axis = 0)

        images[0] = self._unpad_images(images[0], infos[0])
        images[1] = self._unpad_images(images[1], infos[1])
        images[2] = self._unpad_images(images[2], infos[2])
        images[3] = self._unpad_images(images[3], infos[3])

        (box_list[0], 
        class_list[0], 
        is_crowds[0], 
        areas[0]) = self._unpad_gt_comps(box_list[0], class_list[0], is_crowds[0], areas[0])

        (box_list[1], 
        class_list[1], 
        is_crowds[1], 
        areas[1]) = self._unpad_gt_comps(box_list[1], class_list[1], is_crowds[1], areas[1])
        
        (box_list[2], 
        class_list[2], 
        is_crowds[2], 
        areas[2]) = self._unpad_gt_comps(box_list[2], class_list[2], is_crowds[2], areas[2])
        
        (box_list[3], 
        class_list[3], 
        is_crowds[3], 
        areas[3]) = self._unpad_gt_comps(box_list[3], class_list[3], is_crowds[3], areas[3])

        images[0], box_list[0], infos[0] = self._letter_box(images[0], box_list[0], xs = 1.0, ys = 1.0)
        images[1], box_list[1], infos[1] = self._letter_box(images[1], box_list[1], xs = 0.0, ys = 1.0)
        images[2], box_list[2], infos[2] = self._letter_box(images[2], box_list[2], xs = 1.0, ys = 0.0)
        images[3], box_list[3], infos[3] = self._letter_box(images[3], box_list[3], xs = 0.0, ys = 0.0)

        box_list[0] = box_list[0] * 0.5
        box_list[1], class_list[1] = preprocessing_ops.translate_boxes(box_list[1] * 0.5, class_list[1], .5, 0)
        box_list[2], class_list[2] = preprocessing_ops.translate_boxes(box_list[2] * 0.5, class_list[2], 0, .5)
        box_list[3], class_list[3] = preprocessing_ops.translate_boxes(box_list[3] * 0.5, class_list[3], .5, .5)


        patch1 = tf.concat([images[0], images[1]], axis=-2)
        patch2 = tf.concat([images[2], images[3]], axis=-2)
        image = tf.concat([patch1, patch2], axis=-3)

        height, width = self._output_size[0], self._output_size[1]
        image = tf.image.resize(image, (height, width))

        boxes = tf.concat(box_list, axis = 0)
        classes = tf.concat(class_list, axis = 0)
        is_crowd = tf.concat(is_crowds, axis = 0)
        area = tf.concat(areas, axis = 0)

        sample['image'] = tf.expand_dims(image, axis = 0)
        sample['source_id'] = tf.expand_dims(sample['source_id'][0], axis = 0)
        sample['width'] = tf.expand_dims(width, axis = 0)
        sample['height'] = tf.expand_dims(height, axis = 0)
        sample['groundtruth_boxes'] = tf.expand_dims(boxes, axis = 0)
        sample['groundtruth_classes'] = tf.expand_dims(classes, axis = 0)
        sample['groundtruth_is_crowd'] = tf.cast(tf.expand_dims(is_crowd, axis = 0), tf.bool)
        sample['groundtruth_area'] = tf.expand_dims(area, axis = 0)
        sample['info'] = tf.expand_dims(tf.convert_to_tensor([0, 0, height, width]), axis = 0)
        sample['num_detections'] = tf.expand_dims(tf.shape(sample['groundtruth_boxes'])[0], axis = 0)
        sample['is_mosaic'] = tf.expand_dims(tf.cast(1.0, tf.bool), axis = 0)
    return sample

  def resample_unpad(self, sample):
    sample['image'] = self._unpad_images(sample['image'], sample['info'], squeeze = False)

    (sample['groundtruth_boxes'], 
     sample['groundtruth_classes'], 
     sample['groundtruth_is_crowd'], 
     sample['groundtruth_area']) = self._unpad_gt_comps(sample['groundtruth_boxes'], sample['groundtruth_classes'], sample['groundtruth_is_crowd'], sample['groundtruth_area'], squeeze = False)
    return sample

  def __call__(self, dataset):
    dataset = dataset.map(decoder.decode)
    dataset = dataset.map(mosaic._pad_images)

    dataset = dataset.padded_batch(4)
    dataset = dataset.map(mosaic._mapped)
    dataset = dataset.unbatch()
    dataset = dataset.map(mosaic.resample_unpad)
    if self._is_training:
      dataset = dataset.shuffle(4 * 8)
    return dataset

drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=2)
decoder = tfds_coco_decoder.MSCOCODecoder()
mosaic = Mosaic([640, 640], is_training=True)

dataset = tfds.load('coco', split = 'train')
dataset = dataset.apply(mosaic)
dataset = dataset.take(100)

import time 

a = time.time()
for image in dataset:
  im = image['image']/255
  boxes = image['groundtruth_boxes']
  classes = image['groundtruth_classes']
  confidence = image['groundtruth_classes']

  draw_dict = {
    'bbox' : boxes, 
    'classes' : classes, 
    'confidence': confidence, 
  }

  im = drawer(im, draw_dict)  
  print(image["num_detections"])
  plt.imshow(im)
  plt.show()
b = time.time()

print(b - a)