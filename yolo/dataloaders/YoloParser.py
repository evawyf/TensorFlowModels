import tensorflow as tf
import tensorflow.keras.backend as K

from yolo.dataloaders.Parser import Parser
from yolo.dataloaders.decoder import Decoder

from yolo.dataloaders.ops.preprocessing_ops import _scale_image
from yolo.dataloaders.ops.preprocessing_ops import _get_best_anchor
from yolo.dataloaders.ops.preprocessing_ops import _jitter_boxes
from yolo.dataloaders.ops.preprocessing_ops import _translate_image
# from yolo.dataloaders.ops.preprocessing_ops import _translate_boxes

from yolo.dataloaders.ops.random_ops import _box_scale_rand
from yolo.dataloaders.ops.random_ops import _jitter_rand
from yolo.dataloaders.ops.random_ops import _translate_rand

from yolo.utils.box_utils import _xcycwh_to_xyxy
from yolo.utils.box_utils import _xcycwh_to_yxyx
from yolo.utils.box_utils import _yxyx_to_xcycwh
from yolo.utils.loss_utils import build_grided_gt

class YoloDecoder(Decoder):
    def __init__(self,
                 image_w=416,
                 image_h=416,
                 num_classes=80,
                 fixed_size=False,
                 jitter_im=0.1,
                 jitter_boxes=0.005,
                 net_down_scale=32,
                 path_scales=None,
                 max_process_size=608,
                 min_process_size=320,
                 pct_rand=0.5,
                 masks=None,
                 anchors=None):

        self._image_w = image_w
        self._image_h = image_h
        self._num_classes = num_classes
        self._fixed_size = fixed_size
        self._jitter_im = 0.0 if jitter_im == None else jitter_im
        self._jitter_boxes = 0.0 if jitter_boxes == None else jitter_boxes
        self._net_down_scale = net_down_scale
        self._max_process_size = max_process_size
        self._min_process_size = min_process_size
        self._pct_rand = pct_rand
        self._masks = {
            "1024": [6, 7, 8],
            "512": [3, 4, 5],
            "256": [0, 1, 2]
        } if masks == None else masks
        self._anchors = anchors
        return
    
    def decode(self, data):
        shape = tf.shape(data["image"])
        image = _scale_image(data["image"], resize=True, w = self._max_process_size, h = self._max_process_size)
        boxes = _yxyx_to_xcycwh(data["objects"]["bbox"])
        best_anchors = _get_best_anchor(boxes, self._anchors, self._image_w, self._image_h)
        return {
            "source_id": data["image/id"],
            "image": image,
            "bbox": boxes,
            "classes": data["objects"]["label"],
            "area": data["objects"]["area"],
            "is_crowd": tf.cast(data["objects"]["is_crowd"], tf.int32),
            "best_anchors": best_anchors, 
            "width": shape[1],
            "height": shape[2],
            "num_detections": tf.shape(data["objects"]["label"])[0]
        }
    
class YoloParser(Parser):
    def __init__(self,
                 image_w=416,
                 image_h=416,
                 num_classes=80,
                 fixed_size=False,
                 jitter_im=0.1,
                 jitter_boxes=0.005,
                 net_down_scale=32,
                 max_process_size=608,
                 min_process_size=320,
                 pct_rand=0.5,
                 masks=None,
                 anchors=None):

        self._image_w = image_w
        self._image_h = image_h
        self._num_classes = num_classes
        self._fixed_size = fixed_size
        self._jitter_im = 0.0 if jitter_im == None else jitter_im
        self._jitter_boxes = 0.0 if jitter_boxes == None else jitter_boxes
        self._net_down_scale = net_down_scale
        self._max_process_size = max_process_size
        self._min_process_size = min_process_size
        self._pct_rand = pct_rand
        self._masks = {
            "1024": [6, 7, 8],
            "512": [3, 4, 5],
            "256": [0, 1, 2]
        } if masks == None else masks
        self._anchors = anchors  # use K means to find boxes if it is None
        return
    
    def _parse_train_data(self, data, is_training=True):
        randscale = self._image_w // self._net_down_scale
        if not self._fixed_size:
            randscale = tf.py_function(_box_scale_rand, [
                self._min_process_size // self._net_down_scale,
                self._max_process_size // self._net_down_scale, randscale,
                self._pct_rand
            ], tf.int32)

        if self._jitter_im != 0.0:
            translate_x, translate_y = tf.py_function(_translate_rand,
                                                      [self._jitter_im],
                                                      [tf.float32, tf.float32])
        else:
            translate_x, translate_y = 0.0, 0.0

        if self._jitter_boxes != 0.0:
            j_x, j_y, j_w, j_h = tf.py_function(
                _jitter_rand, [self._jitter_boxes],
                [tf.float32, tf.float32, tf.float32, tf.float32])
        else:
            j_x, j_y, j_w, j_h = 0.0, 0.0, 1.0, 1.0

        image = tf.image.resize(data["image"],
                                size=(randscale * 32,
                                      randscale * 32))  # Random Resize
        image = tf.image.random_brightness(image=image,
                                           max_delta=.1)  # Brightness
        image = tf.image.random_saturation(image=image, lower=0.75,
                                           upper=1.25)  # Saturation
        image = tf.image.random_hue(image=image, max_delta=.1)  # Hue
        image = tf.clip_by_value(image, 0.0, 1.0)

        image = tf.image.resize(image, 
                                size=(randscale * self._net_down_scale,
                                      randscale * self._net_down_scale)) # Random Resize
        image = _translate_image(image, translate_x, translate_y)
        boxes = _jitter_boxes(data["bbox"], translate_x, translate_y, j_x, j_y, j_w, j_h)
        #label = tf.concat([boxes, tf.expand_dims(tf.cast(data["classes"], boxes.dtype), axis = -1), data["best_anchors"]], axis=-1)
        return image, {"source_id": data["source_id"],
                        "bbox": boxes,
                        "classes": data["classes"],
                        "area": data["area"],
                        "is_crowd": data["is_crowd"],
                        "best_anchors": data["best_anchors"], 
                        "width": data["width"],
                        "height": data["height"],
                        "num_detections": data["num_detections"]}

    def _parse_eval_data(self, data):
        randscale = self._image_w // self._net_down_scale
        image = tf.image.resize(data["image"], 
                                size=(randscale * self._net_down_scale,
                                      randscale * self._net_down_scale)) # Random Resize
        image = tf.clip_by_value(image, 0.0, 1.0)
        #label = tf.concat([boxes, tf.expand_dims(tf.cast(data["classes"], boxes.dtype), axis = -1), data["best_anchors"]], axis=-1)
        return image, {"source_id": data["source_id"],
                        "bbox": data["bbox"],
                        "classes": data["classes"],
                        "area": data["area"],
                        "is_crowd": data["is_crowd"],
                        "best_anchors": data["best_anchors"], 
                        "width": data["width"],
                        "height": data["height"],
                        "num_detections": data["num_detections"]}
    
    def parse_fn(self, is_training):
        """Returns a parse fn that reads and parses raw tensors from the decoder.
        Args:
            is_training: a `bool` to indicate whether it is in training mode.
        Returns:
            parse: a `callable` that takes the serialized examle and generate the
                images, labels tuple where labels is a dict of Tensors that contains
                labels.
        """
        def parse(decoded_tensors):
            """Parses the serialized example data."""
            if is_training:
                return self._parse_train_data(decoded_tensors)
            else:
                return self._parse_eval_data(decoded_tensors)
        return parse

def batch_dataset(batch_size = 10, drop_remainder = False):
    def foo(dataset, input_context):
        per_replica_batch_size = input_context.get_per_replica_batch_size(batch_size) if input_context else batch_size
        dataset = dataset.padded_batch(per_replica_batch_size, drop_remainder=drop_remainder)
        return dataset
    return foo


# class YoloParser2(DatasetParser):
#     def __init__(self,
#                  image_w=416,
#                  image_h=416,
#                  num_classes=80,
#                  fixed_size=False,
#                  jitter_im=0.1,
#                  jitter_boxes=0.005,
#                  net_down_scale=32,
#                  max_process_size=608,
#                  min_process_size=320,
#                  pct_rand=0.5,
#                  masks=None,
#                  anchors=None):

#         self._image_w = image_w
#         self._image_h = image_h
#         self._num_classes = num_classes
#         self._fixed_size = fixed_size
#         self._jitter_im = 0.0 if jitter_im == None else jitter_im
#         self._jitter_boxes = 0.0 if jitter_boxes == None else jitter_boxes
#         self._net_down_scale = net_down_scale
#         self._max_process_size = max_process_size
#         self._min_process_size = min_process_size
#         self._pct_rand = pct_rand
#         self._masks = {
#             "1024": [6, 7, 8],
#             "512": [3, 4, 5],
#             "256": [0, 1, 2]
#         } if masks == None else masks
#         self._anchors = anchors  # use K means to find boxes if it is None
#         return

#     def _unbatched_processing(self, data):
#         image = _scale_image(data["image"],
#                              square=True,
#                              square_w=self._max_process_size)
#         boxes = _yxyx_to_xcycwh(data["objects"]["bbox"])
#         best_anchor = _get_best_anchor(boxes, self._anchors, self._image_w)
#         return {
#             "image": image,
#             "bbox": boxes,
#             "classes": data["objects"]["label"],
#             "best_anchors": best_anchor
#         }

#     def _batched_processing(self, data, is_training=True):
#         randscale = self._image_w // self._net_down_scale
#         if not self._fixed_size and is_training:
#             randscale = tf.py_function(_box_scale_rand, [
#                 self._min_process_size // self._net_down_scale,
#                 self._max_process_size // self._net_down_scale, randscale,
#                 self._pct_rand
#             ], tf.int32)

#         if self._jitter_im != 0.0 and is_training:
#             translate_x, translate_y = tf.py_function(_translate_rand,
#                                                       [self._jitter_im],
#                                                       [tf.float32, tf.float32])
#         else:
#             translate_x, translate_y = 0.0, 0.0

#         if self._jitter_boxes != 0.0 and is_training:
#             j_x, j_y, j_w, j_h = tf.py_function(
#                 _jitter_rand, [self._jitter_boxes],
#                 [tf.float32, tf.float32, tf.float32, tf.float32])
#         else:
#             j_x, j_y, j_w, j_h = 0.0, 0.0, 1.0, 1.0

#         image = tf.image.resize(data["image"],
#                                 size=(randscale * 32,
#                                       randscale * 32))  # Random Resize
#         image = tf.image.random_brightness(image=image,
#                                            max_delta=.1)  # Brightness
#         image = tf.image.random_saturation(image=image, lower=0.75,
#                                            upper=1.25)  # Saturation
#         image = tf.image.random_hue(image=image, max_delta=.1)  # Hue
#         image = tf.clip_by_value(image, 0.0, 1.0)

#         image = tf.image.resize(image, 
#                                 size=(randscale * self._net_down_scale,
#                                       randscale * self._net_down_scale)) # Random Resize
#         image = _translate_image(image, translate_x, translate_y)
#         boxes = _jitter_boxes(data["bbox"], translate_x, translate_y, j_x, j_y, j_w, j_h)
#         label = tf.concat([boxes, tf.expand_dims(tf.cast(data["classes"], boxes.dtype), axis = -1), data["best_anchors"]], axis=-1)
#         return image, {"label": label, "bbox": boxes, "classes":data["classes"], "best_anchors":data["best_anchors"]}


#     def unbatched_process_fn(self, is_training):
#         def parse(tensor_set):
#             return self._unbatched_processing(tensor_set)

#         return parse

#     def batched_process_fn(self, is_training):
#         def parse(tensor_set):
#             return self._batched_processing(tensor_set,
#                                             is_training=is_training)

#         return parse

#     def build_gt(self, is_training):
#         def parse(tensor_set):
#             return self._label_format_gt(tensor_set)

#         return parse


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    import matplotlib.pyplot as plt
    coco, info = tfds.load('coco', split='train', with_info=True)

    parser = YoloParser(image_h=320,
                        image_w=320,
                        fixed_size=True,
                        anchors=[(10, 13), (16, 30), (33, 23), (30, 61),
                                 (62, 45), (59, 119), (116, 90), (156, 198),
                                 (373, 326)])
    process_1 = parser.unbatched_process_fn(is_training=True)
    process_2 = parser.batched_process_fn(is_training=True)
    process_3 = parser.build_gt(is_training=True)
    coco = coco.map(process_1).padded_batch(1)
    coco = coco.map(process_2)
    #    coco = coco.map(process_3)

    for k in coco.take(10):

        boxes = _xcycwh_to_yxyx(k["label"][..., :4])
        image = tf.image.draw_bounding_boxes(k["image"], boxes,
                                             [[1.0, 0.0, 0.0]])
        tf.print(k["randscale"], boxes)
        plt.imshow(image.numpy()[0])
        plt.show()
