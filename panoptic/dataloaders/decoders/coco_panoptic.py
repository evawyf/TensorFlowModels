import tensorflow_datasets as tfds 
import tensorflow as tf
import matplotlib.pyplot as plt

from panoptic.dataloaders.decoders import tfds_panoptic_coco_decoder

path = "/media/vbanna/DATA_SHARE/tfds"
dataset = "coco/2017_panoptic"
val = tfds.load(dataset, data_dir = path, split = "validation")

decoder = tfds_panoptic_coco_decoder.MSCOCODecoder(include_mask=True)
print(val)
val = val.map(decoder.decode)

lim = 10
for i, sample in enumerate(val):
  fig, axe = plt.subplots(1, 3)
  axe[0].imshow(sample["groundtruth_semantic_mask"])
  axe[1].imshow(sample["groundtruth_instance_id"] % 256)
  axe[2].imshow(sample["image"])
  plt.show()
  if i > (lim + 1):
    break

# import json

# path = "/media/vbanna/DATA_SHARE/Research/TensorFlowModelGardeners/panoptic/dataloaders/specs/coco_panoptic.json"
# file = open(path, 'r')
# file = json.load(file)


# things = []
# stuff = []

# things_names = []
# stuff_names = []

# for key in file:
#   if key["isthing"] == 1:
#     things.append(key["id"])
#     things_names.append(key["name"])
#   else:
#     stuff.append(key["id"])
#     stuff_names.append(key["name"])
  
# print(things)