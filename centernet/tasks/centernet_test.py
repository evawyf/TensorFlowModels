from absl.testing import parameterized
import tensorflow as tf
import numpy as np
import dataclasses

from official.modeling import hyperparams
from official.vision.beta.configs import backbones

from centernet.tasks.centernet import CenterNetTask
from centernet.configs import centernet as exp_cfg


class CenterNetTaskTest(parameterized.TestCase, tf.test.TestCase):

  def testCenterNetTask(self):
    config = exp_cfg.CenterNetTask()
    task = CenterNetTask(config)
    model = task.build_model()
    model.summary()
        


if __name__ == '__main__':
  tf.test.main()
