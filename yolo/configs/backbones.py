"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional
from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class DarkNet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = 'darknet53'
  width_scale: int = 1.0
  depth_scale: int = 1.0
  dilate: bool = False


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  darknet: DarkNet = DarkNet()
