# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataclasses for optimizer configs."""
from typing import List, Optional

import dataclasses
from official.modeling.hyperparams import base_config
from official.modeling.optimization.configs.optimizer_config import BaseOptimizerConfig


@dataclasses.dataclass
class BaseOptimizerConfig(base_config.Config):
  """Base optimizer config.

  Attributes:
    clipnorm: float >= 0 or None. If not None, Gradients will be clipped when
      their L2 norm exceeds this value.
    clipvalue: float >= 0 or None. If not None, Gradients will be clipped when
      their absolute value exceeds this value.
    global_clipnorm: float >= 0 or None. If not None, gradient of all weights is
        clipped so that their global norm is no higher than this value
  """
  clipnorm: Optional[float] = None
  clipvalue: Optional[float] = None
  global_clipnorm: Optional[float] = None


@dataclasses.dataclass
class SGDAccumConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

  Attributes:
    name: name of the optimizer.
    decay: decay rate for SGD optimizer.
    nesterov: nesterov for SGD optimizer.
    momentum: momentum for SGD optimizer.
  """
  name: str = "SGD"
  decay: float = 0.0
  nesterov: bool = False
  momentum: float = 0.0
  momentum_start: float = 0.0
  accumulation_type: str = "mean"
  one_offset: bool = False 
  accumulation_steps: int = 1
  adjusted_for_accum: bool = True
  warmup_steps: int = 1000


@dataclasses.dataclass
class SGDMomentumWarmupConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

  Attributes:
    name: name of the optimizer.
    decay: decay rate for SGD optimizer.
    nesterov: nesterov for SGD optimizer.
    momentum: momentum for SGD optimizer.
  """
  name: str = "SGD"
  decay: float = 0.0
  nesterov: bool = False
  momentum_start: float = 0.0
  momentum: float = 0.9
  warmup_steps: int = 1000

@dataclasses.dataclass
class SGDMomentumWarmupWConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

  Attributes:
    name: name of the optimizer.
    decay: decay rate for SGD optimizer.
    nesterov: nesterov for SGD optimizer.
    momentum: momentum for SGD optimizer.
  """
  name: str = "SGD"
  weight_decay: float = 0.0000
  decay: float = 0.0
  nesterov: bool = False
  momentum_start: float = 0.0
  momentum: float = 0.9
  warmup_steps: int = 1000

@dataclasses.dataclass
class AdamWConfig(BaseOptimizerConfig):
  """Configuration for Adam optimizer.

  The attributes for this class matches the arguments of
  tf.keras.optimizer.Adam.

  Attributes:
    name: name of the optimizer.
    beta_1: decay rate for 1st order moments.
    beta_2: decay rate for 2st order moments.
    epsilon: epsilon value used for numerical stability in Adam optimizer.
    amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond".
  """
  name: str = "Adam"
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  amsgrad: bool = False
  weight_decay: float = 0.0000

@dataclasses.dataclass
class ScaledYoloSGDConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.

  Attributes:
    name: name of the optimizer.
    decay: decay rate for SGD optimizer.
    nesterov: nesterov for SGD optimizer.
    momentum: momentum for SGD optimizer.
  """
  name: str = "SGD"
  decay: float = 0.0
  nesterov: bool = False
  momentum_start: float = 0.0
  momentum: float = 0.8
  warmup_steps: int = 1000
  