# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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


"""Scheduler Classes"""

from dataclasses import dataclass, field
from typing import Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.engine.schedulers import Scheduler, SchedulerConfig

@dataclass
class MultiStepWarmupSchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: MultiStepWarmupScheduler)
    """target class to instantiate"""
    warm_up_end: int = 5000
    """Iteration number where warmp ends"""
    milestones: list[int] = field(default_factory=lambda: [300000, 400000, 500000])
    """The milestone steps at which to decay the learning rate."""
    gamma: float = 0.33
    """The learning rate decay factor."""

class MultiStepWarmupScheduler(Scheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    config: MultiStepWarmupSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            if step < self.config.warm_up_end:
                learning_factor = step / self.config.warm_up_end
            else:
                index = np.searchsorted(self.config.milestones, step, side="left")
                learning_factor = self.config.gamma**index
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler