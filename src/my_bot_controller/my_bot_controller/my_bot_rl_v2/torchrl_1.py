# imports for torchrl
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
import torchvision
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    EnvBase,
    Transform,
    Compose,
    ToTensorImage,
    Resize,
    TransformedEnv,
    UnsqueezeTransform,
)


from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

def _step(self, tensordict):
    # Take a step
    batch_size = td_params.batch_size

    # publish linear and angular velocity
    linVel = tensordict["action", "linear_velocity"]
    angVel = tensordict["action", "angular_velocity"]
    self.send_velocity_command(linVel=linVel, angVel=angVel)
    
    # Create image with proper batch dimension
    image = self.segmentedImage
    image_tensor = torch.tensor(image, dtype=torch.float32)


    # Initialize tensors with proper batch dimension
    reward = torch.tensor(self.rewardVal)
    done = torch.zeros((batch_size, 1), dtype=torch.bool)

    out = TensorDict(
        {
            "image": image_tensor,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        batch_size=[batch_size],
    )
    return out


def _reset(self, td_params=None):
    if td_params is None:
        td_params = self.gen_params()
    
    batch_size = td_params.batch_size
    image = self.segmentedImage
    image_tensor = torch.tensor(image, dtype=torch.float32)
    
    return TensorDict({
        "image": image_tensor,
        "params": td_params["params"]
    }, batch_size=batch_size)

def _make_spec(self, td_params):
    batch_size = td_params.batch_size

    self.observation_spec = CompositeSpec(
        stepInt=BoundedTensorSpec(
            low=td_params["params", "step_start"],
            high=td_params["params", "step_end"],
            shape=batch_size,
            dtype=torch.int32,
        ),
        image=UnboundedContinuousTensorSpec(
            shape=(batch_size[0], td_params["params", "imageHeight"][0].item(), 
                td_params["params", "imageWidth"][0].item(), 3),
            dtype=torch.float32,
        ),
        params=make_composite_from_td(td_params["params"]),
        shape=batch_size,
    )

    self.action_spec = CompositeSpec(
        action=CompositeSpec(
            linear_velocity=BoundedTensorSpec(
                low=-td_params["params", "max_linear_velocity"],
                high=td_params["params", "max_linear_velocity"],
                shape=batch_size,
                dtype=torch.float32,
            ),
            angular_velocity=BoundedTensorSpec(
                low=-td_params["params", "max_angular_velocity"],
                high=td_params["params", "max_angular_velocity"],
                shape=batch_size,
                dtype=torch.float32,
            ),
            shape=batch_size,
        ),
        shape=batch_size,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(batch_size[0], 1))

def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
    shape=td.shape,)
    return composite

def gen_params(batch_size=None, *args, **kwargs) -> TensorDictBase:
    """Returns a tensordict containing the parameters with specified batch size."""
    if batch_size is None:
        batch_size = [1]
        
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "step_start": torch.zeros(batch_size, dtype=torch.int32),
                    "step_end": torch.full(batch_size, 20, dtype=torch.int32),
                    "max_linear_velocity": torch.ones(batch_size, dtype=torch.float32),
                    "max_angular_velocity": torch.ones(batch_size, dtype=torch.float32),
                    "imageHeight": torch.full(batch_size, 480, dtype=torch.int32),
                    "imageWidth": torch.full(batch_size, 640, dtype=torch.int32),
                },
                batch_size=batch_size,
            )
        },
        batch_size=batch_size,
    )
    return td

class my_botEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu", batch_size=None):
        if batch_size is None:
            batch_size = [1]
            
        if td_params is None:
            td_params = self.gen_params(batch_size=batch_size)

        super().__init__(device=device, batch_size=batch_size)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _step = staticmethod(_step)

