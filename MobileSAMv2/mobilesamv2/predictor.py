# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from mobilesamv2.modeling import Sam
from typing import Tuple
from .utils.transforms import ResizeLongestSide


class SamPredictor:
	def __init__(
		self,
		sam
	) -> None:
		"""
		Uses SAM to calculate the image embedding for an image, and then
		allow repeated, efficient mask prediction given prompts.
		Arguments:
		  sam_model (Sam): The model to use for mask prediction.
		"""
		#self.feature_name=0
		self.sam = sam
		return

	@torch.no_grad()
	def predict_torch(
		self,
		point_coords: torch.Tensor,
		point_labels: torch.Tensor):
		return torch.as_tensor([[[0]*1024]*720]*3, device='cpu')
