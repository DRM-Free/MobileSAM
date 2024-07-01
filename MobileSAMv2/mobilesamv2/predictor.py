# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch



class SamPredictor:
	def __init__(
		self,
	) -> None:
		"""
		Uses SAM to calculate the image embedding for an image, and then
		allow repeated, efficient mask prediction given prompts.
		Arguments:
		  sam_model (Sam): The model to use for mask prediction.
		"""
		#self.model = sam_model
		#self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
		#self.reset_image()
		#self.feature_name=0
		return
	def predict_torch(
		self,
		point_coords: torch.Tensor,
		point_labels: torch.Tensor,
	) -> torch.Tensor:
		"""
		Predict masks for the given input prompts, using the currently set image.
		Input prompts are batched torch tensors and are expected to already be
		transformed to the input frame using ResizeLongestSide.
		Arguments:
		  point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
			model. Each point is in (X,Y) in pixels.
		  point_labels (torch.Tensor or None): A BxN array of labels for the
			point prompts. 1 indicates a foreground point and 0 indicates a
			background point.
		  boxes (np.ndarray or None): A Bx4 array given a box prompt to the
			model, in XYXY format.
		  mask_input (np.ndarray): A low resolution mask input to the model, typically
			coming from a previous prediction iteration. Has form Bx1xHxW, where
			for SAM, H=W=256. Masks returned by a previous iteration of the
			predict method do not need further transformation.
		  multimask_output (bool): If true, the model will return three masks.
			For ambiguous input prompts (such as a single click), this will often
			produce better masks than a single prediction. If only a single
			mask is needed, the model's predicted quality score can be used
			to select the best mask. For non-ambiguous prompts, such as multiple
			input prompts, multimask_output=False can give better results.
		  return_logits (bool): If true, returns un-thresholded masks logits
			instead of a binary mask.
		Returns:
		  (torch.Tensor): The output masks in BxCxHxW format, where C is the
			number of masks, and (H, W) is the original image size.
		  (torch.Tensor): An array of shape BxC containing the model's
			predictions for the quality of each mask.
		  (torch.Tensor): An array of shape BxCxHxW, where C is the number
			of masks and H=W=256. These low res logits can be passed to
			a subsequent iteration as mask input.
		"""
		return torch.as_tensor([[[0]*1024]*720]*3, device='cpu')
		if not self.is_image_set:
			raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
