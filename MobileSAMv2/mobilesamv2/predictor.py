# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

#from mobilesamv2.modeling import Sam
from .modeling.sam import Sam
from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamPredictor:
	def __init__(
		self,
		sam_model: Sam,
	) -> None:
		"""
		Uses SAM to calculate the image embedding for an image, and then
		allow repeated, efficient mask prediction given prompts.
		Arguments:
		  sam_model (Sam): The model to use for mask prediction.
		"""
		self.model = sam_model
		#self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
		#self.feature_name=0
	def set_image(
		self,
		image: torch.Tensor,
		image_format: str = "RGB",
	) -> None:
		"""
		Calculates the image embeddings for the provided image, allowing
		masks to be predicted with the 'predict' method.
		Arguments:
		  image (np.ndarray): The image for calculating masks. Expects an
			image in HWC uint8 format, with pixel values in [0, 255].
		  image_format (str): The color format of the image, in ['RGB', 'BGR'].
		"""
		assert image_format in [
			"RGB",
			"BGR",
		], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
		if image_format != self.model.image_format:
			image = image[..., ::-1]

		# Transform the image to the form expected by the model
		# SEE: write transforms for tensor type image
		#input_image = self.transform.apply_image(image)
		#input_image_torch = torch.as_tensor(input_image, device=self.device)
		input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

		self.set_torch_image(input_image_torch, image.shape[:2])

	@torch.no_grad()
	def set_torch_image(
		self,
		transformed_image: torch.Tensor,
		original_image_size: Tuple[int, ...],
	) -> None:
		"""
		Calculates the image embeddings for the provided image, allowing
		masks to be predicted with the 'predict' method. Expects the input
		image to be already transformed to the format expected by the model.
		Arguments:
		  transformed_image (torch.Tensor): The input image, with shape
			1x3xHxW, which has been transformed with ResizeLongestSide.
		  original_image_size (tuple(int, int)): The size of the image
			before transformation, in (H, W) format.
		"""
		assert (
			len(transformed_image.shape) == 4
			and transformed_image.shape[1] == 3
			and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
		), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

		self.original_size = original_image_size
		self.input_size = tuple(transformed_image.shape[-2:])
		input_image = self.model.preprocess(transformed_image)
		#import pdb;pdb.set_trace()
		# import time
		# aa = time.time()
		self.features = self.model.image_encoder(input_image)
		# cc = time.time()
		# print(cc-aa, ',')
		# import pdb;pdb.set_trace()
		
	def predict(
		self,
		point_coords: torch.Tensor = None,
		point_labels: torch.Tensor = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Predict masks for the given input prompts, using the currently set image.
		Arguments:
		  point_coords (np.ndarray or None): A Nx2 array of point prompts to the
			model. Each point is in (X,Y) in pixels.
		  point_labels (np.ndarray or None): A length N array of labels for the
			point prompts. 1 indicates a foreground point and 0 indicates a
			background point.
		  box (np.ndarray or None): A length 4 array given a box prompt to the
			model, in XYXY format.
		  mask_input (np.ndarray): A low resolution mask input to the model, typically
			coming from a previous prediction iteration. Has form 1xHxW, where
			for SAM, H=W=256.
		  multimask_output (bool): If true, the model will return three masks.
			For ambiguous input prompts (such as a single click), this will often
			produce better masks than a single prediction. If only a single
			mask is needed, the model's predicted quality score can be used
			to select the best mask. For non-ambiguous prompts, such as multiple
			input prompts, multimask_output=False can give better results.
		  return_logits (bool): If true, returns un-thresholded masks logits
			instead of a binary mask.
		Returns:
		  (np.ndarray): The output masks in CxHxW format, where C is the
			number of masks, and (H, W) is the original image size.
		  (np.ndarray): An array of length C containing the model's
			predictions for the quality of each mask.
		  (np.ndarray): An array of shape CxHxW, where C is the number
			of masks and H=W=256. These low resolution logits can be passed to
			a subsequent iteration as mask input.
		"""
		return_logits = True
		multimask_output = False

		# Transform input prompts
		coords_torch, labels_torch, box_torch  = None, None, None
		if point_coords is not None:
			assert (
				point_labels is not None
			), "point_labels must be supplied if point_coords is supplied."
			#SEE apply coords transform as tensor
			#point_coords = self.transform.apply_coords(point_coords, self.original_size)
			#coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device="cpu")
			#labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device="cpu")
			#coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

		# import time
		# aa = time.time()
		masks, iou_predictions, low_res_masks = self.predict_torch(
			coords_torch,
			labels_torch,
		)
		# cc = time.time()
		# print('decoder_time:', cc-aa)
		# import pdb; pdb.set_trace()
		
		
		masks_np = masks[0]
		iou_predictions_np = iou_predictions[0]
		low_res_masks_np = low_res_masks[0]
		return masks_np, iou_predictions_np, low_res_masks_np

	@torch.no_grad()
	def predict_torch(
		self,
		point_coords: Optional[torch.Tensor],
		point_labels: Optional[torch.Tensor],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
		multimask_output = False
		return_logits = True
		if point_coords is not None:
			points = (point_coords, point_labels)
		else:
			points = None
	   
		# Embed prompts
		# import pdb;pdb.set_trace()
		sparse_embeddings, dense_embeddings = self.model.prompt_encoder.forward(
			points=points,
			boxes=None,
			masks=None,
		)
		#import pdb;pdb.set_trace()
		# Predict masks
		low_res_masks, iou_predictions = self.model.mask_decoder(
			image_embeddings=self.features,
			image_pe=self.model.prompt_encoder.get_dense_pe(),
			sparse_prompt_embeddings=sparse_embeddings,
			dense_prompt_embeddings=dense_embeddings,
			multimask_output=multimask_output,
		)

		# Upscale the masks to the original image resolution
		# if self.han_size is not None:
		#	  self.original_size=self.han_size
		masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
		# if self.han_size is not None:
		#	  self.original_size=(1024,1024)
		#import pdb;pdb.set_trace()
		if not return_logits:
			masks = masks > self.model.mask_threshold

		return masks, iou_predictions, low_res_masks


###################################################################################   han_change		
	def set_image_AddAverage(
		self,
		image: np.ndarray,
		image_format: str = "RGB",
	) -> None:
		"""
		Calculates the image embeddings for the provided image, allowing
		masks to be predicted with the 'predict' method.

		Arguments:
		  image (np.ndarray): The image for calculating masks. Expects an
			image in HWC uint8 format, with pixel values in [0, 255].
		  image_format (str): The color format of the image, in ['RGB', 'BGR'].
		"""
		assert image_format in [
			"RGB",
			"BGR",
		], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
		if image_format != self.model.image_format:
			image = image[..., ::-1]

		# Transform the image to the form expected by the model
		#input_image = self.transform.apply_image(image)
		input_image_torch = torch.as_tensor(image, device="cpu")
		input_image_torch = input_image_torch.permute(0, 3, 1, 2).contiguous()[:, :, :, :]

		self.set_torch_image_AddAverage(input_image_torch, image.shape[:2])

	@torch.no_grad()
	def set_torch_image_AddAverage(
		self,
		transformed_image: torch.Tensor,
		original_image_size: Tuple[int, int, int, int],
	) -> None:
		"""
		Calculates the image embeddings for the provided image, allowing
		masks to be predicted with the 'predict' method. Expects the input
		image to be already transformed to the format expected by the model.

		Arguments:
		  transformed_image (torch.Tensor): The input image, with shape
			1x3xHxW, which has been transformed with ResizeLongestSide.
		  original_image_size (tuple(int, int)): The size of the image
			before transformation, in (H, W) format.
		"""
		assert (
			len(transformed_image.shape) == 4
			and transformed_image.shape[1] == 3
			and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
		), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."

		self.original_size = original_image_size
		self.input_size = tuple(transformed_image.shape[-2:])
		input_image = self.model.preprocess(transformed_image)
		encoder_change = self.model.image_encoder(input_image)
		self.features=encoder_change#+ torch.from_numpy(np.load('./model_output/mean_100.npy')).to(device=self.device)
				#handongshen
		#ppp=np.load('./data/val1/features/'+self.feature_name)
		#self.features=torch.from_numpy(ppp).to(device=self.device)
		#features = np.load(join('./data','features','sa_227195.npy'))
		#array = self.features.cpu().numpy()
		#np.save(self.featurename[0], array) 
		#eturn 0

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
