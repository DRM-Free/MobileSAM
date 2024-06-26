# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Optional

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
	#mask_threshold: float = 0.0
	image_format: str = "RGB"

	def __init__(
		self,
		image_encoder: ImageEncoderViT,
		prompt_encoder: PromptEncoder,
		mask_decoder: MaskDecoder, 
		pixel_mean: List[float] = [123.675, 116.28, 103.53],
		pixel_std: List[float] = [58.395, 57.12, 57.375],
	) -> None:
		"""
		SAM predicts object masks from an image and input prompts.

		Arguments:
		  image_encoder (ImageEncoderViT): The backbone used to encode the
			image into image embeddings that allow for efficient mask prediction.
		  prompt_encoder (PromptEncoder): Encodes various types of input prompts.
		  mask_decoder (MaskDecoder): Predicts masks from the image embeddings
			and encoded prompts.
		  pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
		  pixel_std (list(float)): Std values for normalizing pixels in the input image.
		"""
		super().__init__()
		self.mask_threshold: float = 0.0
		self.image_encoder = image_encoder
		self.prompt_encoder = prompt_encoder
		self.mask_decoder = mask_decoder
		self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
		self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

	@property
	def device(self) -> Any:
		return self.pixel_mean.device

	# @torch.no_grad()
	def forward(
		self,
		batched_input: List[Dict[str, Any]],
		multimask_output: bool,
	) -> List[Dict[str, torch.Tensor]]:
		"""
		Predicts masks end-to-end from provided images and prompts.
		If prompts are not known in advance, using SamPredictor is
		recommended over calling the model directly.

		Arguments:
		  batched_input (list(dict)): A list over input images, each a
			dictionary with the following keys. A prompt key can be
			excluded if it is not present.
			  'image': The image as a torch tensor in 3xHxW format,
				already transformed for input to the model.
			  'original_size': (tuple(int, int)) The original size of
				the image before transformation, as (H, W).
			  'point_coords': (torch.Tensor) Batched point prompts for
				this image, with shape BxNx2. Already transformed to the
				input frame of the model.
			  'point_labels': (torch.Tensor) Batched labels for point prompts,
				with shape BxN.
			  'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
				Already transformed to the input frame of the model.
			  'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
				in the form Bx1xHxW.
		  multimask_output (bool): Whether the model should predict multiple
			disambiguating masks, or return a single mask.

		Returns:
		  (list(dict)): A list over input images, where each element is
			as dictionary with the following keys.
			  'masks': (torch.Tensor) Batched binary mask predictions,
				with shape BxCxHxW, where B is the number of input prompts,
				C is determined by multimask_output, and (H, W) is the
				original size of the image.
			  'iou_predictions': (torch.Tensor) The model's predictions
				of mask quality, in shape BxC.
			  'low_res_logits': (torch.Tensor) Low resolution logits with
				shape BxCxHxW, where H=W=256. Can be passed as mask input
				to subsequent iterations of prediction.
		"""
		input_images = list()
		for input_ in batched_input:
			input_image = input_["image"]
			assert torch.jit.isinstance(input_image, torch.Tensor)
			input_images.append(self.preprocess(input_image))
		input_images = torch.stack(input_images, dim=0)
		# gpu_name =input_images.get_device()
		# print('handongshen123123',gpu_name)
		image_embeddings = self.image_encoder(input_images)
		outputs: List[Dict[str, torch.Tensor]] = list()
		for image_record, curr_embedding in zip(batched_input, image_embeddings):
			if "point_coords" in image_record:
				points = (image_record["point_coords"], image_record["point_labels"])
			else:
				points = None
			assert torch.jit.isinstance(points,Optional[Tuple[Tensor, Tensor]])
			boxes = image_record.get("boxes", None)
			assert torch.jit.isinstance(boxes,torch.Tensor)
			masks = image_record.get("mask_inputs", None)
			assert torch.jit.isinstance(masks,torch.Tensor)
			with torch.no_grad():
			  sparse_embeddings, dense_embeddings = self.prompt_encoder(
				  points=points,
				  boxes=boxes,
				  masks=masks,
			  )
			low_res_masks, iou_predictions = self.mask_decoder(
				image_embeddings=curr_embedding.unsqueeze(0),
				image_pe=self.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=multimask_output,
			)
			input_image = image_record["image"]
			assert torch.jit.isinstance(input_image,torch.Tensor)
			original_image = image_record["original_size"]
			assert torch.jit.isinstance(original_image,torch.Tensor)
			masks = self.postprocess_masks(
				low_res_masks,
				input_size=(input_image.shape[-2],input_image.shape[-1]),
				original_size=(original_image[0],original_image[1]),
			)
			masks = masks > self.mask_threshold
			outputs.append(
				{
					"masks": masks,
					"iou_predictions": iou_predictions,
					"low_res_logits": low_res_masks,
				}
			)

		return outputs

	def postprocess_masks(
		self,
		masks: torch.Tensor,
		input_size: Tuple[int, int],
		original_size: Tuple[int, int],
	) -> torch.Tensor:
		"""
		Remove padding and upscale masks to the original image size.

		Arguments:
		  masks (torch.Tensor): Batched masks from the mask_decoder,
			in BxCxHxW format.
		  input_size (tuple(int, int)): The size of the image input to the
			model, in (H, W) format. Used to remove padding.
		  original_size (tuple(int, int)): The original size of the image
			before resizing for input to the model, in (H, W) format.

		Returns:
		  (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
			is given by original_size.
		"""
		masks = F.interpolate(
			masks,
			(self.image_encoder.img_size, self.image_encoder.img_size),
			mode="bilinear",
			align_corners=False,
		)
		masks = masks[..., : input_size[0], : input_size[1]]
		masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
		#import pdb;pdb.set_trace()
		return masks

	def preprocess(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Normalize colors
		x = (x - self.pixel_mean) / self.pixel_std

		# Pad
		h, w = x.shape[-2:]
		padh = self.image_encoder.img_size - h
		padw = self.image_encoder.img_size - w
		x = F.pad(x, (0, padw, 0, padh))
		return x
	
	def preprocess_test(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Normalize colors
		x = (x - (self.pixel_mean.unsqueeze(0)).cuda()) / self.pixel_std.unsqueeze(0).cuda()

		# Pad
		h, w = x.shape[-2:]
		padh = self.image_encoder.img_size - h
		padw = self.image_encoder.img_size - w
		x = F.pad(x, (0, padw, 0, padh))
		return x

	def preprocess_change(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize pixel values and pad to a square input."""
		# Normalize colors
		x = (x - (self.pixel_mean).cuda()) / self.pixel_std.cuda()

		# Pad
		# h, w = x.shape[-2:]
		# padh = self.image_encoder.img_size - h
		# padw = self.image_encoder.img_size - w
		# x = F.pad(x, (0, padw, 0, padh))
		return x
