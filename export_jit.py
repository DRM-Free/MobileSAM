import torch
import numpy as np
import os
#from mobile_sam import sam_model_registry, SamPredictor
from MobileSAMv2.mobilesamv2 import SamPredictor, sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide
from MobileSAMv2.mobilesamv2.build_sam import build_sam_vit_b
import sys

from MobileSAMv2.mobilesamv2.modeling import ImageEncoderViT, Sam, MaskDecoder, PromptEncoder, TwoWayTransformer
from MobileSAMv2.mobilesamv2.modeling.image_encoder import Block
from MobileSAMv2.mobilesamv2.modeling.sam import Sam
sys.path.append("MobileSAMv2")
checkpoint = 'weights/mobile_sam.pt'
model_type = 'vit_t'
quantize = True
output_names = ['output']

# Target image size is 1024x720
image_size = (1024, 720)
"""
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
transform = ResizeLongestSide(sam.image_encoder.img_size)

image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
input_image = transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image, device='cpu')
input_image_torch = input_image_torch.permute(
	2, 0, 1).contiguous()[None, :, :, :]
"""

class Model(torch.nn.Module):
	def __init__(self, image_size, checkpoint, model_type):
		super().__init__()

		self.sam : Sam = build_sam_vit_b()
		self.sam.to(device='cpu')
		self.predictor : SamPredictor = SamPredictor(self.sam)
		self.image_size = image_size
	def forward(self, x)->torch.Tensor:
		self.predictor.set_torch_image(x, (self.image_size))
		#return torch.as_tensor([[[0]*1024]*720]*3, device='cpu')
		return self.predictor.predict_torch(
		point_coords=torch.tensor([[0,0]]),
		point_labels=torch.tensor([0]))
		"""
		masks, scores, logits = self.predictor.predict_torch(
		point_coords=torch.tensor([[0,0]]),
		point_labels=torch.tensor([0]),
	)
		"""
		return logits

## Script intermediate functions for debug
#torch.jit.script(Block(1,1))
#torch.jit.script(ImageEncoderViT())
#torch.jit.script(build_sam_vit_b())

prompt_encoder = PromptEncoder(
	embed_dim=4,
	image_embedding_size=(4, 4),
	input_image_size=(4, 4),
	mask_in_chans=16,
)

torch.jit.script(prompt_encoder)

mask_decoder = MaskDecoder(
	num_multimask_outputs=3,
	transformer=TwoWayTransformer(
		depth=2,
		embedding_dim=4,
		mlp_dim=2048,
		num_heads=2,
	),
	transformer_dim=4,
	iou_head_depth=3,
	iou_head_hidden_dim=256,
)

image_encoder = ImageEncoderViT()
sam = Sam(image_encoder,prompt_encoder,mask_decoder)
torch.jit.script(Sam(image_encoder,prompt_encoder,mask_decoder))
torch.jit.script(SamPredictor(sam))
## End script intermediate functions

model = Model(image_size, checkpoint, model_type)
#model_trace = torch.jit.trace(model, input_image_torch).save("mobilesam_logits.pt")
model_script = torch.jit.script(model).save("mobilesam_logits.pt")
#model_script = torch.jit.trace(model).save("mobilesam_logits.pt")
