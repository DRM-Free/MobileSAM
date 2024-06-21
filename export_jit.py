import torch
import numpy as np
import os
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide
from MobileSAMv2.mobilesamv2.build_sam import build_sam_vit_b
import sys
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
		self.sam = build_sam_vit_b()
		self.sam.to(device='cpu')
		self.predictor = SamPredictor(self.sam)
		self.image_size = image_size
	def forward(self, x):
		return torch.as_tensor([[[0]*1024]*720]*3, device='cpu')
		self.predictor.set_torch_image(x, (self.image_size))
		masks, scores, logits = self.predictor.predict(
		point_coords=[[0,0]],
		point_labels=[0],
		multimask_output=True,
	)
		return logits

model = Model(image_size, checkpoint, model_type)
#model_trace = torch.jit.trace(model, input_image_torch).save("mobilesam_logits.pt")
model_script = torch.jit.script(model).save("mobilesam_logits.pt")
