import torch
from MobileSAMv2.tinyvit.tiny_vit import TinyViT#11000 
from MobileSAMv2.mobilesamv2.modeling.sam import Sam
from MobileSAMv2.mobilesamv2.modeling import ImageEncoderViT, Sam, MaskDecoder, PromptEncoder, TwoWayTransformer


def _build_sam():
	encoder_embed_dim=768
	encoder_depth=12
	encoder_num_heads=12
	encoder_global_attn_indexes=[2, 5, 8, 11]
	prompt_embed_dim = 256
	image_size = 1024
	vit_patch_size = 16
	image_embedding_size = image_size // vit_patch_size
	sam = Sam(
		image_encoder=ImageEncoderViT(
			depth=encoder_depth,
			embed_dim=encoder_embed_dim,
			img_size=image_size,
			mlp_ratio=4,
			norm_layer_eps=1e-6,
			num_heads=encoder_num_heads,
			patch_size=vit_patch_size,
			qkv_bias=True,
			use_rel_pos=True,
			global_attn_indexes=encoder_global_attn_indexes,
			window_size=14,
			out_chans=prompt_embed_dim,
		),
		#See: Unused members. Consider moving out of class
		prompt_encoder=PromptEncoder(
			embed_dim=prompt_embed_dim,
			image_embedding_size=(image_embedding_size, image_embedding_size),
			input_image_size=(image_size, image_size),
			mask_in_chans=16,
		),
		mask_decoder=MaskDecoder(
			num_multimask_outputs=3,
			transformer=TwoWayTransformer(
				depth=2,
				embedding_dim=prompt_embed_dim,
				mlp_dim=2048,
				num_heads=8,
			),
			transformer_dim=prompt_embed_dim,
			iou_head_depth=3,
			iou_head_hidden_dim=256,
		),
		pixel_mean=[123.675, 116.28, 103.53],
		pixel_std=[58.395, 57.12, 57.375],
	)
	sam.eval()
	return sam

def predict_torch(point_coords,point_labels,sam):
	multimask_output = False
	return_logits = True
	if point_coords is not None:
		points = (point_coords, point_labels)
	else:
		points = None
   
	# Embed prompts
	# import pdb;pdb.set_trace()
	sparse_embeddings, dense_embeddings = sam.prompt_encoder.forward(
		points=points,
		boxes=None,
		masks=None,
	)
	#import pdb;pdb.set_trace()
	features = sam.image_encoder(input_image)
	# Predict masks
	low_res_masks, iou_predictions = sam.mask_decoder.forward(
		image_embeddings=features,
		image_pe=prompt_encoder.get_dense_pe(),
		sparse_prompt_embeddings=sparse_embeddings,
		dense_prompt_embeddings=dense_embeddings,
		multimask_output=multimask_output,
	)

	# Upscale the masks to the original image resolution
	# if self.han_size is not None:
	#	  self.original_size=self.han_size
	masks = sam.postprocess_masks(low_res_masks, self.input_size, self.original_size)
	# if self.han_size is not None:
	#	  self.original_size=(1024,1024)
	#import pdb;pdb.set_trace()
	if not return_logits:
		masks = masks > self.model.mask_threshold

	return masks, iou_predictions, low_res_masks

class Sam_predictor(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.sam : Sam = _build_sam()

	def forward(self,x):
		return predict_torch(
		point_coords=torch.tensor([[0,0]]),
		point_labels=torch.tensor([0]),
		sam=self.sam)


if __name__ == "__main__":
	predictor = Sam_predictor()
	predictor.forward(torch.zeros(1024,1024,3))
	torch.jit.script(predictor).save("mobilesam_logits.pt")
