# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder


class Sam(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        '''
        Args:
            image_encoder (ImageEncoderViT): The image encoder.
            prompt_encoder (PromptEncoder): The prompt encoder.
            mask_decoder (MaskDecoder): The mask decoder.
            pixel_mean (List[float]): The mean of the pixel values.
            pixel_std (List[float]): The standard deviation of the pixel values.
        '''
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def forward(self, batched_input: List[Dict[str, Any]], multimask_output: bool) -> List[Dict[str, Any]]:
        '''
        Args:
            batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be excluded if it is not present.
              +) 'image': The image as a torch tensor in 3xHxW format, already transformed for input to the model.
              +) 'original_size': (tuple(int, int)) The original size of the image before transformation, as (H, W).
              +) 'point_coords': (torch.Tensor) Batched point prompts for this image, with shape BxNx2. 
                                Already transformed to the input frame of the model.
              +) 'point_labels': (torch.Tensor) Batched labels for point prompts, with shape BxN.
              +) 'boxes': (torch.Tensor) Batched box inputs, with shape Bx4. 
                         Already transformed to the input frame of the model.
              +) 'mask_inputs': (torch.Tensor) Batched mask inputs to the model, in the form Bx1xHxW.
            multimask_output (bool): Whether the model should predict multiple disambiguating masks, or return a single mask.
        '''
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding  in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record["boxes"] if "boxes" in image_record else None,
                masks=image_record["mask_inputs"] if "mask_inputs" in image_record else None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_masks": low_res_masks,
            })
        return outputs
    
    def postprocess_masks(
        self,
        low_res_masks: torch.Tensor,
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
        masks = masks[..., :input_size[0], :input_size[1]] # Remove padding
        masks = F.interpolate(
            masks,
            original_size, # Upscale to the original image size
            mode="bilinear",
            align_corners=False,
        )
        return masks
    
    def process(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x
