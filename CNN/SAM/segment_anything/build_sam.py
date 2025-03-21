import torch

from functools import partial

from .model import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint
    )

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint
    )

def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint
    )

build_sam = build_sam_vit_b
sam_model_registry = {
    "default": build_sam_vit_b,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def _build_sam(encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024 # Appendix A: "Following standard practices (e.g., [40]), we use an input resolution of 1024×1024..."
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size # Appendix A: "The image embedding is therefore 64×64."
    sam = Sam(
        image_encoder=ImageEncoderViT(
            embed_dim=encoder_embed_dim, # b: 768, l: 1024, h: 1280
            depth=encoder_depth, # b: 12, l: 24, h: 32
            num_heads=encoder_num_heads, # b: 12, l: 16, h: 16
            global_attn_indexes=encoder_global_attn_indexes, # b: [2, 5, 8, 11], l: [5, 11, 17, 23], h: [7, 15, 23, 31]
            img_size=image_size, 
            patch_size=vit_patch_size, # 16
            mlp_ratio=4, 
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
            use_rel_pos=True,
            window_size=14, # Appendix A: "...specifically a ViT-H/16 with 14×14 windowed attention..."
            out_chans=prompt_embed_dim
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim, # Appendix A: "Sparse prompts are mapped to 256dimensional vectorial embeddings..."
            input_image_size=(image_size, image_size),
            image_embedding_size=(image_embedding_size, image_embedding_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder( 
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, # Appendix A.Lightweight mask decoder: "We use a two-layer decoder."
                embedding_dim=prompt_embed_dim, # Appendix A.Lightweight mask decoder: "The transformer uses an embedding dimension of 256."
                mlp_dim=2048, # Appendix A.Lightweight mask decoder: "The transformer MLP blocks have a large internal dimension of 2048"
                num_heads=8, # Appendix A.Lightweight mask decoder: "All attention layers use 8 heads."
            ),
            transformer_dim=prompt_embed_dim, # Appendix A.Lightweight mask decoder: "The transformer uses an embedding dimension of 256."
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
