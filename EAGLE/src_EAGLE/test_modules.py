import EAGLE.src_EAGLE.dino.vision_transformer as vits
import torch

# We use a custom fork of dinov2 that adds this functionality:
# https://github.com/facebookresearch/dinov2/compare/main...3cology:dinov2_with_attention_extraction:main
# This means we should now be able to obtain feature maps from dinov2 aswell
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
patch_size = 14
n_register_tokens = 4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# dinov2 = vit_giant2(
#         patch_size=14,
#         img_size=600,
#         init_values=1.0,
#         num_register_tokens=n_register_tokens,
#         block_chunks=0
# )

img_size = 224

configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/vitg14.yaml"
from dinov2.models import build_model_from_cfg
import yaml
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config

default_config = dinov2_default_config
cfg = OmegaConf.load(configpath)
cfg = OmegaConf.merge(default_config, cfg)

cfg.student["block_chunks"] = 0
cfg.student["num_register_tokens"] = 4
cfg.crops.global_crops_size = 518

dinov2 = build_model_from_cfg(cfg, only_teacher=True)[0]


model_type = "dinov2_vitg14_reg"
state_dict = torch.hub.load('facebookresearch/dinov2', f"{model_type}").state_dict()

model = torch.hub.load('facebookresearch/dinov2', f"{model_type}")

dinov2.load_state_dict(state_dict)

# size mismatch for pos_embed
# checkpoint:   [1, 1370, 1536]
# hull:        [1, 257, 1536]
# middle dimension = num_patches + num_tokens
# num_patches = self.patch_embed.num_patches
# num_tokens = 1
# num_patches = 1369

patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

image_HW = (518, 518)
patch_HW = (patch_size, patch_size)
patch_grid_size = (
    image_HW[0] // patch_HW[0],
    image_HW[1] // patch_HW[1],
)

num_patches = patch_grid_size[0] * patch_grid_size[1]





for p in dinov2.parameters():
    p.requires_grad = False

dinov2.to(device)
dinov2.eval()


# loading a single 3x224x224 image
img = torch.ones([1, 3, 224, 224])
img = img.cuda()


# forward of dinov1 featurizer uses the last three attention layer representations like so:
# stepping forward through dinov1 and dinov2 to verify everything has the correct shape

dinov1.eval()

def collate_features(img, patch_size, encoder, k=3):
    """
    This function wraps the fancy feature extraction EAGLE does with Dino to a generic function that can 
    also be used to extract image features and their attention maps from DINOv2, as long as DINOv2 implements
    the `get_intermediate_feat` method.

    Args:
        img: Image to be encoded, height and width must be evenly divisible by `patch_size`
        patch_size: The patch size of the DINO vision transformer used to encode the image.
        encoder: Some vision Transformer that implements the `get_intermediate_feat` method.
        k: integer that specifies how many feature maps should be collected. EAGLE uses 3 by default
    """
    assert (img.shape[2] % patch_size == 0 and img.shape[3] % patch_size == 0)
    feat_h, feat_w = img.shape[2] // patch_size, img.shape[3] // patch_size
    # pass image through frozen encoder
    with torch.no_grad():
        feat_all, attn_all, qkv_all = model.get_intermediate_feat(img, n=1)
        image_features, image_features_kk = [], []
        for index in range(k):
            # we're interested in the k last feature maps so we loop over the reversed lists
            feat, attn, qkv = feat_all[::-1][index], attn_all[::-1][index], qkv_all[::-1][index]
            img_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            img_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], attn.shape[1], feat_h, feat_w, -1)
            B, H, I, J, D = img_k.shape
            img_kk = img_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            # Will yield features in order: high, mid, low
            image_features.append(img_feat)
            image_features_kk.append(img_kk)
        # now reorder to low, mid, high and concatenate to torch tensors
        image_feat = torch.cat(image_features[::-1], dim=1)
        image_kk = torch.cat(image_features_kk[::-1], dim=1)
                
        # class feat is just high level features
        if return_class_feat:
            return feat_all[-1][:, :1, :].reshape(feat_all[-1].shape[0], 1, 1, -1).permute(0, 3, 1, 2)

    if proj_type is not None:
        with torch.no_grad():
            code = cluster1(dropout(image_feat))
        code_kk = cluster1(dropout(image_kk))
        if proj_type == "nonlinear":
            code += cluster2(dropout(image_feat))
            code_kk += cluster2(dropout(image_kk))
    else:
        code = image_feat
        code_kk = image_kk

    if cfg.dropout:
        return dropout(image_feat), dropout(image_kk), code, code_kk
    else:
        return image_feat, image_kk, code, code_kk