import EAGLE.src_EAGLE.dino.vision_transformer as vits
import torch

# loading dinov1
patch_size = 8
arch = "vit_base"
dinov1 = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
for p in dinov1.parameters():
    p.requires_grad = False

dinov1.eval().cuda()

url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
dinov1.load_state_dict(state_dict, strict=True)
dinov1.eval().cuda()

# We use a custom fork of dinov2 that adds this functionality:
# https://github.com/facebookresearch/dinov2/compare/main...3cology:dinov2_with_attention_extraction:main
# This means we should now be able to obtain feature maps from dinov2 aswell
from dinov2.models.vision_transformer import vit_small, vit_base, vit_large
patch_size = 14
n_register_tokens = 4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dinov2 = vit_base(
        patch_size=14,
        img_size=526,
        init_values=1.0,
        num_register_tokens=n_register_tokens,
        block_chunks=0
)
model_type = "dinov2_vitb14_reg"
state_dict = torch.hub.load('facebookresearch/dinov2', f"{model_type}").state_dict()
dinov2.load_state_dict(state_dict)

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
with torch.no_grad():
    patch_size_v1 = 8
    patch_size_v2 = 14
    assert (img.shape[2] % patch_size_v1 == 0)
    assert (img.shape[3] % patch_size_v1 == 0)
    assert (img.shape[2] % patch_size_v2 == 0)
    assert (img.shape[3] % patch_size_v2 == 0)


    feat_h_v1 = img.shape[2] // patch_size_v1
    feat_w_v1 = img.shape[3] // patch_size_v1

    feat_h_v2 = img.shape[2] // patch_size_v2
    feat_w_v2 = img.shape[3] // patch_size_v2


    feat_all, attn_all, qkv_all = dinov1.get_intermediate_feat(img, n=1)
    feat_all_v2, attn_all_v2, qkv_all_v2 = dinov2.get_intermediate_feat(img, n=1)
    # all lists are still of length 12 because both models have 12 blocks each
    feat_all[-1].shape, feat_all_v2[-1].shape 
    attn_all[-1].shape, attn_all_v2[-1].shape
    qkv_all[-1].shape, qkv_all_v2[-1].shape
    # they still have slightly different shapes but we're on the right track at least
    

    # high level
    feat, attn, qkv = feat_all[-1], attn_all[-1], qkv_all[-1]
    feat_v2, attn_v2, qkv_v2 = feat_all_v2[-1], attn_all_v2[-1], qkv_all_v2[-1]
    
    image_feat_high = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
    image_k_high = qkv[1, :, :, 1:, :].reshape(feat.shape[0], attn.shape[1], feat_h, feat_w, -1)
    B, H, I, J, D = image_k_high.shape
    image_kk_high = image_k_high.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
                
    # mid level
    feat_mid, attn_mid, qkv_mid = feat_all[-2], attn_all[-2], qkv_all[-2]
    
    image_feat_mid = feat_mid[:, 1:, :].reshape(feat_mid.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
    image_k_mid = qkv_mid[1, :, :, 1:, :].reshape(feat_mid.shape[0], attn.shape[1], feat_h, feat_w, -1)
    B, H, I, J, D = image_k_mid.shape
    image_kk_mid = image_k_mid.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
    
    # low level
    feat_low, attn_low, qkv_low = feat_all[-3], attn_all[-3], qkv_all[-3]
    
    image_feat_low = feat_low[:, 1:, :].reshape(feat_low.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
    image_k_low = qkv_low[1, :, :, 1:, :].reshape(feat_low.shape[0], attn.shape[1], feat_h, feat_w, -1)
    B, H, I, J, D = image_k_low.shape
    image_kk_low = image_k_low.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
    
    image_feat = torch.cat([image_feat_low, image_feat_mid, image_feat_high], dim=1)
    image_kk  = torch.cat([image_kk_low, image_kk_mid, image_kk_high], dim=1)
    
    
    if return_class_feat:
        return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

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