import torch
from EAGLE.src_EAGLE.modules import DinoFeaturizer, DinoV2Featurizer

from omegaconf import OmegaConf

v1_configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/train_config_cityscapes.yml"
v2_configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/train_cityscapes_dinov2.yml"

v1_cfg = OmegaConf.load(v1_configpath)
v2_cfg = OmegaConf.load(v2_configpath)

featurizerv1 = DinoFeaturizer(dim=v1_cfg.dim, cfg=v1_cfg)
featurizerv2 = DinoV2Featurizer(dim=v2_cfg.dim, cfg=v2_cfg)

featurizerv1.cuda()
featurizerv2.cuda()

# loading a single 3x224x224 image
img = torch.ones([1, 3, 224, 224])
img = img.cuda()

with torch.no_grad():
    featv1, kkv1, codev1, code_kkv1 = featurizerv1(img)

with torch.no_grad():
    featv2, kkv2, codev2, code_kkv2 = featurizerv2(img)

assert featv1.shape[1] == featv2.shape[1]
assert codev1.shape[1] == codev2.shape[1]

# guarantee that patch dimensions add up
assert featv1.shape[-1] * 8 == featv2.shape[-1] * 14
