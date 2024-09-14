# Tests to verify Dinov2 Featurizer based on custom Dinov2 class method works as intended
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from io import BytesIO
from EAGLE.src_EAGLE.modules import DinoFeaturizer, DinoV2Featurizer
from EAGLE.src_EAGLE.utils import visualize_attention

#v1_configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/train_config_cityscapes.yml"
v2_configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/train_cityscapes_dinov2.yml"

#v1_cfg = OmegaConf.load(v1_configpath)
v2_cfg = OmegaConf.load(v2_configpath)

#featurizerv1 = DinoFeaturizer(dim=v1_cfg.dim, cfg=v1_cfg)
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

# # guarantee that patch dimensions add up
assert featv1.shape[-1] * featurizerv1.patch_size == featv2.shape[-1] * featurizerv2.patch_size

# passing in an actual image and visualizing how the feature maps look to verify that everything is correct now
# featurizerv1.model
# featurizerv2.model

image_url = "https://i.natgeofe.com/k/e7ba8001-23ac-457f-aedb-abd5f2fdda62/moms5_4x3.png"
image_url = "https://media.istockphoto.com/id/1467126728/photo/modern-scandinavian-and-japandi-style-bedroom-interior-design-with-bed-white-color-wooden.jpg?s=612x612&w=0&k=20&c=oa94MeFcLIs6l4hJQGztLbWe3BGOH9LtvLebnUXgUus="

response = requests.get(image_url)

if response.status_code == 200:
    # Open the image
    original_image = Image.open(BytesIO(response.content))
    # Display the image
    original_image.save("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/cat.jpg")
else:
    print(f"Failed to download the image. Status code: {response.status_code}")

# for very large images we need to resize
size = original_image.size
newsize = [int(x*1) for x in size]
image = original_image.resize(newsize)

visualize_attention(image, model=featurizerv2.model, gamma=0.75)