# Tests to verify Dinov2 Featurizer based on custom Dinov2 class method works as intended
import torch
from omegaconf import OmegaConf
from EAGLE.src_EAGLE.modules import DinoFeaturizer, DinoV2Featurizer
from EAGLE.src_EAGLE.utils import visualize_attention

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
assert featv1.shape[-1] * featurizerv1.patch_size == featv2.shape[-1] * featurizerv2.patch_size

# passing in an actual image and visualizing how the feature maps look to verify that everything is correct now
# featurizerv1.model
# featurizerv2.model

class ResizeAndPad:
    def __init__(self, target_size, multiple):
        self.target_size = target_size
        self.multiple = multiple
    def __call__(self, img):
        # Resize the image
        img = transforms.Resize(self.target_size)(img)
        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple
        # Apply padding
        img = transforms.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)
        return img

image_dimension = 448

target_size = (image_dimension, image_dimension)

data_transforms = transforms.Compose([
    ResizeAndPad(target_size, 14),
    transforms.CenterCrop(image_dimension),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dinov2 = featurizerv2.model
dinov2.eval()
dinov2.to(device)

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"

response = requests.get(image_url)

if response.status_code == 200:
    # Open the image
    original_image = Image.open(BytesIO(response.content))
    # Display the image
    original_image.save("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/cat.jpg")
else:
    print(f"Failed to download the image. Status code: {response.status_code}")

