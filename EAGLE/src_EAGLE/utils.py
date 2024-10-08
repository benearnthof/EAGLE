import os
import io
import collections
import requests
from os.path import join
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.optimize import linear_sum_assignment

string_classes = str
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchmetrics import Metric
from torchvision import models, transforms
from torchvision import transforms as T
from torch.utils.tensorboard.summary import hparams
from pytorch_lightning.loggers import WandbLogger
import wandb


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = T.ToTensor()(image)
    # writer.add_image(name, image, step)
    # writer.log_image(key=name, images=image)
    writer({name: [wandb.Image(image)]}, step = step)
    plt.clf()
    plt.close()


@torch.jit.script
def shuffle(x):
    return x[torch.randperm(x.shape[0])]


def add_hparams_fixed(writer, hparam_dict, metric_dict, global_step):
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_step)


@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


def one_hot_feats(labels, n_classes):
    return F.one_hot(labels, n_classes).permute(0, 3, 1, 2).to(torch.float32)


def load_model(model_type, data_dir):
    if model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download",
                          model_file)
        model_weights = torch.load(model_file)
        # model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
        #                          'model' in name}
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    model.eval()
    model.cuda()
    return model


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self, training=False):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            # print(self.assignments)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)
        
        if training:
            metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                        self.prefix + "Accuracy": opc.item()}
            return {k: 100 * v for k, v in metric_dict.items()}
    
        else:    
            metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                    self.prefix + "Accuracy": opc.item(),
                    "assignments": (self.assignments[1]).tolist()}
            return {k: 100 * v for k, v in metric_dict.items()}

        # metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
        #            self.prefix + "Accuracy": opc.item(),
        #            "assignments": (self.assignments[1]).tolist()}
        # return {k: 100 * v for k, v in metric_dict.items()}


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class ResizeAndPad:
    "Helper for Model Introspection"
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

def visualize_attention(
    image,
    model, 
    output_dir="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/",
    alpha_max_value=1.0, 
    gamma=0.75,
    arch="dinov2"
    ):
    """
    Overlays the last attention map onto a target image to help visualize if the model loaded correctly.
    """
    if arch == "dinov2":
        patch_size = model.patch_size
    else:
        patch_size = 8
    target_size = (
        min(image.size) - min(image.size) % patch_size, 
        min(image.size) - min(image.size) % patch_size
    )
    print(target_size)
    # image_dimension = 448
    # target_size = (image_dimension, image_dimension)
    data_transforms = transforms.Compose([
        ResizeAndPad(target_size, patch_size),
        transforms.CenterCrop(target_size[0]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    (original_w, original_h) = image.size
    img = data_transforms(image)
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    img = img.unsqueeze(0)
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        _, attns, _ = model.get_intermediate_feat(img)
        # get_intermediate_features returns returns in order from first block to last
        for i, attention in enumerate(attns):
            number_of_heads = attention.shape[1]
            if arch == "dinov2":
                attention = attention[0, :, 0, 1 + model.num_register_tokens:].reshape(number_of_heads, -1)
            elif arch == "dinov1":
                attention = attention[0, :, 0, 1:].reshape(number_of_heads, -1)
            # resolution of attention from transformer tokens
            attention = attention.reshape(number_of_heads, w_featmap, h_featmap)
            # resize to original image resolution
            attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu()
            # sum all attention across 12 different heads to get one map of attention across entire image
            attention = torch.sum(attention, dim=0)
            # interpolate attention map back into original image dimensions
            attention_of_image = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
            attention_of_image = attention_of_image.squeeze()

            # Normalize image_metric to the range [0, 1]
            image_metric = attention_of_image.numpy()
            normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)
            # Apply the Reds colormap
            reds = plt.cm.Reds(normalized_metric)
            # Create the alpha channel
            # Apply gamma transformation to enhance lower values
            enhanced_metric = np.power(normalized_metric, gamma)
            # Create the alpha channel with enhanced visibility for lower values
            alpha_channel = enhanced_metric * alpha_max_value
            # Add the alpha channel to the RGB data
            rgba_mask = np.zeros((image_metric.shape[0], image_metric.shape[1], 4))
            rgba_mask[..., :3] = reds[..., :3]  # RGB
            rgba_mask[..., 3] = alpha_channel  # Alpha
            # Convert the numpy array to PIL Image
            rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))
            # Save the image
            rgba_image.save(f"{output_dir}attn_map{i}.png")
            # Load the attention mask with PIL
            attention_mask_image = Image.open(f"{output_dir}attn_map{i}.png")
            # Ensure both images are in the same mode
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            # Overlay the second image onto the first image
            # The second image must be the same size as the first image
            image.paste(attention_mask_image, (0, 0), attention_mask_image)
            # Save combined image
            image.save(f"{output_dir}image_with_attn_map{i}.png")
            print("Saved Attention Mask and Image successfully.")
