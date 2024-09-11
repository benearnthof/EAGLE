from EAGLE.src_EAGLE.modules import *
from EAGLE.src_EAGLE.data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from EAGLE.src_EAGLE.crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from EAGLE.src_EAGLE.train_segmentation_eigen import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
import json
import pytz
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])

def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

@hydra.main(config_path="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    cityscapes_config = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/train_config_cityscapes.yml"
    configpath = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/EAGLE/EAGLE/src_EAGLE/configs/eval_config.yml"
    import yaml
    with open(cityscapes_config) as stream:
        cfg = yaml.safe_load(stream)
    cfg = DictConfig(cfg)

    pytorch_data_dir = cfg.pytorch_data_dir

    model_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/EAGLE/EAGLE_Cityscapes_ViTB8.ckpt"
    model_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/EAGLE/EAGLE_COCO_ViTS8.ckpt"
    for model_path in cfg.model_paths:
        print(str(model_path))
        path_ = str(model_path)

        checkpoint = torch.load(model_path)
        # checkpoint["state_dict"]["CELoss.learned_centroids"].shape
        # torch.Size([28, 512])
        # segmenter = LitUnsupervisedSegmenter(cfg=cfg, n_classes=27)
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path, n_classes=27)

        loader_crop = "center"

        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            mask=True,
            cfg=cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True, collate_fn=flexible_collate)

        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
        else:
            par_model = model.net

        # saved_data = defaultdict(list)
        for i, batch in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                img = batch["img"].cuda()
                label = batch["label"].cuda()
                feats, feats_kk, code1, code_kk = par_model(img)
                feats, feats2_kk, code2, code2_kk = par_model(img.flip(dims=[3]))
                code = (code_kk + code2_kk.flip(dims=[3])) / 2
                code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
                linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
                _, cluster_probs = model.cluster_probe(code, 4, log_probs=True)
                # if cfg.run_crf:
                #     linear_preds = batched_crf(pool, img, linear_probs).argmax(1).cuda()
                #     cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                # else:
                linear_preds = linear_probs.argmax(1)
                cluster_preds = cluster_probs.argmax(1)
                model.test_linear_metrics.update(linear_preds, label)
                model.test_cluster_metrics.update(cluster_preds, label)

        tb_metrics = {
            **model.test_linear_metrics.compute(training=False),
            **model.test_cluster_metrics.compute(training=False),
        }
        
        tb_metrics['assignments'] = tb_metrics['assignments'][-27:]

        print("")
        print(model_path)
        print(tb_metrics)

if __name__ == "__main__":
    prep_args()
    my_app()
