# Tests to verify Dinov2 Featurizer based on custom Dinov2 class method works as intended
import requests
import torch
from omegaconf import OmegaConf
from PIL import Image
from io import BytesIO
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
imgv2 = torch.ones([1, 3, 392, 392])

img = img.cuda()
imgv2= imgv2.cuda()

with torch.no_grad():
    featv1, kkv1, codev1, code_kkv1 = featurizerv1(img)

with torch.no_grad():
    featv2, kkv2, codev2, code_kkv2 = featurizerv2(img)

with torch.no_grad():
    feat392, kk392, code392, code_kk392, = featurizerv2(imgv2)

assert featv1.shape[1] == featv2.shape[1]
assert codev1.shape[1] == codev2.shape[1]
assert feat392.shape == featv1.shape
assert kk392.shape == kkv1.shape

# # guarantee that patch dimensions add up
assert featv1.shape[-1] * featurizerv1.patch_size == featv2.shape[-1] * featurizerv2.patch_size

# we need to adress the training loop to deal with the new feature shapes
# feats, feats_kk, code, code_kk
featsv1, feats_kkv1, codev1, code_kkv1 = featurizerv1(img)
# feats_pos, feats_pos_kk, code_pos, code_pos_kk = self.net(img_pos) # same shape
# feats, feats_kk, code, code_kk
featsv2, feats_kkv2, codev2, code_kkv2 = featurizerv2(img)





def test_corr_loss():
    feats_pos_rev1 = feats_kkv1.reshape(featsv1.shape[0], featsv1.shape[1], -1).permute(0,2,1)
    feats_pos_rev2 = feats_kkv2.reshape(featsv2.shape[0], featsv2.shape[1], -1).permute(0,2,1)

    # We need to initialize EigenLoss with the reduced number of channels in mind
    # we get 224/patch_size * 224/patch_size = 16*16 = 256
    # instead of 224/8 * 224/8 = 28*28 = 784

    # Investigating Correspondence loss and grid sampling
    # IMG: torch.Size([32, 3, 224, 224])
    # POS: torch.Size([32, 3, 224, 224])
    # AUG: torch.Size([32, 3, 224, 224])
    # All feature tensors have the same input shape => they also have the same output shape

    # Correspondence Loss: 
    CorrespondenceLoss.forward(# Dinov1                 Dinov2
        feats_kk,               # [32, 2304, 28, 28]    [32, 2304, 16, 16]      orig_feats
        feats_pos_kk,           # [32, 2304, 28, 28]    [32, 2304, 16, 16]      orig_feats_pos
        feats_pos_aug_kk,       # [32, 2304, 28, 28]    [32, 2304, 16, 16]      orig_feats_pos_aug
        code_kk,                # [32, 512, 28, 28]     [32, 512, 16, 16]       orig_code
        code_pos_kk,            # [32, 512, 28, 28]     [32, 512, 16, 16]       orig_code_pos
        code_pos_aug_kk         # [32, 512, 28, 28]     [32, 512, 16, 16]       orig_code_pos_aug
    )

    orig_feats = feats_kkv2
    orig_feats_pos = feats_kkv2
    orig_feats_pos_aug = feats_kkv2
    orig_code = code_kkv2
    orig_code_pos = code_kkv2
    orig_code_pos_aug = code_kkv2

    # 28 is hardcoded as cfg.feature_samples
    coord_shape_v1 = [orig_feats.shape[0], 28, 28, 2]
    coord_shape_v2 = [orig_feats.shape[0], 28, 28, 2]

    coords1 = torch.rand(coord_shape_v1, device=orig_feats.device) * 2 - 1
    coords2 = torch.rand(coord_shape_v1, device=orig_feats.device) * 2 - 1
    coords3 = torch.rand(coord_shape_v1, device=orig_feats.device) * 2 - 1

    import torch.nn.functional as F

    def sample(t, coords): # the permute does nothing, just swaps 28 x 28 channels with eachother
        return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

    feats = sample(orig_feats, coords1)
    code = sample(orig_code, coords1)

    feats_pos = sample(orig_feats_pos, coords2)
    code_pos = sample(orig_code_pos, coords2)

    feats_pos_aug = sample(orig_feats_pos_aug, coords3)
    code_pos_aug = sample(orig_code_pos_aug, coords3)

    # For Dinov1 grid sampled features and codes match the shapes of the input tensors
    # feats: [32, 2304, 28, 28]
    # code: [32, 512, 28, 28]
    # For Dinov2 we obtain more samples than we put data in:
    # feats: [32, 2304, 28, 28] with given [1, 2304, 16, 16]
    # code: [32, 512, 28, 28] with given [1, 512, 16, 16]

    # lets try training with cfg.feature_samples <= 16
    pass


code_pos_kk, code_pos_aug_kk, code_pos = code_kkv2, code_kkv2, code_kkv2 
feats_pos_kk, feats_pos_aug_kk, feats_pos = feats_kkv2, feats_kkv2, feats_kkv2
# [784, 512] # [256, 512]
code_pos_z, code_pos_aug_z = code_pos_kk.permute(0,2,3,1).reshape(-1, 512), code_pos_aug_kk.permute(0,2,3,1).reshape(-1, 512)

from torch import nn
import torch.nn.functional as F
project_head = nn.Linear(512, 512).cuda()
if True:
    code_pos_z = project_head(code_pos_z) # [784*batch, 512]
    code_pos_aug_z = project_head(code_pos_aug_z) #[25088, 70]
    code_pos_aug_z = F.normalize(code_pos_aug_z, dim=1)
    code_pos_z = F.normalize(code_pos_z, dim=1)

feats_pos_reshaped = feats_pos_kk.view(feats_pos.shape[0], feats_pos.shape[1], -1)
corr_feats_pos = torch.matmul(feats_pos_reshaped.transpose(2, 1), feats_pos_reshaped)
corr_feats_pos = F.normalize(corr_feats_pos, dim=1)

feats_pos_aug_reshaped = feats_pos_reshaped
corr_feats_pos_aug = torch.matmul(feats_pos_aug_reshaped.transpose(2, 1), feats_pos_aug_reshaped)
corr_feats_pos_aug = F.normalize(corr_feats_pos_aug, dim=1)

loss = 0    

# neg_samples > 0
# 2. Eigenloss
# pos 
# [1, 784, 2304] # [1, 256, 2304] # 256 because we only get 16x16 patches of dim 14 at 224 resolution
feats_pos_re = feats_pos_kk.reshape(feats_pos.shape[0], feats_pos.shape[1], -1).permute(0,2,1)
# [1, 784, 512] # [1, 256, 512]
code_pos_re = code_pos_kk.reshape(code_pos.shape[0], code_pos.shape[1], -1).permute(0,2,1) 
# adjust for new dimensions
# v1: [b, 3, 224, 224], [b, 784, 2304], [b, 784, 512], [1, 784, 784]
# v2: [b, 3, 224, 224], [b, 256, 2304], [b, 256, 512], [1, 256, 256]
eigen_loss_fn = EigenLoss(arch="dinov2")
eigenvectors =  eigen_loss_fn(img, feats_pos_re, code_pos_re, corr_feats_pos, None, neg_sample=5)

# v1: [1, 784, 4] v2: [1, 256, 4]

eigenvectors = eigenvectors[:, :, 1:].reshape(eigenvectors.shape[0], feats_pos.shape[-1], feats_pos.shape[-1], -1).permute(0,3,1,2)
# eigenvectorsv2 = eigenvectors[:, :, 1:].reshape(eigenvectors.shape[0], feats_pos.shape[-1], feats_pos.shape[-1], -1).permute(0,3,1,2)

# v1: [1, 3, 28, 28], v2: [1, 3, 16, 16]

# 3 = dim, 28 = n_classes
train_cluster_probe_eigen = ClusterLookup(3, 28).cuda()
# x, alpha, log_probs=True
cluster_eigen_loss, cluster_eigen_probs = train_cluster_probe_eigen(eigenvectors, 1, log_probs = True)

# v1: [1, 28, 28, 28] v2: [1, 28, 16, 16]

cluster_eigen_probs = cluster_eigen_probs.argmax(1)
cluster_eigen_probs
# v1: [1, 28, 28], v2: [1, 16, 16]
# Up to this point everything seems correct. The only difference is that because of our patch size the 
# output mask is very coarse. We could try to train on val resolution instead.
# 336 resolution would yield 24x24 tensor
# 392 resolution would yield 28x28 tensor
# passing in an image of only torch ones we obtain large clusters in both v1 and v2
# of course cluster probe projects randomly so this is what we expect. 

# # pos_aug
# Pos aug is copy paste of the code above, the images and features all have the exact same shapes.

# Last possible avenue for errors: CELoss calculations
# takes in
# v1: [784, 512], [784, 512], [1, 28, 28], [1, 784, 784]
ce_loss = newLocalGlobalInfoNCE()
# ce_loss.forward(S1, S2, segmentation_map, similarity_matrix)
local_pos_mid_loss = ce_loss(code_pos_z, code_pos_aug_z, cluster_eigen_probs, corr_feats_pos)
local_pos_loss = local_pos_mid_loss 

# Rest of the loss calculation is done on detached tensors for linear and cluster probes that are not optimized during training
S1, S2, segmentation_map, similarity_matrix = code_pos_z, code_pos_aug_z, cluster_eigen_probs, corr_feats_pos











# passing in an actual image and visualizing how the feature maps look to verify that everything is correct now
# featurizerv1.model
# featurizerv2.model

image_url = "https://i.natgeofe.com/k/e7ba8001-23ac-457f-aedb-abd5f2fdda62/moms5_4x3.png"
# image_url = "https://media.istockphoto.com/id/1467126728/photo/modern-scandinavian-and-japandi-style-bedroom-interior-design-with-bed-white-color-wooden.jpg?s=612x612&w=0&k=20&c=oa94MeFcLIs6l4hJQGztLbWe3BGOH9LtvLebnUXgUus="

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
newsize = [int(x*0.25) for x in size]
image = original_image.resize(newsize)

visualize_attention(image, model=featurizerv1.model, gamma=0.75, arch="dinov1")
# Is the fundamental error in the featurizer that we use the features from the wrong end of the blocks? 
# TODO: doublecheck with v1 featurizer
# TODO: check qkv, we dont pass in attention we pass in qkv
# Order and values of qkv: 
img = torch.ones([1, 3, 392, 392]).cuda()
feat_h, feat_w = img.shape[2] // 14, img.shape[3] // 14
feat, attns, qkv = featurizerv2.model.get_intermediate_feat(img)
feat, attn, qkv = feat[0], attns[0], qkv[0]
# [1, 789, 768], [1, 12, 789, 789], [3, 1, 12, 789, 64]
img_feat = feat[:, 5:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
# [1, 768, 28, 28]
img_k = qkv[1, :, :, 5:, :].reshape(feat.shape[0], attn.shape[1], feat_h, feat_w, -1)
B, H, I, J, D = img_k.shape
img_kk = img_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
# [1, 768, 28, 28]



































































# Eigen Loss
import torch
from EAGLE.src_EAGLE.utils import *
import torch.nn.functional as F
# from sklearn.cluster import KMeans, MiniBatchKMeans
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from kmeans_pytorch import kmeans, kmeans_predict
import math
from scipy.spatial.distance import cdist
import torch.nn.functional as F

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
    device = image.device
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h
    r, g, b = r.to(device), g.to(device), b.to(device)
    x = torch.repeat_interleave(torch.linspace(0, 1, w).to(device), h)
    y = torch.cat([torch.linspace(0, 1, h)] * w).to(device)
    i, j = [], [] 
    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = torch.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
             axis=1,
             out=torch.zeros((n, 5), dtype=torch.float32).to(device)
        ).to(device) 
        distances, neighbors = knn(f.cpu().numpy(), f.cpu().numpy(), k=k)
        distances = torch.tensor(distances)
        neighbors = torch.tensor(neighbors)
        i.append(torch.repeat_interleave(torch.arange(n), k))
        j.append(neighbors.view(-1))
    ij = torch.cat(i + j)
    ji = torch.cat(j + i)
    coo_data = torch.ones(2 * sum(n_neighbors) * n)
    W = scipy.sparse.csr_matrix((coo_data.cpu().numpy(), (ij.cpu().numpy(), ji.cpu().numpy())), (n, n))
    return torch.tensor(W.toarray())


def rw_affinity(image, sigma=0.033, radius=1):
    """Computes a random walk-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.laplacian.rw_laplacian import _rw_laplacian
    except:
        raise ImportError(
            'Please install pymatting to compute RW affinity matrices:\n'
            'pip3 install pymatting'
        )
    h, w = image.shape[:2]
    n = h * w
    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)
    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))
    return W

def multi_seg(img, eigenvalues, eigenvectors, adaptive = True, num_eigenvectors: int = 1_000_000):
    adaptive = False
    non_adaptive_num_segments = 27
    if adaptive: 
        indices_by_gap = np.argsort(np.diff(eigenvalues))[::-1]
        index_largest_gap = indices_by_gap[indices_by_gap != 0][0]  # remove zero and take the biggest
        n_clusters = index_largest_gap + 1
        print(f'Number of clusters: {n_clusters}')
    else: # of class
        n_clusters = non_adaptive_num_segments
    eigenvectors = eigenvectors[:, :, 1:] # take non-constant eigenvectors
    segmap_list = []
    for i in range(eigenvectors.shape[0]):
        C, H, W = img[i].shape
        H_patch, W_patch = H // 8, W // 8
        eigenvector_batch = eigenvectors[i]
        clusters, cluster_centers = kmeans(X=eigenvector_batch, distance = 'euclidean', num_clusters=n_clusters, device=eigenvector_batch.device)
        if clusters.cpu().numpy().size == H_patch * W_patch:
            segmap = clusters.reshape(H_patch, W_patch)
        elif clusters.cpu().numpy().size == H_patch * W_patch * 4:
            segmap = clusters.reshape(H_patch * 2, W_patch * 2)
        elif clusters.cpu().numpy().size == (H_patch * W_patch - 1):
            clusters = np.append(clusters, 0)
            segmap = clusters.reshape(H_patch, W_patch)
        else:
            raise ValueError()
        segmap_list.append(segmap)
    return torch.stack(segmap_list)
    
def visualize_segmap(segmap_list):
    for segmap in segmap_list:
        segmap_uint8 = segmap.to(torch.uint8)
        output_file = f'./img/image_segmap.png'
        colormap = [[0,0,0], [120,0,0], [0, 150, 0],[240, 230, 140],[176, 48, 96],[48, 176, 96],[103, 255, 255],[238, 186, 243],[119, 159, 176],[122, 186, 220],[96, 204, 96],[220, 247, 164],[60, 60, 60],[220, 216, 20],[196, 58, 250],[120, 18, 134],[12, 48, 255],[236, 13, 176],[0, 118, 14],[165, 42, 42],[160, 32, 240],[56, 192, 255],[184, 237, 194],[180, 231, 250],[299, 300, 0], [100, 200, 94],[39,203, 123]]
        colormap = np.array(colormap)
        out_conf = np.zeros((segmap_uint8.shape[0], segmap_uint8.shape[1],3))
        for x in range(segmap_uint8.shape[0]):
            for y in range(segmap_uint8.shape[1]):
                out_conf[x,y,:] = colormap[segmap_uint8[x,y]]
        import imageio
        imageio.imsave(output_file, out_conf.astype(np.uint8))

def attention_map(image_feat):
    ax = sns.heatmap(image_feat[1])
    plt.title('feat')
    plt.savefig(f'laplacian_1.png')
    plt.close()
    ax = sns.heatmap(image_feat[2])
    plt.title('feat')
    plt.savefig(f'laplacian_2.png')
    plt.close()
    return

def get_diagonal(W, threshold: float=1e-12):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W, dtype=torch.float32)
    D = torch.matmul(W, torch.ones(W.shape[1], dtype=W.dtype).to(W.device))
    D[D < threshold] = 1.0  # Prevent division by zero.
    D_diag = torch.diag(D)
    return D_diag


class EigenLoss(nn.Module):
    def __init__(self, arch):
        super(EigenLoss, self).__init__()
        self.eigen_cluster = 4
        self.arch = arch
    def normalized_laplacian(self, L, D):
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        D_inv_sqrt = D_inv_sqrt.diagonal(dim1=-2, dim2=-1)
        D_inv_sqrt_diag = torch.diag_embed(D_inv_sqrt)
        L_norm = torch.bmm(D_inv_sqrt_diag, torch.bmm(L, D_inv_sqrt_diag))
        return L_norm
    def batch_trace(self,tensor):
        diagonals = torch.diagonal(tensor, dim1=1, dim2=2)
        trace_values = torch.sum(diagonals, dim=1)
        return trace_values
    def eigen(self, lap, K):
        eigenvalues_all, eigenvectors_all = torch.linalg.eigh(lap, UPLO='U')
        eigenvalues = eigenvalues_all[:, :K]
        eigenvectors = eigenvectors_all[:, :, :K]    
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()
        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  
                eigenvectors[k] = 0 - eigenvectors[k]
        return eigenvalues, eigenvectors
    def pairwise_distances(self, x, y=None, w=None):
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)
    def compute_color_affinity(self, image, sigma_c=30, radius=1):
        H, W, _ = image.shape
        N = H * W
        pixels = image.view(-1, 3).float() / 255.0 
        color_distances = self.pairwise_distances(pixels)
        W_color = torch.exp(-color_distances**2 / (2 * sigma_c**2))
        y, x = np.mgrid[:H, :W]
        coords = np.c_[y.ravel(), x.ravel()]
        spatial_distances = cdist(coords, coords, metric='euclidean')
        W_color[spatial_distances > radius] = 0
        return W_color
    def laplacian(self, adj, W):
        adj = (adj * (adj > 0))
        max_values = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        adj = adj / max_values 
        w_combs = W.to(adj.device)
        max_values = w_combs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        w_combs = w_combs / max_values
        W_comb = w_combs + adj
        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap
    def color_affinity(self, img):
        color = []
        for img_ in img:
            normalized_image = img_ / 255.0 
            pixels = normalized_image.view(-1, 3)
            color_distances = torch.cdist(pixels, pixels, p=2.0)
            color_affinity = torch.exp(-color_distances ** 2 / (2 * (0.1 ** 2)))  
            color.append(color_affinity)
        aff_color = torch.stack(color)
        return aff_color
    def laplacian_matrix(self, img, image_feat, image_color_lambda=0, which_color_matrix='knn'):
        threshold_at_zero = True
        if threshold_at_zero:
            image_feat = (image_feat * (image_feat > 0))
        max_values = image_feat.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        image_feat = image_feat / max_values 
        if image_color_lambda > 0:
            img_resize = F.interpolate(img, size=(28, 28), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            max_values = img_resize.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            img_norm = img_resize / max_values
            affinity_matrices = []
            for img_norm_b in img_norm:
                if which_color_matrix == 'knn':
                    W_lr = knn_affinity(img_norm_b)
                elif which_color_matrix == 'rw':
                    W_lr = rw_affinity(img_norm_b)
                affinity_matrices.append(W_lr)
            W_color = torch.stack(affinity_matrices).to(image_feat.device)
            W_comb = image_feat + W_color * image_color_lambda
        else:
            W_comb = image_feat
        D_comb = torch.stack([get_diagonal(w_comb) for w_comb in W_comb])
        L_comb = D_comb - W_comb
        lap = self.normalized_laplacian(L_comb, D_comb)
        return lap
    def lalign(self, img, Y, code, adj, adj_code, code_neg_torch, neg_sample=5):
        if code_neg_torch is None:
            if self.arch == "dinov2":
                # FIXME: replace magic numbers with imgsize/patch_size
                print("Interpolating image from:", img.shape)
                img = F.interpolate(img, size=(16, 16), mode='bilinear', align_corners=False).permute(0,2,3,1)
                print("To:", img.shape)
            elif Y.shape[1] == 196:
                img = F.interpolate(img, size=(14, 14), mode='bilinear', align_corners=False).permute(0,2,3,1)
            else:
                img = F.interpolate(img, size=(28, 28), mode='bilinear', align_corners=False).permute(0,2,3,1)
            print("Computing Color Affinity", img.shape)
            color_W = self.color_affinity(img)
            nor_adj_lap = self.laplacian(adj_code, color_W) 
            nor_adj_lap_detach = torch.clone(nor_adj_lap.detach()) 
            eigenvalues, eigenvectors = self.eigen(nor_adj_lap_detach, K=self.eigen_cluster) 
            return eigenvectors
        else:
            adj_lap = self.laplacian_matrix(img, adj, image_color_lambda=0.1) 
            max_values = code.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            code_norm = code / max_values 
            code_neg_torch = code_neg_torch.reshape(code_neg_torch.shape[0],code_neg_torch.shape[1],code_neg_torch.shape[2], -1).permute(0,1,3,2) # [5, B, 121, 512]
            return eigenvectors
    def forward(self, img, feat, code, corr_feats_pos, code_neg_torch, neg_sample=5):
        feat = F.normalize(feat, p=2, dim=-1)
        adj = torch.bmm(feat, feat.transpose(1,2))
        adj_code = torch.bmm(code, code.transpose(1,2))
        if code_neg_torch is None:
            eigenvectors = self.lalign(img, feat, code, adj, adj_code, code_neg_torch, neg_sample)
            return eigenvectors 
        else:
            eigenvectors, pos, neg = self.lalign(img, feat, code, adj, adj_code, code_neg_torch, neg_sample)
            return eigenvectors, pos, neg


class newLocalGlobalInfoNCE(nn.Module):
    def __init__(self):
        super(newLocalGlobalInfoNCE, self).__init__()
        # cityscapes
        self.learned_centroids = nn.Parameter(torch.randn(27+1, 512))
        self.prototypes = torch.randn(27 + 1, 512, requires_grad=True)
    def compute_centroid(self, features, labels):
        unique_labels = torch.unique(labels)
        centroids = []
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            pairwise_dist = torch.cdist(class_features, class_features)
            prototype = class_features[torch.argmin(pairwise_dist.sum(0))]
            new_prototypes = self.prototypes.clone()  
            new_prototypes[label] =  prototype 
            self.prototypes = new_prototypes
            centroids.append(prototype)      
        return torch.stack(centroids)
    def forward(self, S1, S2, segmentation_map, similarity_matrix):
        label_smoothing_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        batch_size, patch_size = segmentation_map.size(0), segmentation_map.size(1)
        segmentation_map = segmentation_map.reshape(-1)
        S1_centroids = self.compute_centroid(S1, segmentation_map)
        local_logits = torch.mm(S1, S1_centroids.t()) / 0.07
        global_logits = torch.mm(S2, S1_centroids.t()) / 0.07
        mask = (segmentation_map.unsqueeze(1) == torch.unique(segmentation_map)) 
        labels = mask.float().argmax(dim=1)
        local_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        global_weights = (similarity_matrix.mean(dim=2).reshape(-1)) * 1.0
        # we could just do regular cross entropy i guess? why do we do label smoothing on cityscapes?
        # if self.cfg.dataset_name=='cityscapes':
        #     local_loss = label_smoothing_criterion(local_logits, labels)
        #     global_loss = label_smoothing_criterion(global_logits, labels)
        # else:
        local_loss = F.cross_entropy(local_logits, labels, reduction='none')
        global_loss = F.cross_entropy(global_logits, labels, reduction='none')
        local_loss = (local_loss * local_weights).mean()
        global_loss = (global_loss * global_weights).mean()
        total_loss = ((1-0.7) * local_loss + 0.7 * global_loss) / 2
        return total_loss