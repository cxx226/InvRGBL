import numpy as np
from typing import Literal, Union

import torch
import torch.nn.functional as F
from torch import autograd, nn, Tensor
from skimage.segmentation import slic
from skimage.util import img_as_float


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn.cluster import KMeans


from skimage.segmentation import slic
from skimage.util import img_as_float




def segment_a_with_slic(a, n_segments=100):
    """ Use SLIC to segment `a` into regions. """
    a_np = img_as_float(a.cpu().numpy())  # Normalize
    labels = slic(a_np, n_segments=n_segments, compactness=10)
    return labels

def region_consistency_loss(a, b):
    eps=1e-6
    """ Compute variance of `b` within each connected region of `a`. """
    labels = segment_a_with_slic(a)  # Find connected regions

    min_num = labels.min()
    num = labels.max()
    loss = 0.0
    count = 0  # Keep track of valid regions

    for i in range(min_num, num + 1):
        mask = (labels == i)  # Extract region
        b_region = b[mask]

        if b_region.numel() <= 2 :  # Skip single-pixel regions
            continue

        var_b = torch.var(b_region,dim=0).mean() + eps  # Add small epsilon to avoid NaN
        loss += var_b
        count += 1

    return loss / (count + eps)  # Normalize by number of valid regions

def neighborhood_smoothness_loss(a, b):
    dx = b[:, :-1] - b[:, 1:]  
    dy = b[:-1, :] - b[1:, :]  
    weight_x = torch.exp(- (a[:, :-1] - a[:, 1:]) ** 2)
    weight_y = torch.exp(- (a[:-1, :] - a[1:, :]) ** 2)
    loss = (weight_x * dx**2).mean() + (weight_y * dy**2).mean()
    return loss

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

def calculate_scale_and_shift_loss(prediction, target, mask):
    mask = mask.view(mask.shape[0], mask.shape[1])
    prediction_masked = prediction[mask > 0]  
    target_masked = target[mask > 0]          
    ones = torch.ones((prediction_masked.shape[0], 1), device=prediction.device)  
    A = torch.cat([prediction_masked, ones], dim=1)  
    b = target_masked  

    x = torch.linalg.lstsq(A, b)  
    x = x.solution.detach()

    # 提取 scale 和 shift
    scale = x[:3,:].view(3, 3)  # (1, 1, 3)
    shift = x[3,:].view(1, 3)  # (1, 1, 3)
    
    scaled_prediction = torch.matmul(prediction_masked, scale) + shift  # (H*W, 3)
    loss = torch.abs(scaled_prediction - target_masked).mean()
    return loss



class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = calculate_scale_and_shift(prediction, target, mask[...,0])
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy

def constraint_loss(reference_map, prediction, mask):
    # Simulated reference map: H x W x 3
    H, W, C = reference_map.shape
    # Flatten the spatial dimensions for clustering
    pixels = reference_map.view(-1, C).detach().cpu().numpy()
    # Specify the desired number of groups
    num_clusters = 10
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    # Reshape labels to the original spatial dimensions
    cluster_map = torch.tensor(labels).view(H, W)
    # Visualize the reduced group map (optional)
    print("Cluster Map:")
    print(cluster_map)



def normal_map_smooth_loss(normal_map):
    """
    Computes the smooth loss for a normal map.

    Args:
        normal_map (torch.Tensor): Normal map of shape (B, H, W, 3),
                                   where B is batch size, H and W are height and width,
                                   and 3 is the normal vector (x, y, z).

    Returns:
        torch.Tensor: Smooth loss scalar.
    """
    # Ensure the normal map is normalized
    #normal_map = F.normalize(normal_map, dim=-1)

    # Compute horizontal and vertical gradients
    grad_x = normal_map[:, :, :-1, :] - normal_map[:, :, 1:, :]
    grad_y = normal_map[:, :-1, :, :] - normal_map[:, 1:, :, :]

    # Compute the squared norm of the gradients
    grad_x_loss = torch.sum(grad_x ** 2)
    grad_y_loss = torch.sum(grad_y ** 2)

    # Total loss is the sum of horizontal and vertical gradient losses
    smooth_loss = grad_x_loss + grad_y_loss

    # Normalize by the number of elements for a scalar loss
    num_elements = normal_map.size(0) * normal_map.size(1) * normal_map.size(2)
    return smooth_loss / num_elements

def ref_map_smooth_loss(normal_map, mask):
    grad_x = normal_map[:, :, :-1, :] - normal_map[:, :, 1:, :]
    grad_y = normal_map[:, :-1, :, :] - normal_map[:, 1:, :, :]

    # Compute the squared norm of the gradients
    grad_x_loss = torch.sum(grad_x ** 2)
    grad_y_loss = torch.sum(grad_y ** 2)

    # Total loss is the sum of horizontal and vertical gradient losses
    smooth_loss = grad_x_loss + grad_y_loss

    loss_x = (weight_x * grad_x).mean()
    valid_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]  # Ensure both pixels are valid
    loss_y = ((weight_y * grad_y) * valid_y).sum() / (valid_y.sum() + 1e-6)  # Avoid division by zero
    # Normalize by the number of elements for a scalar loss
    num_elements = normal_map.size(0) * normal_map.size(1) * normal_map.size(2)
    return smooth_loss / num_elements


def save_tensor_as_image(tensor, file_name):
    # Convert tensor from shape (3, H, W) to (H, W, 3)
    # Assuming the tensor values are in range [0, 1] or [0, 255]
    tensor = np.transpose(tensor.detach().cpu().numpy(), (1, 2, 0))
    tensor = (tensor * 255).astype(np.uint8)
    # Convert NumPy array to an image
    image = Image.fromarray(tensor)
    # Save the image
    image.save(file_name)


def calculate_fov(fx, fy, width, height):
    tanfovx = width / (2 * fx)
    tanfovy = height / (2 * fy)
    return tanfovx, tanfovy

def get_proj_matrix(intrinsics,extrinsics):
    eK_mat = torch.eye(4, dtype=intrinsics.dtype, device=intrinsics.device)
    eK_mat[0:3, 0:3] = intrinsics
    return torch.bmm(eK_mat.unsqueeze(0), extrinsics.unsqueeze(0)).squeeze(0)




def reduce(
    loss: Union[torch.Tensor, np.ndarray], 
    mask: Union[torch.Tensor, np.ndarray] = None, 
    reduction: Literal['mean', 'mean_in_mask', 'sum', 'max', 'min', 'none']='mean'):

    if mask is not None:
        if mask.dim() == loss.dim() - 1:
            mask = mask.view(*loss.shape[:-1], 1).expand_as(loss)
        assert loss.dim() == mask.dim(), f"Expects loss.dim={loss.dim()} to be equal to mask.dim()={mask.dim()}"
    
    if reduction == 'mean':
        return loss.mean() if mask is None else (loss * mask).mean()
    elif reduction == 'mean_in_mask':
        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum().clip(1e-5)
    elif reduction == 'sum':
        return loss.sum() if mask is None else (loss * mask).sum()
    elif reduction == 'max':
        return loss.max() if mask is None else loss[mask].max()
    elif reduction == 'min':
        return loss.min() if mask is None else loss[mask].min()
    elif reduction == 'none':
        return loss if mask is None else loss * mask
    else:
        raise RuntimeError(f"Invalid reduction={reduction}")

class SafeBCE(autograd.Function):
    """ Perform clipped BCE without disgarding gradients (preserve clipped gradients)
        This function is equivalent to torch.clip(x, limit), 1-limit) before BCE, 
        BUT with grad existing on those clipped values.
        
    NOTE: pytorch original BCELoss implementation is equivalent to limit = np.exp(-100) here.
        see doc https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    @staticmethod
    def forward(ctx, x, y, limit):
        assert (torch.where(y!=1, y+1, y)==1).all(), u'target must all be {0,1}'
        ln_limit = ctx.ln_limit = np.log(limit)
        # ctx.clip_grad = clip_grad
        
        # NOTE: for example, torch.log(1-torch.tensor([1.000001])) = nan
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        return -torch.where(y==0, torch.log(1-x).clamp_min_(ln_limit), torch.log(x).clamp_min_(ln_limit))
        # return -(y * torch.log(x).clamp_min_(ln_limit) + (1-y)*torch.log(1-x).clamp_min_(ln_limit))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        ln_limit = ctx.ln_limit
        
        # NOTE: for y==0, do not clip small x; for y==1, do not clip small (1-x)
        limit = np.exp(ln_limit)
        # x = torch.clip(x, eclip, 1-eclip)
        x = torch.where(y==0, torch.clip(x, 0, 1-limit), torch.clip(x, limit, 1))
        
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # ttt = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # with open('grad.txt', 'a') as fp:
            #     fp.write(f"{ttt.min().item():.05f}, {ttt.max().item():.05f}\n")
            # NOTE: " * (~(x==y))" so that those already match will not generate gradients.
            grad_x = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # grad_x = ( (1-y)/(1-x) - y/x ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_y = (torch.log(1-x) - torch.log(x)) * grad_output * (~(x==y))
        #---- x, y, limit
        return grad_x, grad_y, None

def safe_binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, limit: float = 0.1, reduction="mean") -> torch.Tensor:
    loss = SafeBCE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    loss = F.binary_cross_entropy(input, target, reduction="none")
    return reduce(loss, None, reduction=reduction)

def normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)

def safe_normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 1e-06, 1.0)

class DepthLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        normalize: bool = True,
        use_inverse_depth: bool = False,
        depth_error_percentile: float = None,
        upper_bound: float = 80,
        reduction: Literal["mean_on_hit", "mean_on_hw", "sum", "none"] = "mean_on_hit",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.use_inverse_depth = use_inverse_depth
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile
        self.reduction = reduction

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        max_depth: float = 80,
        hit_mask: Tensor = None,
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        if hit_mask is not None:
            pred_depth = pred_depth * hit_mask
            gt_depth = gt_depth * hit_mask
        
        # cal valid mask to make sure gt_depth is valid
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)
        
        # normalize depth to (0, 1)
        if self.normalize:
            pred_depth = safe_normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
            gt_depth = safe_normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
        else:
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]
        
        # inverse the depth map (0, 1) -> (1, +inf)
        if self.use_inverse_depth:
            pred_depth = 1./pred_depth
            gt_depth = 1./gt_depth
            
        # cal loss
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        hit_mask: Tensor = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth, self.upper_bound, hit_mask)
        if self.depth_error_percentile is not None:
            # to avoid outliers. not used for now
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]
        
        if self.reduction == "sum":
            depth_error = depth_error.sum()
        elif self.reduction == "none":
            depth_error = depth_error
        elif self.reduction == "mean_on_hit":
            depth_error = depth_error.mean()
        elif self.reduction == "mean_on_hw":
            n = gt_depth.shape[0]*gt_depth.shape[1]
            depth_error = depth_error.sum() / n
        else:
            raise NotImplementedError(f"Unknown reduction method: {self.reduction}")

        return depth_error