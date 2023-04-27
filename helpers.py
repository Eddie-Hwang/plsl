import importlib
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import check_shape
from modules.constant import (FACE_INDICES, LEFT_HAND_INDICES,
                                  POSE_INDICES, RIGHT_HAND_INDICES)


def remove_noisy_frames(X, threshold=120.):
    """
    Remove noisy frames based on the Euclidean distance between consecutive frames.

    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints
        threshold: threshold value for the Euclidean distance, frames with distance above the threshold are removed

    Returns:
        X_clean: torch tensor of shape (num_clean_frames, num_joints, 3) representing the cleaned 3D keypoints
    """
    num_frames, num_joints, _ = X.shape
    X_diff = torch.diff(X, dim=0)  # compute the difference between consecutive frames
    distances = torch.norm(X_diff, dim=-1)  # compute the Euclidean distance
    distances = torch.mean(distances, dim=-1)

    mask = torch.ones(num_frames, dtype=torch.bool)  # initialize a mask to keep all frames
    
    mask[1:] = distances <= threshold  # set to False all frames with distance above the threshold
    X_clean = X[mask]  # apply the mask to the input keypoints
    
    return X_clean


def normalize_keypoints_with_resolution(keypoints, width=800, height=900):
    normalized_keypoints = keypoints.clone()

    if keypoints.shape[2] == 2:  # 2D keypoints
        scale = torch.tensor([width, height], device=keypoints.device)
    elif keypoints.shape[2] == 3:  # 3D keypoints
        scale = torch.tensor([width, height, (width + height) * 0.5], device=keypoints.device)
    else:
        raise ValueError("Invalid keypoints shape. The last dimension should be 2 or 3.")

    return normalized_keypoints / scale


def center_keypoints(keypoints, joint_idx=1):
    """
    Center the keypoints around a specific joint.
    
    Args:
        keypoints (torch.Tensor): Tensor of shape (batch_size, nframes, njoints, nfeat) or (nframes, njoints, nfeat) containing the keypoints.
        joint_idx (int): Index of the joint to center the keypoints around.
        
    Returns:
        torch.Tensor: Tensor of the same shape as input with the keypoints centered around the specified joint.
    """
    if len(keypoints.shape) == 4:
        batch_size, nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, :, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
    elif len(keypoints.shape) == 3:
        nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(nframes, 1, nfeats)
    else:
        raise ValueError("Input keypoints tensor must have either 3 or 4 dimensions")

    return centered_keypoints


def normalize(X):
    """
    Normalize 3D keypoints using min-max normalization.

    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints

    Returns:
        X_norm: torch tensor of shape (num_frames, num_joints, 3) representing the normalized 3D keypoints
    """
    T, n, d = X.shape
    X = X.reshape(T*n, d)
    X_min = torch.min(X, dim=0)[0]
    X_max = torch.max(X, dim=0)[0]
    X_norm = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
    X_norm = X_norm.reshape(T, n, d)
    
    return X_norm


def normalize_skeleton_3d(X, resize_factor=None):
    def distance_3d(x1, y1, z1, x2, y2, z2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 3)  # neck

    if resize_factor is None:
        neck_height = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2],
                                  X[:, 0, 0], X[:, 0, 1], X[:, 0, 2]).float()
        shoulder_length = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 2, 0], X[:, 2, 1], X[:, 2, 2]) + \
                          distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 5, 0], X[:, 5, 1], X[:, 5, 2])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor
        normalized_X[:, i, 2] = (X[:, i, 2] - anchor_pt[:, 2]) / resize_factor

    return normalized_X


def normalize_skeleton_2d(X, resize_factor=None):
    def distance_2d(x1, y1, x2, y2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 2)  # neck

    if resize_factor is None:
        neck_height = distance_2d(X[:, 1, 0], X[:, 1, 1],
                                  X[:, 0, 0], X[:, 0, 1]).float()
        shoulder_length = distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 2, 0], X[:, 2, 1]) + \
                          distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 5, 0], X[:, 5, 1])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor

    return normalized_X


def scaling_keypoints(keypoints, width=800, height=900):
    """
    Unnormalize 3D keypoints by scaling the x, y, and z coordinates based on the width, height, and depth
    of the original image.

    Args:
        keypoints: a PyTorch tensor of shape (batch_size, num_frames, num_joints, 3) or (num_frames, num_joints, 3)
            representing 3D keypoints
        width: width of the original image
        height: height of the original image

    Returns:
        unnormalized_keypoints: a PyTorch tensor of the same shape as keypoints, with each element
        unnormalized based on the width, height, and depth of the original image
    """
    if keypoints.dim() == 4:
        batch_size, num_frames, num_joints, _ = keypoints.size()
        keypoints_slice = keypoints
    elif keypoints.dim() == 3:
        num_frames, num_joints, _ = keypoints.size()
        batch_size = 1
        keypoints_slice = keypoints.unsqueeze(0)
    else:
        raise ValueError(f"Invalid number of dimensions in input tensor: {keypoints.dim()}.")

    depth = (width + height) * 0.5

    # Scale the x, y, and z coordinates based on the width, height, and depth of the original image
    unnormalized_keypoints = keypoints_slice.clone()
    unnormalized_keypoints[..., 0] *= width
    unnormalized_keypoints[..., 1] *= height
    unnormalized_keypoints[..., 2] *= depth

    if keypoints.dim() == 3:
        unnormalized_keypoints = unnormalized_keypoints.squeeze(0)

    return unnormalized_keypoints


def center_keypoints(keypoints, joint_idx=1):
    """
    Center the keypoints around a specific joint.
    
    Args:
        keypoints (torch.Tensor): Tensor of shape (batch_size, nframes, njoints, nfeat) or (nframes, njoints, nfeat) containing the keypoints.
        joint_idx (int): Index of the joint to center the keypoints around.
        
    Returns:
        torch.Tensor: Tensor of the same shape as input with the keypoints centered around the specified joint.
    """
    if len(keypoints.shape) == 4:
        batch_size, nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, :, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
    elif len(keypoints.shape) == 3:
        nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(nframes, 1, nfeats)
    else:
        raise ValueError("Input keypoints tensor must have either 3 or 4 dimensions")

    return centered_keypoints


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key "target" to instatiate.')
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def extract(a, t, x_shape):
    """
    Extract values from a tensor a at indices t.

    Args:
        a: A tensor of shape (batch_size, dim1, dim2, ..., dimN) from which to extract values.
        t: A tensor of shape (batch_size, index1, index2, ..., indexN) indicating the indices to extract values from a.
        x_shape: The shape of the original tensor, used to properly reshape the extracted values.

    Returns:
        A tensor of shape (batch_size, dim2, dim3, ..., dimN, 1) containing the extracted values.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


def is_perfect_square(n):
    """
    Returns True if n is a perfect square, False otherwise.
    """
    root = math.isqrt(n) # integer square root
    return root * root == n


def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.bool()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def add_noise(x, noise_std=0.5):
    # noise = torch.randn_like(x) * noise_std * x.mean()
    noise = torch.randn_like(x) * noise_std
    return x + noise


def calculate_mean_stddev(tensor):
    """
    Calculates the mean and standard deviation of a PyTorch tensor with shape [batch_size, nframes, njoints, nfeats].
    Returns a tuple (mean, stddev).
    """
    flattened_tensor = tensor.flatten()
    mean = torch.mean(flattened_tensor)
    stddev = torch.std(flattened_tensor)
    return mean, stddev


def calculate_adaptive_keypoints_weight(inputs, alpha=1.0):
    assert len(inputs.shape) == 4
    device = inputs.device
    num_keypoints, num_feats = inputs.shape[2], inputs.shape[3]
    weights = torch.ones(num_keypoints, device=device)

    mean_pose_movement = torch.mean(inputs[:, :, POSE_INDICES], dim=2, keepdim=True)
    expected_finger_positions = inputs[:, :, LEFT_HAND_INDICES + RIGHT_HAND_INDICES] - mean_pose_movement 
    residual_finger_movemnet = inputs[:, :, LEFT_HAND_INDICES + RIGHT_HAND_INDICES] - expected_finger_positions

    pose_var = torch.var(inputs[:, :, POSE_INDICES])
    finger_var = torch.var(residual_finger_movemnet)
    face_var = torch.var(inputs[:, :, FACE_INDICES])
    total_var = pose_var + finger_var + face_var

    pose_contribution = pose_var / total_var
    hand_contribution = finger_var / total_var
    face_contribution = face_var / total_var

    hand_weight = alpha * (pose_contribution / hand_contribution)
    face_weight = alpha * (pose_contribution / face_contribution)
    
    for ind in LEFT_HAND_INDICES + RIGHT_HAND_INDICES:
        weights[ind] = hand_weight

    for ind in FACE_INDICES:
        weights[ind] = face_weight
    
    weights = rearrange(weights, "j -> 1 1 j 1")
    weights = weights.repeat(1, 1, 1, num_feats)
    
    return weights


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_random_sampled_sequence(input_tensor, window_size):
    check_shape(input_tensor, "f v c")
    nframes, njoints, nfeats = input_tensor.shape

    if nframes <= window_size:
        return input_tensor

    max_start_idx = nframes - window_size
    start_index = torch.randint(low=0, high=max_start_idx, size=(1,))
    output_tensor = input_tensor[start_index:start_index+window_size]
    
    return output_tensor