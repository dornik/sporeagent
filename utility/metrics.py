from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
import environment.transformations as tra
import dataset.dataset as dtst
# Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet

import config as cfg


def compute_stats(pred_transforms, data_loader):
    metrics_for_iter = defaultdict(list)
    num_processed = 0
    for data in tqdm(data_loader, leave=False):
        if isinstance(data, list):
            data = dtst.collate_data(data)

        dict_all_to_device(data, pred_transforms.device)

        batch_size = data['points_src'].shape[0]
        cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size]
        metrics = compute_metrics(data, cur_pred_transforms)
        for k in metrics:
            metrics_for_iter[k].append(metrics[k])
        num_processed += batch_size
    summary_metrics = summarize_metrics(metrics_for_iter)

    return metrics_for_iter, summary_metrics


def compute_metrics(data, pred_transforms):
    # points_src = data['points_src'][..., :3]
    points_ref = data['points_ref'][..., :3]
    if 'points_raw' in data:
        points_raw = data['points_raw'][..., :3]
    else:
        points_raw = points_ref
    gt_transforms = data['transform_gt']  # src->ref
    igt_transforms = torch.eye(4, device=pred_transforms.device).repeat(gt_transforms.shape[0], 1, 1)
    igt_transforms[:, :3, :3] = gt_transforms[:, :3, :3].transpose(2, 1)
    igt_transforms[:, :3, 3] = -(igt_transforms[:, :3, :3] @ gt_transforms[:, :3, 3].view(-1, 3, 1)).view(-1, 3)

    ref_clean = points_raw
    residual_transforms = pred_transforms @ igt_transforms
    src_clean = (residual_transforms[:, :3, :3] @ points_raw.transpose(2, 1)).transpose(2, 1)\
                + residual_transforms[:, :3, 3][:, None, :]

    # ADD/ADI
    src_diameters = torch.sqrt(tra.square_distance(src_clean, src_clean).max(dim=-1)[0]).max(dim=-1)[0]
    dist_add = torch.norm(src_clean - ref_clean, p=2, dim=-1).mean(dim=1) / src_diameters
    dist_adi = torch.sqrt(tra.square_distance(ref_clean, src_clean)).min(dim=-1)[0].mean(dim=-1) / src_diameters

    metrics = {
        'add': dist_add.cpu().numpy(),
        'adi': dist_adi.cpu().numpy()
    }

    if cfg.USE_SYMMETRY:
        symmetries = data['symmetries']
        igt_transforms = igt_transforms[:, None, ...] @ symmetries  # ref->src
        pred_transforms = pred_transforms[:, None, ...]  # src->ref

        residual_transforms = pred_transforms @ igt_transforms
        src_clean = (residual_transforms[..., :3, :3] @ points_raw[:, None, ...].transpose(-1, -2)).transpose(-1, -2) \
                    + residual_transforms[..., :3, 3][..., None, :]

        # MSSD -- max over points, min over symmetries
        dist_mssd = torch.norm(src_clean - ref_clean[:, None, ...], p=2, dim=-1).max(dim=-1)[0].min(dim=-1)[0] / src_diameters
        metrics['mssd'] = dist_mssd.cpu().numpy()

    return metrics


def summarize_metrics(metrics):
    summarized = {}
    for k in metrics:
        metrics[k] = np.hstack(metrics[k])
        if k.startswith('ad'):
            summarized[k] = np.mean(metrics[k])
            step_precision = 1e-3
            max_precision = 0.1
            precisions = np.arange(step_precision, max_precision + step_precision, step_precision)
            recalls = np.array([(metrics[k] <= precision).mean() for precision in precisions])
            # integrate area under precision-recall curve -- normalize to 100% (= area given by 1.0 * max_precision)
            summarized[k + '_auc10'] = (recalls * step_precision).sum()/max_precision
        summarized[k] = np.mean(metrics[k])

    return summarized


def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device
    via RPMNet """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)
            if tensor_dict[k].dtype == torch.double:
                tensor_dict[k] = tensor_dict[k].float()
        if isinstance(tensor_dict[k], dict):
            dict_all_to_device(tensor_dict[k], device)
