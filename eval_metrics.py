import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DatasetFromHdf5
from model import build_model

try:
    from datasets.split_config import VAL_DATASET
except Exception:
    VAL_DATASET = "datasets/val_dataset.h5"


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model(ckpt_path, device, model_name="bt-net"):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("net", checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    num_channel = int(state_dict["refine.2.weight"].shape[0])
    input_channels = int(state_dict["embedding.weight"].shape[1])
    model = build_model(model_name, num_channel=num_channel,
                        msi_channels=input_channels - num_channel).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def psnr_batch(pred, gt, data_range=1.0, eps=1e-12):
    # pred, gt: [B, C, H, W]
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    max_i = float(data_range)
    psnr = 10.0 * torch.log10((max_i * max_i) / torch.clamp(mse, min=eps))
    return psnr


def sam_batch(pred, gt, eps=1e-12):
    # Spectral Angle Mapper in degrees; average over all pixels for each sample.
    # pred, gt: [B, C, H, W]
    dot = torch.sum(pred * gt, dim=1)
    pred_norm = torch.linalg.norm(pred, dim=1)
    gt_norm = torch.linalg.norm(gt, dim=1)
    cos = dot / torch.clamp(pred_norm * gt_norm, min=eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    angle = torch.acos(cos)
    angle_deg = angle * (180.0 / math.pi)
    return torch.mean(angle_deg, dim=(1, 2))


def cc_batch(pred, gt, eps=1e-12):
    # Average Pearson correlation over spectral bands for each sample.
    # pred, gt: [B, C, H, W]
    b, c, h, w = pred.shape
    pred_f = pred.reshape(b, c, h * w)
    gt_f = gt.reshape(b, c, h * w)

    pred_mean = pred_f.mean(dim=2, keepdim=True)
    gt_mean = gt_f.mean(dim=2, keepdim=True)

    pred_centered = pred_f - pred_mean
    gt_centered = gt_f - gt_mean

    numerator = torch.sum(pred_centered * gt_centered, dim=2)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2, dim=2) * torch.sum(gt_centered ** 2, dim=2)
    )
    corr = numerator / torch.clamp(denominator, min=eps)
    corr = torch.clamp(corr, -1.0, 1.0)
    return corr.mean(dim=1)


def ergas_batch(pred, gt, scale_factor=4.0, eps=1e-12):
    # ERGAS for hyperspectral images, averaged over bands for each sample.
    # pred, gt: [B, C, H, W]
    rmse_band = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(2, 3)))  # [B, C]
    mean_gt_band = torch.mean(gt, dim=(2, 3))  # [B, C]
    # CAVE data are non-negative, but abs also makes the guard well-defined for
    # any accidentally centred input.
    rel = rmse_band / torch.clamp(mean_gt_band.abs(), min=eps)
    ergas = (100.0 / float(scale_factor)) * torch.sqrt(torch.mean(rel ** 2, dim=1))
    return ergas


def _gaussian_window(window_size=11, sigma=1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d


def ssim_batch(pred, gt, data_range=1.0, window_size=11, sigma=1.5, eps=1e-12):
    # Multi-channel SSIM; compute per-sample average over channels and space.
    # pred, gt: [B, C, H, W]
    b, c, h, w = pred.shape
    # Keep the usual 11x11 window for normal CAVE scenes.  The smaller odd
    # fallback only matters for synthetic/tiny inputs and permits valid conv.
    window_size = min(window_size, h, w)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 1:
        raise ValueError(f"SSIM requires non-empty spatial dimensions, got {(h, w)}")
    window_2d = _gaussian_window(
        window_size=window_size,
        sigma=sigma,
        device=pred.device,
        dtype=pred.dtype,
    )
    kernel = window_2d.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    # Valid convolution avoids treating the image exterior as black pixels.
    mu_x = F.conv2d(pred, kernel, padding=0, groups=c)
    mu_y = F.conv2d(gt, kernel, padding=0, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = (F.conv2d(pred * pred, kernel, padding=0, groups=c) - mu_x2).clamp_min(0.0)
    sigma_y2 = (F.conv2d(gt * gt, kernel, padding=0, groups=c) - mu_y2).clamp_min(0.0)
    sigma_xy = F.conv2d(pred * gt, kernel, padding=0, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / torch.clamp(den, min=eps)

    return ssim_map.reshape(b, c, -1).mean(dim=(1, 2))


def _pad_patch(patch, target_h, target_w):
    """Pad a CHW patch on its bottom/right, preferring reflection."""
    pad_h = target_h - patch.shape[-2]
    pad_w = target_w - patch.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Patch is larger than its requested padded size")
    if pad_h == 0 and pad_w == 0:
        return patch
    # Reflection requires each padding amount to be smaller than its input
    # dimension. Replication is a safe fallback for very small edge patches.
    mode = "reflect"
    if ((pad_h and (patch.shape[-2] <= 1 or pad_h >= patch.shape[-2])) or
            (pad_w and (patch.shape[-1] <= 1 or pad_w >= patch.shape[-1]))):
        mode = "replicate"
    return F.pad(patch, (0, pad_w, 0, pad_h), mode=mode)


def _checked_metric(metric, pred, gt, name):
    """Evaluate one single-image metric and return a finite Python float."""
    value = float(metric(pred, gt).item())
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name} value encountered")
    return value


def evaluate(ckpt_path, val_path, batch_size=1, num_workers=0, model_name="bt-net", device_str=None):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Validation set: {val_path}")

    dataset = DatasetFromHdf5(val_path)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(ckpt_path, device, model_name=model_name)

    psnr_all = []
    sam_all = []
    cc_all = []
    ergas_all = []
    ssim_all = []

    # Patch inference to avoid huge attention matrices for large full-scene images.
    PATCH_SIZE = 64
    SCALE = 4
    with torch.no_grad():
        for gt, lrhsi, hrmsi in loader:
            # gt: [B, C, H, W], lrhsi: [B, C, h, w], hrmsi: [B, 3, H, W]
            b, c, H, W = gt.shape
            # Resolve the ambiguous CAVE LR shape (C, 128, 128), which the
            # generic loader can mistake for HWC because 128 is also a known
            # Chikusei band count.  Keep DatasetFromHdf5 itself unchanged.
            if lrhsi.shape[1] != c:
                if lrhsi.shape[2] == c:
                    # DatasetFromHdf5 used HWC->CHW on an already-CHW
                    # (C, H, W) CAVE array, producing (W, C, H). Undo that
                    # exact permutation; swapping only the first two axes
                    # would silently transpose the square spatial image.
                    lrhsi = lrhsi.permute(0, 2, 3, 1).contiguous()
                elif lrhsi.shape[3] == c:
                    lrhsi = lrhsi.permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(
                        f"Cannot align LRHSI shape {tuple(lrhsi.shape)} with {c} GT bands"
                    )

            for bi in range(b):
                gt_img = gt[bi].cpu()
                lr_img = lrhsi[bi].cpu()
                hr_img = hrmsi[bi].cpu()
                pred_full = torch.zeros_like(gt_img, device="cpu")

                # Starts are scale-aligned; the final short patches are padded
                # to the model's 64x64/16x16 input sizes and cropped on writeback.
                for top in range(0, H, PATCH_SIZE):
                    for left in range(0, W, PATCH_SIZE):
                        valid_h = min(PATCH_SIZE, H - top)
                        valid_w = min(PATCH_SIZE, W - left)
                        lr_top, lr_left = top // SCALE, left // SCALE
                        lr_valid_h = min(PATCH_SIZE // SCALE, lr_img.shape[-2] - lr_top)
                        lr_valid_w = min(PATCH_SIZE // SCALE, lr_img.shape[-1] - lr_left)
                        if lr_valid_h <= 0 or lr_valid_w <= 0:
                            raise ValueError("LRHSI does not spatially cover the GT image")

                        hr_patch = hr_img[:, top:top + valid_h, left:left + valid_w]
                        lr_patch = lr_img[:, lr_top:lr_top + lr_valid_h,
                                          lr_left:lr_left + lr_valid_w]
                        hr_patch = _pad_patch(hr_patch, PATCH_SIZE, PATCH_SIZE).unsqueeze(0).to(device)
                        lr_patch = _pad_patch(
                            lr_patch, PATCH_SIZE // SCALE, PATCH_SIZE // SCALE
                        ).unsqueeze(0).to(device)

                        pred_patch, _, _ = model(lr_patch, hr_patch)
                        pred_patch = pred_patch.clamp(0, 1)
                        if pred_patch.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
                            raise ValueError(
                                f"Model patch output has shape {pred_patch.shape[-2:]}, "
                                f"expected {(PATCH_SIZE, PATCH_SIZE)}"
                            )
                        pred_full[:, top:top + valid_h, left:left + valid_w] = (
                            pred_patch[0, :, :valid_h, :valid_w].cpu()
                        )

                if pred_full.shape != gt_img.shape:
                    raise RuntimeError("Reconstructed prediction and GT shapes differ")

                # Metrics are deliberately computed once per reconstructed image.
                # Float64 improves stability for full-cube reductions.
                pred_metric = pred_full.unsqueeze(0).double()
                gt_metric = gt_img.unsqueeze(0).double()
                psnr_all.append(_checked_metric(psnr_batch, pred_metric, gt_metric, "PSNR"))
                sam_all.append(_checked_metric(sam_batch, pred_metric, gt_metric, "SAM"))
                cc_all.append(_checked_metric(cc_batch, pred_metric, gt_metric, "CC"))
                ergas_all.append(_checked_metric(ergas_batch, pred_metric, gt_metric, "ERGAS"))
                ssim_all.append(_checked_metric(ssim_batch, pred_metric, gt_metric, "SSIM"))

    if not psnr_all:
        raise ValueError("Validation set contains no images")
    psnr_mean = float(np.mean(psnr_all))
    sam_mean = float(np.mean(sam_all))
    cc_mean = float(np.mean(cc_all))
    ergas_mean = float(np.mean(ergas_all))
    ssim_mean = float(np.mean(ssim_all))

    print("\nValidation metrics")
    print(f"SAM (deg): {sam_mean:.6f}")
    print(f"CC:        {cc_mean:.6f}")
    print(f"ERGAS:     {ergas_mean:.6f}")
    print(f"SSIM:      {ssim_mean:.6f}")
    print(f"PSNR (dB): {psnr_mean:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BT-Net on a full-image HDF5 dataset")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "bt_net_epoch_200.pth"))
    parser.add_argument("--val", type=str, default=VAL_DATASET)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", type=str, default="bt-net", choices=["bt-net"])
    parser.add_argument("--device", type=str, default=None, help="device string, e.g. 'cpu' or 'cuda'")
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        val_path=args.val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model,
        device_str=args.device,
    )
