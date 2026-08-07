import argparse
import math
import os

import numpy as np
import torch
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
    rel = rmse_band / torch.clamp(mean_gt_band, min=eps)
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
    b, c, _, _ = pred.shape
    window_2d = _gaussian_window(
        window_size=window_size,
        sigma=sigma,
        device=pred.device,
        dtype=pred.dtype,
    )
    kernel = window_2d.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    mu_x = torch.nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=c)
    mu_y = torch.nn.functional.conv2d(gt, kernel, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv2d(pred * pred, kernel, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(gt * gt, kernel, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(pred * gt, kernel, padding=window_size // 2, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / torch.clamp(den, min=eps)

    return ssim_map.reshape(b, c, -1).mean(dim=(1, 2))


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

    psnr_sse = 0.0
    psnr_numel = 0
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

            # If image is larger than PATCH_SIZE, run non-overlapping patch inference on CPU tensors
            if H > PATCH_SIZE or W > PATCH_SIZE:
                # iterate per-sample in batch
                for bi in range(b):
                    gt_img = gt[bi]  # CPU tensor
                    lr_img = lrhsi[bi]
                    hr_img = hrmsi[bi]

                    n_y = (H - PATCH_SIZE) // PATCH_SIZE + 1
                    n_x = (W - PATCH_SIZE) // PATCH_SIZE + 1
                    for iy in range(n_y):
                        for ix in range(n_x):
                            top = iy * PATCH_SIZE
                            left = ix * PATCH_SIZE
                            bottom = top + PATCH_SIZE
                            right = left + PATCH_SIZE

                            gt_patch = gt_img[:, top:bottom, left:right].unsqueeze(0).to(device)
                            lr_top = top // SCALE
                            lr_left = left // SCALE
                            lr_patch = lr_img[:, lr_top: lr_top + (PATCH_SIZE // SCALE), lr_left: lr_left + (PATCH_SIZE // SCALE)].unsqueeze(0).to(device)
                            hr_patch = hr_img[:, top:bottom, left:right].unsqueeze(0).to(device)

                            pred, _, _ = model(lr_patch, hr_patch)
                            pred = pred.clamp(0, 1)

                            sam_vals = sam_batch(pred, gt_patch)
                            cc_vals = cc_batch(pred, gt_patch)
                            ergas_vals = ergas_batch(pred, gt_patch)
                            ssim_vals = ssim_batch(pred, gt_patch)

                            psnr_sse += float(torch.sum((pred - gt_patch) ** 2).item())
                            psnr_numel += pred.numel()
                            sam_all.append(sam_vals.cpu().numpy())
                            cc_all.append(cc_vals.cpu().numpy())
                            ergas_all.append(ergas_vals.cpu().numpy())
                            ssim_all.append(ssim_vals.cpu().numpy())
            else:
                gt = gt.to(device, non_blocking=True)
                lrhsi = lrhsi.to(device, non_blocking=True)
                hrmsi = hrmsi.to(device, non_blocking=True)

                pred, _, _ = model(lrhsi, hrmsi)
                pred = pred.clamp(0, 1)

                sam_vals = sam_batch(pred, gt)
                cc_vals = cc_batch(pred, gt)
                ergas_vals = ergas_batch(pred, gt)
                ssim_vals = ssim_batch(pred, gt)

                psnr_sse += float(torch.sum((pred - gt) ** 2).item())
                psnr_numel += pred.numel()
                sam_all.append(sam_vals.cpu().numpy())
                cc_all.append(cc_vals.cpu().numpy())
                ergas_all.append(ergas_vals.cpu().numpy())
                ssim_all.append(ssim_vals.cpu().numpy())

    mse = max(psnr_sse / psnr_numel, 1e-12)
    psnr_mean = float(10.0 * math.log10(1.0 / mse))
    sam_mean = float(np.concatenate(sam_all).mean())
    cc_mean = float(np.concatenate(cc_all).mean())
    ergas_mean = float(np.concatenate(ergas_all).mean())
    ssim_mean = float(np.concatenate(ssim_all).mean())

    print("\nValidation metrics")
    print(f"SAM (deg): {sam_mean:.6f}")
    print(f"CC:        {cc_mean:.6f}")
    print(f"ERGAS:     {ergas_mean:.6f}")
    print(f"SSIM:      {ssim_mean:.6f}")
    print(f"PSNR (dB): {psnr_mean:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BT-Net on validation set")
    parser.add_argument("--ckpt", type=str, default=os.path.join("Trained_model", "175.pth"))
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
